from flask import Flask, request, jsonify
from flask_cors import CORS
from ..backtest import Backtest
from ..strategies.single.rsi import RSI
from ..strategies.gpu_optimized.NP.rsi_adx_np import RSI_ADX_NP
from ..strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
from ..strategies.gpu_optimized.NP.rsi_bollinger_np import BollingerBands_RSI
from ..strategies.gpu_optimized.NP.bollinger_vwap import BollingerBands_VWAP
from ..strategies.gpu_optimized.NP.macd_atr_np import MACD_ATR
from ..strategies.single.rsi import RSI
from ..strategies.single.bollinger import BollingerBands
from ..strategies.single.macd import MACD
from ..strategies.single.adx import ADX
from ..strategies.single.vwap import Vwap
from ..strategies.single.atr import ATR
from ..strategies.single.stochastic import StochasticOscillator
from ..strategies.single.efratio import EFratio
from ..strategies.single.williams import WilliamsR
from ..strategies.single.kama import Kama

import pandas as pd
import base64
import plotly.io as pio
import logging
import datetime
import jwt
import core.database_interaction as database_interaction
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
# logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)


@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        response = app.response_class()
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.status_code = 200
        return response

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled Exception: {str(e)}")
    return jsonify({"status": "error", "message": "Internal server error"}), 500

def to_png(fig):
    try:
        img_bytes = pio.to_image(fig, format="png")
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        logging.error(f"Error converting graph to PNG: {str(e)}")
        app.logger.error(f"Error converting graph to PNG: {str(e)}")
        raise ValueError("Graph conversion failed.")

def convert_results_to_json(results):
    """Convert pandas Timestamp and Timedelta objects to strings,
       handling NaT values gracefully."""
    for key, value in results.items():
        if pd.isna(value):
            results[key] = None  # or you can set to an empty string ''
        elif isinstance(value, pd.Timestamp):
            try:
                results[key] = value.isoformat()
            except Exception:
                results[key] = str(value)
        elif isinstance(value, pd.Timedelta):
            results[key] = str(value)
    return results


@app.route('/api/backtest', methods=['POST'])
def backtest():
    params = request.json
    app.logger.info(f"Received params: {params}")

    strategy_mapping = { 
        "RSI": RSI,
        "RSI_ADX_NP": RSI_ADX_NP,
        "RSI_ADX_GPU": RSI_ADX_GPU,
        "BollingerBands_RSI": BollingerBands_RSI,
        "BollingerBands_VWAP": BollingerBands_VWAP,
        "MACD_ATR": MACD_ATR,
        "BollingerBands": BollingerBands,
        "MACD": MACD,
        "ADX": ADX,
        "Vwap": Vwap,
        "ATR": ATR,
        "StochasticOscillator": StochasticOscillator,
        "EFratio": EFratio,
        "WilliamsR": WilliamsR,
        "Kama": Kama
    }

    # Ensure the strategy exists
    if params["strategy_obj"] not in strategy_mapping:
        return jsonify({"status": "error", "message": "Invalid strategy"}), 400

    strategy_obj = strategy_mapping[params["strategy_obj"]]

    backtest_instance = Backtest()
    try:
        stats, graph_base64 = backtest_instance.run_basic_backtest(
            symbol=[params["symbol"]],
            granularity=params["granularity"],
            strategy_obj=strategy_obj,
            num_days=params["num_days"],
            sizing=params["sizing"],
            best_params=params["best_params"],
            graph_callback=to_png
        )
    except Exception as e:
        app.logger.error(f"Backtest failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

    if stats is None:
        return jsonify({"status": "error", "message": "Backtest returned no stats"}), 500

    json_friendly_stats = convert_results_to_json(stats)

    # Save backtest to the database
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"status": "error", "message": "Token is missing"}), 403

    if token.startswith('Bearer '):
        token = token[7:]
    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        email = decoded['email']
        database_interaction.save_backtest(
            email=email,
            symbol=params["symbol"],
            strategy=params["strategy_obj"],
            result=stats,
            date=datetime.datetime.now().isoformat()  
        )
    except jwt.ExpiredSignatureError:
        return jsonify({"status": "error", "message": "Token has expired"}), 403
    except jwt.InvalidTokenError:
        return jsonify({"status": "error", "message": "Invalid token"}), 403
    except Exception as e:
        app.logger.error(f"Failed to save backtest: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to save backtest"}), 500

    return jsonify({"status": "success", "stats": json_friendly_stats, "graph": graph_base64})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    # Ensure the email is provided
    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400
    
    # Get all users from the database
    users = database_interaction.get_users()
    
    if email in users:
        return jsonify({"status": "error", "message": "User already exists"}), 400
    
    # Encrypt the password before saving it
    encrypted_password = generate_password_hash(password)

    # Perform your registration logic here, save to DB
    
    database_interaction.save_user(email, encrypted_password)
    app.logger.info(f"Registering user: {email}")
    return jsonify({"status": "success", "message": "User registered successfully"}), 201

# Login route
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Ensure the email is provided
    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    # Get all users from the database
    users = database_interaction.get_users()
    
    if email not in users:
        return jsonify({"status": "error", "message": "User not found"}), 404
    
    # Check if the password matches the hashed password in the database
    hashed_password = users[email]
    if not check_password_hash(hashed_password, password):
        return jsonify({"status": "error", "message": "Invalid password"}), 401

    # Authentication successful
    app.logger.info(f"Logging in user: {email}")

    token = jwt.encode(
    {"email": email, "exp": datetime.datetime.now() + datetime.timedelta(hours=1)},
    app.config['SECRET_KEY'],
    algorithm="HS256"
    )
    return jsonify({"status": "success", "token": token}), 200

@app.route('/api/backtests', methods=['GET'])
def get_history():
    token = request.headers.get('Authorization')
    app.logger.info(f"Authorization header: {token}")
    
    if not token or not token.startswith("Bearer "):
        app.logger.error("Token is missing or invalid")
        return jsonify({"status": "error", "message": "Token is missing or invalid"}), 403
    
    token = token.split("Bearer ")[1]
    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        app.logger.info(f"Decoded token: {decoded}")
        email = decoded['email']
        history = database_interaction.get_backtest_history(email)
        return jsonify({"status": "success", "history": history}), 200
    except jwt.ExpiredSignatureError:
        app.logger.error("Token has expired")
        return jsonify({"status": "error", "message": "Token has expired"}), 403
    except jwt.InvalidTokenError as e:
        app.logger.error(f"Invalid token: {e}")
        return jsonify({"status": "error", "message": "Invalid token"}), 403

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

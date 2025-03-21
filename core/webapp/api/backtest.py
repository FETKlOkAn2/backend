# mypy: disable-error-code=import-untyped
# pylint: disable=C0114

from flask import Blueprint, request, jsonify, current_app
import datetime
import jwt
import pandas as pd
import base64
import plotly.io as pio
import core.database_interaction as database_interaction
from core.webapp.services.backtest_service import get_strategy_class, run_backtest

backtest_bp = Blueprint('backtest', __name__)

def to_png(fig):
    """Convert a Plotly figure to a base64-encoded PNG."""
    try:
        img_bytes = pio.to_image(fig, format="png")
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        current_app.logger.error(f"Error converting graph to PNG: {str(e)}")
        raise ValueError("Graph conversion failed.")

def convert_results_to_json(results):
    """
    Convert pandas Timestamp and Timedelta objects to strings,
    handling NaT values gracefully.
    """
    # Create a copy of the results dictionary to avoid modifying the original
    converted_results = results.copy()
    
    for key, value in converted_results.items():
        if pd.isna(value):
            converted_results[key] = None  # Convert NaN/NaT to None
        elif isinstance(value, pd.Timestamp):
            try:
                converted_results[key] = value.isoformat()
            except Exception:
                converted_results[key] = str(value)
        elif isinstance(value, pd.Timedelta):
            converted_results[key] = str(value)
            
    return converted_results

@backtest_bp.route('/backtest', methods=['POST'])
def backtest():
    """Run a backtest with the provided parameters."""
    params = request.json
    current_app.logger.info(f"Received backtest params: {params}")

    # Validate the strategy
    strategy_name = params.get("strategy_obj")
    if not strategy_name:
        return jsonify({"status": "error", "message": "Strategy not provided"}), 400
        
    strategy_obj = get_strategy_class(strategy_name)
    if not strategy_obj:
        return jsonify({"status": "error", "message": f"Invalid strategy: {strategy_name}"}), 400

    # Extract authentication token
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"status": "error", "message": "Authentication required"}), 401

    if token.startswith('Bearer '):
        token = token[7:]
        
    try:
        # Decode token to get user email
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
        email = decoded['email']
        
        # Run the backtest
        stats, graph_base64 = run_backtest(
            symbol=params.get("symbol"),
            granularity=params.get("granularity"),
            strategy_obj=strategy_obj,
            num_days=params.get("num_days", current_app.config.get('BACKTEST_DEFAULT_DAYS', 30)),
            sizing=params.get("sizing", current_app.config.get('TRADING_DEFAULT_SIZING', 1000)),
            best_params=params.get("best_params", {}),
            graph_callback=to_png
        )
        
        if stats is None:
            return jsonify({"status": "error", "message": "Backtest returned no stats"}), 500
            
        # Convert results to JSON-serializable format
        json_friendly_stats = convert_results_to_json(stats)
        
        # Save backtest to database
        database_interaction.save_backtest(
            email=email,
            symbol=params["symbol"],
            strategy=params["strategy_obj"],
            result=stats,
            date=datetime.datetime.now().isoformat()
        )
        
        return jsonify({
            "status": "success", 
            "stats": json_friendly_stats, 
            "graph": graph_base64
        })
        
    except jwt.ExpiredSignatureError:
        return jsonify({"status": "error", "message": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"status": "error", "message": "Invalid token"}), 401
    except Exception as e:
        current_app.logger.error(f"Backtest failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@backtest_bp.route('/backtests', methods=['GET'])
def get_backtest_history():
    """Get backtest history for the authenticated user."""
    token = request.headers.get('Authorization')
    current_app.logger.info(f"Get backtest history - Authorization header: {token}")
    
    if not token or not token.startswith("Bearer "):
        current_app.logger.error("Token is missing or invalid")
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    
    token = token.split("Bearer ")[1]
    
    try:
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
        current_app.logger.info(f"Decoded token: {decoded}")
        email = decoded['email']
        
        history = database_interaction.get_backtest_history(email)
        return jsonify({"status": "success", "history": history}), 200
        
    except jwt.ExpiredSignatureError:
        current_app.logger.error("Token has expired")
        return jsonify({"status": "error", "message": "Token has expired"}), 401
    except jwt.InvalidTokenError as e:
        current_app.logger.error(f"Invalid token: {e}")
        return jsonify({"status": "error", "message": "Invalid token"}), 401
    except Exception as e:
        current_app.logger.error(f"Failed to get backtest history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
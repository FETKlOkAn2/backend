import json
import os
import sys
import logging
from datetime import datetime
import jwt
import pandas as pd
import base64
import plotly.io as pio
from flask import Blueprint, request, jsonify

# Add your project's core directory to sys.path
sys.path.append('/opt/python')

# Import your existing modules
import core.database_interaction as database_interaction
from core.webapp.services.backtest_service import get_strategy_class, run_backtest

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

backtest_bp = Blueprint('backtest', __name__)
def to_png(fig):
    """Convert a Plotly figure to a base64-encoded PNG."""
    try:
        img_bytes = pio.to_image(fig, format="png")
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting graph to PNG: {str(e)}")
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

def lambda_handler(event, context):
    """AWS Lambda handler for backtest endpoint"""
    
    # Extract HTTP method and path from the event
    http_method = event.get('httpMethod', '')
    path = event.get('path', '')
    
    # Process based on HTTP method
    if http_method == 'POST' and path.endswith('/backtest'):
        return handle_backtest(event)
    elif http_method == 'GET' and path.endswith('/backtests'):
        return handle_get_history(event)
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'status': 'error', 'message': 'Invalid endpoint or method'})
        }

def handle_backtest(event):
    """Handle POST /backtest request"""
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        logger.info(f"Received backtest params: {body}")
        
        # Validate the strategy
        strategy_name = body.get("strategy_obj")
        if not strategy_name:
            return {
                'statusCode': 400,
                'body': json.dumps({"status": "error", "message": "Strategy not provided"})
            }
            
        strategy_obj = get_strategy_class(strategy_name)
        if not strategy_obj:
            return {
                'statusCode': 400,
                'body': json.dumps({"status": "error", "message": f"Invalid strategy: {strategy_name}"})
            }
    
        # Extract authentication token
        headers = event.get('headers', {})
        token = headers.get('Authorization')
        
        if not token:
            return {
                'statusCode': 401,
                'body': json.dumps({"status": "error", "message": "Authentication required"})
            }
    
        if token.startswith('Bearer '):
            token = token[7:]
            
        # Decode token to get user email
        secret_key = os.environ.get('SECRET_KEY')
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        email = decoded['email']
        
        # Get config values from environment variables
        backtest_default_days = int(os.environ.get('BACKTEST_DEFAULT_DAYS', '30'))
        trading_default_sizing = int(os.environ.get('TRADING_DEFAULT_SIZING', '1000'))
        
        # Run the backtest
        stats, graph_base64 = run_backtest(
            symbol=body.get("symbol"),
            granularity=body.get("granularity"),
            strategy_obj=strategy_obj,
            num_days=body.get("num_days", backtest_default_days),
            sizing=body.get("sizing", trading_default_sizing),
            best_params=body.get("best_params", {}),
            graph_callback=to_png
        )
        
        if stats is None:
            return {
                'statusCode': 500,
                'body': json.dumps({"status": "error", "message": "Backtest returned no stats"})
            }
            
        # Convert results to JSON-serializable format
        json_friendly_stats = convert_results_to_json(stats)
        
        # Save backtest to database
        database_interaction.save_backtest(
            email=email,
            symbol=body["symbol"],
            strategy=body["strategy_obj"],
            result=stats,
            date=datetime.now().isoformat()
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                "status": "success", 
                "stats": json_friendly_stats, 
                "graph": graph_base64
            })
        }
        
    except jwt.ExpiredSignatureError:
        return {
            'statusCode': 401,
            'body': json.dumps({"status": "error", "message": "Token has expired"})
        }
    except jwt.InvalidTokenError:
        return {
            'statusCode': 401,
            'body': json.dumps({"status": "error", "message": "Invalid token"})
        }
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"status": "error", "message": str(e)})
        }

def handle_get_history(event):
    """Handle GET /backtests request"""
    try:
        # Extract token from headers
        headers = event.get('headers', {})
        token = headers.get('Authorization')
        logger.info(f"Get backtest history - Authorization header: {token}")
        
        if not token or not token.startswith("Bearer "):
            logger.error("Token is missing or invalid")
            return {
                'statusCode': 401,
                'body': json.dumps({"status": "error", "message": "Authentication required"})
            }
        
        token = token.split("Bearer ")[1]
        
        # Decode token
        secret_key = os.environ.get('SECRET_KEY')
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        logger.info(f"Decoded token: {decoded}")
        email = decoded['email']
        
        # Get backtest history
        history = database_interaction.get_backtest_history(email)
        return {
            'statusCode': 200,
            'body': json.dumps({"status": "success", "history": history})
        }
            
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return {
            'statusCode': 401,
            'body': json.dumps({"status": "error", "message": "Token has expired"})
        }
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        return {
            'statusCode': 401,
            'body': json.dumps({"status": "error", "message": "Invalid token"})
        }
    except Exception as e:
        logger.error(f"Failed to get backtest history: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"status": "error", "message": str(e)})
        }

# Register the backtest blueprint with the Flask app
@backtest_bp.route('/backtest', methods=['POST'])
def handle_backtest():
    """Handle backtest request"""
    try:
        # Parse request body
        body = request.json
        logger.info(f"Received backtest params: {body}")
        
        # Validate the strategy
        strategy_name = body.get("strategy_obj")
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
            
        # Decode token to get user email
        secret_key = os.environ.get('SECRET_KEY')
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        email = decoded['email']
        
        # Get config values from environment variables
        backtest_default_days = int(os.environ.get('BACKTEST_DEFAULT_DAYS', '30'))
        trading_default_sizing = int(os.environ.get('TRADING_DEFAULT_SIZING', '1000'))
        
        # Run the backtest
        stats, graph_base64 = run_backtest(
            symbol=body.get("symbol"),
            granularity=body.get("granularity"),
            strategy_obj=strategy_obj,
            num_days=body.get("num_days", backtest_default_days),
            sizing=body.get("sizing", trading_default_sizing),
            best_params=body.get("best_params", {}),
            graph_callback=to_png
        )
        
        if stats is None:
            return jsonify({"status": "error", "message": "Backtest returned no stats"}), 500
            
        # Convert results to JSON-serializable format
        json_friendly_stats = convert_results_to_json(stats)
        
        # Save backtest to database
        database_interaction.save_backtest(
            email=email,
            symbol=body["symbol"],
            strategy=body["strategy_obj"],
            result=stats,
            date=datetime.now().isoformat()
        )
        
        return jsonify({
            "status": "success", 
            "stats": json_friendly_stats, 
            "graph": graph_base64
        }), 200
        
    except jwt.ExpiredSignatureError:
        return jsonify({"status": "error", "message": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"status": "error", "message": "Invalid token"}), 401
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@backtest_bp.route('/backtests', methods=['GET'])
def handle_get_history():
    """Handle GET backtests history request"""
    try:
        # Extract token from headers
        token = request.headers.get('Authorization')
        logger.info(f"Get backtest history - Authorization header: {token}")
        
        if not token or not token.startswith("Bearer "):
            logger.error("Token is missing or invalid")
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        
        token = token.split("Bearer ")[1]
        
        # Decode token
        secret_key = os.environ.get('SECRET_KEY')
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        logger.info(f"Decoded token: {decoded}")
        email = decoded['email']
        
        # Get backtest history
        history = database_interaction.get_backtest_history(email)
        return jsonify({"status": "success", "history": history}), 200
            
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return jsonify({"status": "error", "message": "Token has expired"}), 401
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        return jsonify({"status": "error", "message": "Invalid token"}), 401
    except Exception as e:
        logger.error(f"Failed to get backtest history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
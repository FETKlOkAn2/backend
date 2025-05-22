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
from functools import wraps
import requests
# Add your project's core directory to sys.path
sys.path.append('/opt/python')

# Import your existing modules
import core.database_interaction as database_interaction
from core.webapp.services.backtest_service import get_strategy_class, run_backtest

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

COGNITO_REGION = os.environ.get('COGNITO_REGION', 'eu-north-1')
COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID', 'eu-north-1_oh72FkuWi')
COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID', '6fvcfv0kh50t2ukq800jckf0ic')

# Cache for Cognito public keys
_cognito_keys = None

def get_cognito_public_keys():
    """Fetch and cache Cognito public keys"""
    global _cognito_keys
    if _cognito_keys is None:
        try:
            url = f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json'
            response = requests.get(url)
            response.raise_for_status()
            _cognito_keys = response.json()['keys']
            logger.info("Successfully fetched Cognito public keys")
        except Exception as e:
            logger.error(f"Failed to fetch Cognito public keys: {e}")
            raise
    return _cognito_keys

def verify_cognito_token(token):
    """Verify Cognito ID token"""
    try:
        # Decode header to get key ID
        header = jwt.get_unverified_header(token)
        kid = header['kid']
        
        # Find the corresponding public key
        keys = get_cognito_public_keys()
        key = None
        for k in keys:
            if k['kid'] == kid:
                key = k
                break
        
        if not key:
            raise jwt.InvalidTokenError("Unable to find appropriate key")
        
        # Construct the public key
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
        
        # Verify and decode the token
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=['RS256'],
            audience=COGNITO_CLIENT_ID,
            issuer=f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}'
        )
        
        logger.info(f"Successfully verified Cognito token for user: {decoded.get('email')}")
        return decoded
        
    except jwt.ExpiredSignatureError:
        logger.error("Cognito token has expired")
        raise
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid Cognito token: {e}")
        raise
    except Exception as e:
        logger.error(f"Error verifying Cognito token: {e}")
        raise

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            # Extract token from headers
            token = request.headers.get('Authorization')
            
            if not token or not token.startswith("Bearer "):
                return jsonify({"status": "error", "message": "Authentication required"}), 401
            
            token = token.split("Bearer ")[1]
            
            # Try to verify as Cognito token first
            try:
                decoded = verify_cognito_token(token)
                # Add user info to request context
                request.user = {
                    'email': decoded.get('email'),
                    'username': decoded.get('cognito:username'),
                    'user_id': decoded.get('sub')
                }
                return f(*args, **kwargs)
            except Exception as cognito_error:
                logger.warning(f"Cognito token verification failed: {cognito_error}")
                
                # Fallback to custom JWT verification (for backward compatibility)
                try:
                    secret_key = os.environ.get('SECRET_KEY')
                    if not secret_key:
                        raise Exception("No SECRET_KEY configured")
                    
                    decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
                    request.user = {
                        'email': decoded.get('email'),
                        'username': decoded.get('username'),
                        'user_id': decoded.get('user_id')
                    }
                    return f(*args, **kwargs)
                except Exception as custom_error:
                    logger.error(f"Custom JWT verification also failed: {custom_error}")
                    raise cognito_error  # Return the original Cognito error
                    
        except jwt.ExpiredSignatureError:
            return jsonify({"status": "error", "message": "Token has expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"status": "error", "message": "Invalid token"}), 401
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return jsonify({"status": "error", "message": "Authentication failed"}), 401
    
    return decorated

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
@require_auth
def handle_get_history():
    """Handle GET backtests history request"""
    try:
        email = request.user['email']
        logger.info(f"Getting backtest history for user: {email}")
        
        # Get backtest history
        history = database_interaction.get_backtest_history(email)
        return jsonify({"status": "success", "history": history}), 200
            
    except Exception as e:
        logger.error(f"Failed to get backtest history: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
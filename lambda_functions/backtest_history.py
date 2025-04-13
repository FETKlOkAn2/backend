import json
import os
import sys
import logging
import jwt

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Import your core modules
import core.database_interaction as database_interaction

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """Handle GET request for backtest history"""
    try:
        # Extract token from headers
        headers = event.get('headers', {})
        token = headers.get('Authorization')
        logger.info(f"Get backtest history - Authorization header: {token}")
        
        if not token or not token.startswith("Bearer "):
            logger.error("Token is missing or invalid")
            return {
                'statusCode': 401,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({"status": "error", "message": "Authentication required"})
            }
        
        token = token.split("Bearer ")[1]
        
        # Decode token
        secret_key = os.environ.get('SECRET_KEY')
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        email = decoded['email']
        
        # Get backtest history
        history = database_interaction.get_backtest_history(email)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({"status": "success", "history": history})
        }
            
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return {
            'statusCode': 401,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({"status": "error", "message": "Token has expired"})
        }
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        return {
            'statusCode': 401,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({"status": "error", "message": "Invalid token"})
        }
    except Exception as e:
        logger.error(f"Failed to get backtest history: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({"status": "error", "message": str(e)})
        }
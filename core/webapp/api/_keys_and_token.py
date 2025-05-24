import os
import logging
import requests
import jwt
from flask import request, jsonify
from functools import wraps

COGNITO_REGION = os.environ.get('COGNITO_REGION', 'eu-north-1')
COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID', 'eu-north-1_oh72FkuWi')
COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID', '6fvcfv0kh50t2ukq800jckf0ic')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
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
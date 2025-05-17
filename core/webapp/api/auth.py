from flask import Blueprint, request, jsonify, current_app, redirect, url_for, session
import jwt
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import core.database_interaction as database_interaction

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/status', methods=['GET'])
def status():
    """Check authentication status"""
    user = session.get('user')
    if user:
        return jsonify({
            "status": "authenticated",
            "user": user
        })
    else:
        return jsonify({
            "status": "unauthenticated"
        }), 401

@auth_bp.route('/login', methods=['GET'])
def login_redirect():
    """Redirect to Cognito login"""
    oauth = current_app.extensions['oauth']
    redirect_uri = url_for('authorize', _external=True)
    return oauth.cognito.authorize_redirect(redirect_uri)

# For API clients that need tokens
@auth_bp.route('/token', methods=['GET'])
def get_token():
    """Get current session token information"""
    user = session.get('user')
    if not user:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    # You might want to return the OAuth token or a custom token here
    # This example creates a simple app token based on the Cognito user
    token = jwt.encode(
        {"email": user.get('email'), "exp": datetime.datetime.now() + current_app.config['JWT_EXPIRATION']},
        current_app.config['SECRET_KEY'],
        algorithm="HS256"
    )
    
    return jsonify({
        "status": "success",
        "token": token,
        "user": user
    })

@auth_bp.route('/validate', methods=['POST'])
def validate_token():
    """Validate a token (for API access)"""
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        return jsonify({"status": "error", "message": "Authorization header is missing"}), 401
    
    try:
        # For API clients using our custom token
        email = decode_auth_token(auth_header)
        return jsonify({"status": "success", "email": email}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"status": "error", "message": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"status": "error", "message": "Invalid token"}), 401

def decode_auth_token(token):
    """
    Decode an authentication token and return the user email.
    
    Args:
        token (str): The JWT token to decode
        
    Returns:
        str: The user email if token is valid
        
    Raises:
        jwt.ExpiredSignatureError: If the token has expired
        jwt.InvalidTokenError: If the token is invalid
    """
    # Remove 'Bearer ' prefix if present
    if token.startswith('Bearer '):
        token = token[7:]
        
    # Decode the token
    decoded = jwt.decode(
        token, 
        current_app.config['SECRET_KEY'], 
        algorithms=["HS256"]
    )
    
    return decoded['email']

# Legacy routes with deprecation warnings - can be removed later
@auth_bp.route('/register', methods=['POST'])
def register():
    """Legacy registration endpoint (deprecated)"""
    current_app.logger.warning("Legacy registration endpoint used - should migrate to Cognito")
    
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    # Ensure the email and password are provided
    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400
    
    # Get all users from the database
    users = database_interaction.get_users()
    
    if email in users:
        return jsonify({"status": "error", "message": "User already exists"}), 400
    
    # Encrypt the password before saving it
    encrypted_password = generate_password_hash(password)

    # Save user to database
    try:
        database_interaction.save_user(email, encrypted_password)
        current_app.logger.info(f"User registered successfully: {email}")
        return jsonify({
            "status": "success", 
            "message": "User registered successfully",
            "note": "Consider using Cognito authentication instead of this legacy endpoint"
        }), 201
    except Exception as e:
        current_app.logger.error(f"Failed to register user: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to register user"}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Legacy login endpoint (deprecated)"""
    current_app.logger.warning("Legacy login endpoint used - should migrate to Cognito")
    
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Ensure the email and password are provided
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
    current_app.logger.info(f"User logged in: {email}")

    # Generate JWT token
    token = jwt.encode(
        {"email": email, "exp": datetime.datetime.now() + current_app.config['JWT_EXPIRATION']},
        current_app.config['SECRET_KEY'],
        algorithm="HS256"
    )
    
    return jsonify({
        "status": "success", 
        "token": token,
        "note": "Consider using Cognito authentication instead of this legacy endpoint"
    }), 200
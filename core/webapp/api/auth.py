from flask import Blueprint, request, jsonify, current_app
import jwt
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import core.database_interaction as database_interaction

auth_bp = Blueprint('auth', __name__)

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

@auth_bp.route('/register', methods=['POST'])
def register():
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
        return jsonify({"status": "success", "message": "User registered successfully"}), 201
    except Exception as e:
        current_app.logger.error(f"Failed to register user: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to register user"}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
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
    
    return jsonify({"status": "success", "token": token}), 200
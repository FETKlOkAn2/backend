from flask import Blueprint, request, jsonify, current_app
import boto3
import os
import hmac
import hashlib
import base64
from flask_cors import cross_origin
import dotenv
from core import database_interaction

dotenv.load_dotenv(override=True)

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/proxy/login', methods=['POST'])
@cross_origin(supports_credentials=True)
def proxy_login():
    """
    Proxy endpoint for Cognito authentication that handles SECRET_HASH generation
    """
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Validate inputs
        if not username or not password:
            return jsonify({"status": "error", "message": "Username and password are required"}), 400
        
        # Get Cognito credentials from environment variables
        client_id = os.getenv('COGNITO_APP_CLIENT_ID')
        client_secret = os.getenv('COGNITO_APP_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            current_app.logger.error("Missing Cognito credentials in environment variables")
            return jsonify({"status": "error", "message": "Server configuration error"}), 500
        
        # Calculate SECRET_HASH
        message = username + client_id
        dig = hmac.new(
            client_secret.encode('utf-8'),
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        secret_hash = base64.b64encode(dig).decode()
        
        # Initialize Cognito client
        client = boto3.client(
            'cognito-idp',
            region_name=os.getenv('AWS_REGION', 'eu-north-1')
        )
        
        # Attempt authentication with Cognito
        response = client.initiate_auth(
            ClientId=client_id,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password,
                'SECRET_HASH': secret_hash
            }
        )
        
        # Return authentication result to the client
        return jsonify({
            "status": "success",
            "message": "Login successful",
            "tokens": {
                "idToken": response['AuthenticationResult']['IdToken'],
                "accessToken": response['AuthenticationResult']['AccessToken'],
                "refreshToken": response['AuthenticationResult']['RefreshToken'],
                "expiresIn": response['AuthenticationResult']['ExpiresIn']
            }
        })
        
    except client.exceptions.NotAuthorizedException:
        return jsonify({"status": "error", "message": "Incorrect username or password"}), 401
    
    except client.exceptions.UserNotFoundException:
        return jsonify({"status": "error", "message": "User does not exist"}), 404
    
    except client.exceptions.UserNotConfirmedException:
        return jsonify({"status": "error", "message": "User is not confirmed"}), 403
    
    except Exception as e:
        current_app.logger.error(f"Authentication error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@auth_bp.route('/proxy/signup', methods=['POST'])
@cross_origin(supports_credentials=True)
def proxy_signup():
    """
    Proxy endpoint for Cognito user registration that handles SECRET_HASH generation
    """
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        name = data.get('name')
        birthdate = data.get('birthdate')
        gender = data.get('gender')
        locale = data.get('locale')
        
        # Validate inputs
        if not username or not password or not email or not name or not birthdate or not gender or not locale:
            missing_fields = []
            if not username: missing_fields.append("username")
            if not password: missing_fields.append("password")
            if not email: missing_fields.append("email")
            if not name: missing_fields.append("name")
            if not birthdate: missing_fields.append("birthdate")
            if not gender: missing_fields.append("gender")
            if not locale: missing_fields.append("locale")
            
            print(f"Missing fields: {missing_fields}")
            current_app.logger.error(f"Missing required fields: {', '.join(missing_fields)}")
            return jsonify({
                "status": "error", 
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        dotenv.load_dotenv(override=True)
        # Get Cognito credentials from environment variables
        client_id = os.getenv('COGNITO_APP_CLIENT_ID')
        client_secret = os.getenv('COGNITO_APP_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            current_app.logger.error("Missing Cognito credentials in environment variables")
            return jsonify({"status": "error", "message": "Server configuration error"}), 500
        
        # Create Cognito client
        client = boto3.client(
            'cognito-idp',
            region_name=os.getenv('AWS_REGION', 'eu-north-1')
        )
        
        # Calculate SECRET_HASH (only if client secret is provided)
        secret_hash = None
        if client_secret:
            message = username + client_id
            dig = hmac.new(
                client_secret.encode('utf-8'),
                msg=message.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
            secret_hash = base64.b64encode(dig).decode()
        
        # Prepare user attributes
        user_attributes = [
            {'Name': 'email', 'Value': email},
            {'Name': 'name', 'Value': name},
            {'Name': 'birthdate', 'Value': birthdate},
            {'Name': 'gender', 'Value': gender},
            {'Name': 'locale', 'Value': locale}
        ]
        
        # Prepare sign_up params
        signup_params = {
            'ClientId': client_id,
            'Username': username,
            'Password': password,
            'UserAttributes': user_attributes
        }
        
        # Add SecretHash if available
        if secret_hash:
            signup_params['SecretHash'] = secret_hash
        
        # Attempt user registration with Cognito
        response = client.sign_up(**signup_params)
        current_app.logger.info(f"User {email} registered successfully in Cognito.")

        # Save user to database - separate try/catch for database errors
        try:
            database_interaction.save_user(email)
            print(f"User {email} registered successfully in the database.")
            current_app.logger.info(f"User {email} saved to database successfully.")
        except Exception as db_error:
            current_app.logger.error(f"Database error for user {email}: {str(db_error)}")
            # You might want to decide whether to return success or error here
            # Since Cognito registration succeeded but DB failed
            return jsonify({
                "status": "warning",
                "message": "User registered in Cognito but database save failed. Please contact support.",
                "userSub": response.get('UserSub')
            }), 201
        
        # Return registration result to the client
        return jsonify({
            "status": "success",
            "message": "User registration successful. Verification code sent to email.",
            "userSub": response.get('UserSub')
        })
        
    except client.exceptions.UsernameExistsException:
        return jsonify({"status": "error", "message": "User already exists"}), 400
    
    except client.exceptions.InvalidPasswordException as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    
    except client.exceptions.InvalidParameterException as e:
        return jsonify({"status": "error", "message": f"Invalid parameter: {str(e)}"}), 400
    
    except Exception as e:
        current_app.logger.error(f"Registration error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@auth_bp.route('/proxy/confirm-signup', methods=['POST'])
@cross_origin(supports_credentials=True)
def proxy_confirm_signup():
    """
    Proxy endpoint for confirming user registration with verification code
    """
    try:
        data = request.get_json()
        username = data.get('username')
        confirmation_code = data.get('code')
        
        # Validate inputs
        if not username or not confirmation_code:
            return jsonify({"status": "error", "message": "Username and confirmation code are required"}), 400
        
        # Get Cognito credentials from environment variables
        client_id = os.getenv('COGNITO_APP_CLIENT_ID')
        client_secret = os.getenv('COGNITO_APP_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            current_app.logger.error("Missing Cognito credentials in environment variables")
            return jsonify({"status": "error", "message": "Server configuration error: Missing Cognito credentials"}), 500
        
        # Calculate SECRET_HASH
        message = username + client_id
        dig = hmac.new(
            client_secret.encode('utf-8'),
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        secret_hash = base64.b64encode(dig).decode()
        
        # Initialize Cognito client
        client = boto3.client(
            'cognito-idp',
            region_name=os.getenv('AWS_REGION', 'eu-north-1')
        )
        
        # Confirm signup
        client.confirm_sign_up(
            ClientId=client_id,
            SecretHash=secret_hash,
            Username=username,
            ConfirmationCode=confirmation_code
        )
        
        return jsonify({
            "status": "success",
            "message": "User confirmed successfully"
        })
        
    except client.exceptions.CodeMismatchException:
        current_app.logger.error(f"Invalid verification code for user: {username}")
        return jsonify({"status": "error", "message": "Invalid verification code"}), 400
    
    except client.exceptions.ExpiredCodeException:
        current_app.logger.error(f"Expired verification code for user: {username}")
        return jsonify({"status": "error", "message": "Verification code has expired"}), 400
    
    except Exception as e:
        current_app.logger.error(f"Confirmation error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@auth_bp.route('/proxy/forgot-password', methods=['POST'])
@cross_origin(supports_credentials=True)
def proxy_forgot_password():
    """
    Proxy endpoint for initiating password reset
    """
    try:
        data = request.get_json()
        username = data.get('username')
        
        # Validate inputs
        if not username:
            return jsonify({"status": "error", "message": "Username is required"}), 400
        
        dotenv.load_dotenv(override=True)
        # Get Cognito credentials from environment variables
        client_id = os.getenv('COGNITO_APP_CLIENT_ID')
        client_secret = os.getenv('COGNITO_APP_CLIENT_SECRET')
        
        # Calculate SECRET_HASH
        message = username + client_id
        dig = hmac.new(
            client_secret.encode('utf-8'),
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        secret_hash = base64.b64encode(dig).decode()
        
        # Initialize Cognito client
        client = boto3.client(
            'cognito-idp',
            region_name=os.getenv('AWS_REGION', 'eu-north-1')
        )
        
        # Initiate forgot password flow
        client.forgot_password(
            ClientId=client_id,
            SecretHash=secret_hash,
            Username=username
        )
        
        return jsonify({
            "status": "success",
            "message": "Password reset code sent"
        })
        
    except client.exceptions.UserNotFoundException:
        return jsonify({"status": "error", "message": "User does not exist"}), 404
    
    except client.exceptions.InvalidParameterException as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    
    except Exception as e:
        current_app.logger.error(f"Forgot password error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@auth_bp.route('/proxy/confirm-forgot-password', methods=['POST'])
@cross_origin(supports_credentials=True)
def proxy_confirm_forgot_password():
    """
    Proxy endpoint for confirming password reset
    """
    try:
        data = request.get_json()
        username = data.get('username')
        confirmation_code = data.get('code')
        new_password = data.get('password')
        
        # Validate inputs
        if not username or not confirmation_code or not new_password:
            return jsonify({"status": "error", "message": "Username, confirmation code, and new password are required"}), 400
        
        dotenv.load_dotenv(override=True)
        # Get Cognito credentials from environment variables
        client_id = os.getenv('COGNITO_APP_CLIENT_ID')
        client_secret = os.getenv('COGNITO_APP_CLIENT_SECRET')
        
        # Calculate SECRET_HASH
        message = username + client_id
        dig = hmac.new(
            client_secret.encode('utf-8'),
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        secret_hash = base64.b64encode(dig).decode()
        
        # Initialize Cognito client
        client = boto3.client(
            'cognito-idp',
            region_name=os.getenv('AWS_REGION', 'eu-north-1')
        )
        
        # Confirm password reset
        client.confirm_forgot_password(
            ClientId=client_id,
            SecretHash=secret_hash,
            Username=username,
            ConfirmationCode=confirmation_code,
            Password=new_password
        )
        
        return jsonify({
            "status": "success",
            "message": "Password reset successfully"
        })
        
    except client.exceptions.CodeMismatchException:
        return jsonify({"status": "error", "message": "Invalid verification code"}), 400
    
    except client.exceptions.ExpiredCodeException:
        return jsonify({"status": "error", "message": "Verification code has expired"}), 400
    
    except client.exceptions.InvalidPasswordException as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    
    except Exception as e:
        current_app.logger.error(f"Confirm forgot password error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
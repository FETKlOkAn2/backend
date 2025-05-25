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
        missing_fields = [f for f in ['username', 'password', 'email', 'name', 'birthdate', 'gender', 'locale'] if not data.get(f)]
        if missing_fields:
            current_app.logger.error(f"Missing required fields: {', '.join(missing_fields)}")
            return jsonify({
                "status": "error", 
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        dotenv.load_dotenv(override=True)
        client_id = os.getenv('COGNITO_APP_CLIENT_ID')
        client_secret = os.getenv('COGNITO_APP_CLIENT_SECRET')
        region = os.getenv('AWS_REGION', 'eu-north-1')
        
        if not client_id or not client_secret:
            current_app.logger.error("Missing Cognito credentials in environment variables")
            return jsonify({"status": "error", "message": "Server configuration error"}), 500

        client = boto3.client('cognito-idp', region_name=region)

        if check_user_exists(email, client):
            return jsonify({"status": "error", "message": "User with this email already exists"}), 400  
        # Generate secret hash
        secret_hash = base64.b64encode(hmac.new(
            client_secret.encode('utf-8'),
            msg=(username + client_id).encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()).decode()

        user_attributes = [
            {'Name': 'email', 'Value': email},
            {'Name': 'name', 'Value': name},
            {'Name': 'birthdate', 'Value': birthdate},
            {'Name': 'gender', 'Value': gender},
            {'Name': 'locale', 'Value': locale}
        ]

        signup_params = {
            'ClientId': client_id,
            'Username': username,
            'Password': password,
            'UserAttributes': user_attributes,
            'SecretHash': secret_hash
        }

        # FIRST: Attempt Cognito sign-up
        try:
            response = client.sign_up(**signup_params)
            current_app.logger.info(f"User {email} registered successfully in Cognito.")
        except client.exceptions.UsernameExistsException:
            return jsonify({"status": "error", "message": "User already exists"}), 400
        except client.exceptions.InvalidPasswordException as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        except client.exceptions.InvalidParameterException as e:
            return jsonify({"status": "error", "message": f"Invalid parameter: {str(e)}"}), 400
        except Exception as e:
            current_app.logger.error(f"Error registering user in Cognito: {str(e)}")
            return jsonify({"status": "error", "message": f"Cognito sign-up failed: {str(e)}"}), 500

        # THEN: Save to database
        try:
            database_interaction.save_user(email)
            current_app.logger.info(f"User {email} saved to database successfully.")
        except Exception as e:
            current_app.logger.error(f"Error saving user {email} to database: {str(e)}")
            # Consider rolling back Cognito user (e.g., admin_delete_user if email verification not yet done)
            return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500

        return jsonify({
            "status": "success",
            "message": "User registration successful. Verification code sent to email.",
            "userSub": response.get('UserSub')
        })

    except Exception as e:
        current_app.logger.error(f"Unexpected registration error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
def check_user_exists(email, client):
    """Check if user with email already exists"""
    try:
        response = client.admin_get_user(
            UserPoolId=os.getenv('COGNITO_USER_POOL_ID'),
            Username=email  # or however you want to search
        )
        return True
    except client.exceptions.UserNotFoundException:
        return False
    except Exception:
        # If we can't check, assume it might exist
        return True

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
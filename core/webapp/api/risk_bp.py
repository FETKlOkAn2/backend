import sys
import logging
from flask import Blueprint, request, jsonify
from core.webapp.api._keys_and_token import require_auth

sys.path.append('/opt/python')

from core import database_interaction
logger = logging.getLogger(__name__)

# Create the risk blueprint
risk_bp = Blueprint('risk', __name__)

@risk_bp.route('/risk', methods=['GET'])
@require_auth
def get_risk_settings():
    """Get user's risk settings"""
    try:
        # Enhanced logging for debugging
        logger.info("=== GET /risk endpoint called ===")
        
        # Check if user info exists in request
        if not hasattr(request, 'user') or not request.user:
            logger.error("No user information found in request")
            return jsonify({"status": "error", "message": "Authentication failed - no user data"}), 401
        
        if 'email' not in request.user:
            logger.error("No email found in user data")
            return jsonify({"status": "error", "message": "Authentication failed - no email"}), 401
            
        email = request.user['email']
        logger.info(f"Getting risk settings for user: {email}")
        
        # Check if database_interaction module has the required function
        if not hasattr(database_interaction, 'get_risk_settings'):
            logger.error("database_interaction.get_risk_settings function not found")
            return jsonify({"status": "error", "message": "Database function not available"}), 500
        
        # Get risk settings from database with enhanced error handling
        try:
            risk_settings = database_interaction.get_risk_settings(email)
            logger.info(f"Database returned risk_settings: {risk_settings}")
        except Exception as db_error:
            logger.error(f"Database error in get_risk_settings: {str(db_error)}")
            logger.exception("Full database error traceback:")
            return jsonify({"status": "error", "message": f"Database error: {str(db_error)}"}), 500
        
        if risk_settings is None:
            logger.info("No risk settings found, returning defaults")
            # Return default settings if user not found or no settings exist
            default_settings = {
                "risk_percent": 0.02,
                "max_drawdown": 0.15,
                "max_open_trades": 3
            }
            return jsonify({
                "status": "success", 
                "risk_settings": default_settings,
                "message": "Using default risk settings"
            }), 200
        
        logger.info(f"Returning risk settings: {risk_settings}")
        return jsonify({
            "status": "success", 
            "risk_settings": risk_settings
        }), 200
            
    except Exception as e:
        logger.error(f"Unexpected error in get_risk_settings: {str(e)}")
        logger.exception("Full error traceback:")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@risk_bp.route('/risk', methods=['POST'])
@require_auth
def update_risk_settings():
    """Update user's risk settings"""
    try:
        logger.info("=== POST /risk endpoint called ===")
        
        # Enhanced authentication checks
        if not hasattr(request, 'user') or not request.user:
            logger.error("No user information found in request")
            return jsonify({"status": "error", "message": "Authentication failed"}), 401
            
        email = request.user['email']
        logger.info(f"Updating risk settings for user: {email}")
        
        # Parse request body with better error handling
        try:
            body = request.get_json()
            if not body:
                logger.error("No JSON body found in request")
                return jsonify({"status": "error", "message": "Request body is required"}), 400
        except Exception as json_error:
            logger.error(f"Error parsing JSON: {str(json_error)}")
            return jsonify({"status": "error", "message": "Invalid JSON format"}), 400
        
        logger.info(f"Request body received: {body}")
        
        # Validate required fields
        required_fields = ['risk_percent', 'max_drawdown', 'max_open_trades']
        missing_fields = [field for field in required_fields if field not in body]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        # Validate data types and ranges
        try:
            risk_percent = float(body['risk_percent'])
            max_drawdown = float(body['max_drawdown'])
            max_open_trades = int(body['max_open_trades'])
            
            logger.info(f"Parsed values - risk_percent: {risk_percent}, max_drawdown: {max_drawdown}, max_open_trades: {max_open_trades}")
            
            # Validate ranges
            if not (0.001 <= risk_percent <= 1.0):  # 0.1% to 100%
                logger.error(f"Risk percent out of range: {risk_percent}")
                return jsonify({"status": "error", "message": "Risk percent must be between 0.1% and 100%"}), 400
            
            if not (0.01 <= max_drawdown <= 1.0):  # 1% to 100%
                logger.error(f"Max drawdown out of range: {max_drawdown}")
                return jsonify({"status": "error", "message": "Max drawdown must be between 1% and 100%"}), 400
            
            if not (1 <= max_open_trades <= 50):  # 1 to 50 trades
                logger.error(f"Max open trades out of range: {max_open_trades}")
                return jsonify({"status": "error", "message": "Max open trades must be between 1 and 50"}), 400
                
        except (ValueError, TypeError) as e:
            logger.error(f"Data type validation error: {str(e)}")
            return jsonify({"status": "error", "message": f"Invalid data type: {str(e)}"}), 400
        
        # Check if update function exists
        if not hasattr(database_interaction, 'update_risk_settings'):
            logger.error("database_interaction.update_risk_settings function not found")
            return jsonify({"status": "error", "message": "Database update function not available"}), 500
        
        # Update risk settings in database
        try:
            success = database_interaction.update_risk_settings(
                email=email,
                risk_percent=risk_percent,
                max_drawdown=max_drawdown,
                max_open_trades=max_open_trades
            )
            logger.info(f"Database update result: {success}")
        except Exception as db_error:
            logger.error(f"Database error in update_risk_settings: {str(db_error)}")
            logger.exception("Full database update error traceback:")
            return jsonify({"status": "error", "message": f"Database update error: {str(db_error)}"}), 500
        
        if success:
            logger.info("Risk settings updated successfully")
            return jsonify({
                "status": "success", 
                "message": "Risk settings updated successfully"
            }), 200
        else:
            logger.error("Database update returned False")
            return jsonify({
                "status": "error", 
                "message": "Failed to update risk settings"
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in update_risk_settings: {str(e)}")
        logger.exception("Full error traceback:")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@risk_bp.route('/risk/presets', methods=['GET'])
@require_auth
def get_risk_presets():
    """Get available risk profile presets"""
    try:
        logger.info("=== GET /risk/presets endpoint called ===")
        
        presets = {
            "conservative": {
                "risk_percent": 0.01,
                "max_drawdown": 0.10,
                "max_open_trades": 2,
                "description": "Low risk, suitable for beginners"
            },
            "moderate": {
                "risk_percent": 0.02,
                "max_drawdown": 0.15,
                "max_open_trades": 3,
                "description": "Balanced risk and reward"
            },
            "aggressive": {
                "risk_percent": 0.05,
                "max_drawdown": 0.25,
                "max_open_trades": 5,
                "description": "Higher risk, higher potential reward"
            }
        }
        
        logger.info("Returning risk presets successfully")
        return jsonify({
            "status": "success", 
            "presets": presets
        }), 200
            
    except Exception as e:
        logger.error(f"Error in get_risk_presets: {str(e)}")
        logger.exception("Full error traceback:")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@risk_bp.route('/risk/validate', methods=['POST'])
@require_auth
def validate_risk_settings():
    """Validate risk settings without saving"""
    try:
        logger.info("=== POST /risk/validate endpoint called ===")
        
        # Parse request body
        try:
            body = request.get_json()
            if not body:
                return jsonify({"status": "error", "message": "Request body is required"}), 400
        except Exception as json_error:
            logger.error(f"Error parsing JSON in validate: {str(json_error)}")
            return jsonify({"status": "error", "message": "Invalid JSON format"}), 400
        
        # Validate required fields
        required_fields = ['risk_percent', 'max_drawdown', 'max_open_trades']
        missing_fields = [field for field in required_fields if field not in body]
        if missing_fields:
            return jsonify({"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        # Validate data types and ranges
        try:
            risk_percent = float(body['risk_percent'])
            max_drawdown = float(body['max_drawdown'])
            max_open_trades = int(body['max_open_trades'])
            
            warnings = []
            errors = []
            
            # Validate ranges
            if not (0.001 <= risk_percent <= 1.0):
                errors.append("Risk percent must be between 0.1% and 100%")
            elif risk_percent > 0.10:  # Warning for very high risk
                warnings.append("Risk percent above 10% is considered very aggressive")
            
            if not (0.01 <= max_drawdown <= 1.0):
                errors.append("Max drawdown must be between 1% and 100%")
            elif max_drawdown > 0.50:  # Warning for high drawdown
                warnings.append("Max drawdown above 50% is considered very high")
            
            if not (1 <= max_open_trades <= 50):
                errors.append("Max open trades must be between 1 and 50")
            elif max_open_trades > 10:  # Warning for many concurrent trades
                warnings.append("More than 10 concurrent trades may be difficult to manage")
            
            # Calculate risk score
            risk_score = (risk_percent / 0.02) + (max_drawdown / 0.15) + (max_open_trades / 3)
            
            if risk_score <= 2:
                risk_class = "Conservative"
            elif risk_score <= 3:
                risk_class = "Moderate"
            elif risk_score <= 4:
                risk_class = "Balanced"
            elif risk_score <= 5:
                risk_class = "Growth"
            else:
                risk_class = "Aggressive"
            
            logger.info(f"Validation complete - valid: {len(errors) == 0}, risk_class: {risk_class}")
            
            return jsonify({
                "status": "success",
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "risk_class": risk_class,
                "risk_score": round(risk_score, 2)
            }), 200
                
        except (ValueError, TypeError) as e:
            logger.error(f"Data type validation error in validate: {str(e)}")
            return jsonify({
                "status": "error", 
                "valid": False,
                "errors": [f"Invalid data type: {str(e)}"]
            }), 400
            
    except Exception as e:
        logger.error(f"Error in validate_risk_settings: {str(e)}")
        logger.exception("Full error traceback:")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

# Health check endpoint for debugging
@risk_bp.route('/risk/health', methods=['GET'])
def health_check():
    """Health check endpoint to test basic functionality"""
    try:
        logger.info("=== Risk API Health Check ===")
        
        # Check if database module is importable
        db_status = "OK"
        try:
            from core import database_interaction
            if hasattr(database_interaction, 'get_risk_settings'):
                db_status = "get_risk_settings function available"
            else:
                db_status = "get_risk_settings function NOT FOUND"
        except Exception as e:
            db_status = f"Database import error: {str(e)}"
        
        return jsonify({
            "status": "success",
            "message": "Risk API is running",
            "database_status": db_status,
            "timestamp": str(__import__('datetime').datetime.now())
        }), 200
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Health check failed: {str(e)}"
        }), 500
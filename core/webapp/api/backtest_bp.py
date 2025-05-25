import json
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import base64
import plotly.io as pio
from flask import Blueprint, request, jsonify

# Add your project's core directory to sys.path
sys.path.append('/opt/python')

# Import your existing modules
from core import database_interaction
from core.webapp.services.backtest_service import get_strategy_class, run_backtest
from core.webapp.api._keys_and_token import require_auth

logger = logging.getLogger(__name__)

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

@backtest_bp.route('/backtest', methods=['POST'])
@require_auth
def handle_backtest():
    """Handle backtest request"""
    try:
        logger.info("=== POST /backtest endpoint called ===")
        
        # Enhanced authentication checks (similar to risk endpoints)
        if not hasattr(request, 'user') or not request.user:
            logger.error("No user information found in request")
            return jsonify({"status": "error", "message": "Authentication failed"}), 401
            
        if 'email' not in request.user:
            logger.error("No email found in user data")
            return jsonify({"status": "error", "message": "Authentication failed - no email"}), 401
            
        email = request.user['email']
        logger.info(f"Running backtest for user: {email}")
        
        # Parse request body with better error handling
        try:
            body = request.get_json()
            if not body:
                logger.error("No JSON body found in request")
                return jsonify({"status": "error", "message": "Request body is required"}), 400
        except Exception as json_error:
            logger.error(f"Error parsing JSON: {str(json_error)}")
            return jsonify({"status": "error", "message": "Invalid JSON format"}), 400
        
        logger.info(f"Received backtest params: {body}")
        
        # Validate the strategy
        strategy_name = body.get("strategy_obj")
        if not strategy_name:
            logger.error("Strategy not provided in request")
            return jsonify({"status": "error", "message": "Strategy not provided"}), 400
            
        strategy_obj = get_strategy_class(strategy_name)
        if not strategy_obj:
            logger.error(f"Invalid strategy: {strategy_name}")
            return jsonify({"status": "error", "message": f"Invalid strategy: {strategy_name}"}), 400
        
        # Validate required fields
        required_fields = ['symbol', 'granularity']
        missing_fields = [field for field in required_fields if not body.get(field)]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        # Get config values from environment variables
        backtest_default_days = int(os.environ.get('BACKTEST_DEFAULT_DAYS', '30'))
        trading_default_sizing = int(os.environ.get('TRADING_DEFAULT_SIZING', '1000'))
        
        # Run the backtest with enhanced error handling
        try:
            stats, graph_base64 = run_backtest(
                symbol=body.get("symbol"),
                granularity=body.get("granularity"),
                strategy_obj=strategy_obj,
                num_days=body.get("num_days", backtest_default_days),
                sizing=body.get("sizing", trading_default_sizing),
                best_params=body.get("best_params", {}),
                graph_callback=to_png
            )
        except Exception as backtest_error:
            logger.error(f"Backtest execution failed: {str(backtest_error)}")
            logger.exception("Full backtest error traceback:")
            return jsonify({"status": "error", "message": f"Backtest execution failed: {str(backtest_error)}"}), 500
        
        if stats is None:
            logger.error("Backtest returned no stats")
            return jsonify({"status": "error", "message": "Backtest returned no stats"}), 500
            
        # Convert results to JSON-serializable format
        try:
            json_friendly_stats = convert_results_to_json(stats)
        except Exception as conversion_error:
            logger.error(f"Error converting stats to JSON: {str(conversion_error)}")
            return jsonify({"status": "error", "message": "Error processing backtest results"}), 500
        
        # Save backtest to database with error handling
        try:
            database_interaction.save_backtest(
                email=email,
                symbol=body["symbol"],
                strategy=body["strategy_obj"],
                result=stats,
                date=datetime.now().isoformat()
            )
            logger.info("Backtest results saved to database successfully")
        except Exception as db_error:
            logger.error(f"Failed to save backtest to database: {str(db_error)}")
            # Don't fail the request if database save fails, just log it
            logger.exception("Database save error traceback:")
        
        logger.info("Backtest completed successfully")
        return jsonify({
            "status": "success", 
            "stats": json_friendly_stats, 
            "graph": graph_base64
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in backtest: {str(e)}")
        logger.exception("Full error traceback:")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@backtest_bp.route('/backtests', methods=['GET'])
@require_auth
def handle_get_history():
    """Handle GET backtests history request"""
    try:
        logger.info("=== GET /backtests endpoint called ===")
        
        # Enhanced authentication checks
        if not hasattr(request, 'user') or not request.user:
            logger.error("No user information found in request")
            return jsonify({"status": "error", "message": "Authentication failed"}), 401
            
        if 'email' not in request.user:
            logger.error("No email found in user data")
            return jsonify({"status": "error", "message": "Authentication failed - no email"}), 401
            
        email = request.user['email']
        logger.info(f"Getting backtest history for user: {email}")
        
        # Check if database function exists
        if not hasattr(database_interaction, 'get_backtest_history'):
            logger.error("database_interaction.get_backtest_history function not found")
            return jsonify({"status": "error", "message": "Database function not available"}), 500
        
        # Get backtest history with error handling
        try:
            history = database_interaction.get_backtest_history(email)
            logger.info(f"Retrieved {len(history) if history else 0} backtest records")
        except Exception as db_error:
            logger.error(f"Database error in get_backtest_history: {str(db_error)}")
            logger.exception("Full database error traceback:")
            return jsonify({"status": "error", "message": f"Database error: {str(db_error)}"}), 500
        
        if history is None:
            logger.info("No backtest history found, returning empty list")
            history = []
        
        return jsonify({
            "status": "success", 
            "history": history
        }), 200
            
    except Exception as e:
        logger.error(f"Unexpected error in get_backtest_history: {str(e)}")
        logger.exception("Full error traceback:")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

# Health check endpoint for debugging
@backtest_bp.route('/backtest/health', methods=['GET'])
def health_check():
    """Health check endpoint to test basic functionality"""
    try:
        logger.info("=== Backtest API Health Check ===")
        
        # Check if database module is importable
        db_status = "OK"
        try:
            from core import database_interaction
            if hasattr(database_interaction, 'save_backtest'):
                db_status = "save_backtest function available"
            else:
                db_status = "save_backtest function NOT FOUND"
        except Exception as e:
            db_status = f"Database import error: {str(e)}"
        
        # Check if backtest service is importable
        service_status = "OK"
        try:
            from core.webapp.services.backtest_service import get_strategy_class, run_backtest
            service_status = "Backtest service functions available"
        except Exception as e:
            service_status = f"Backtest service import error: {str(e)}"
        
        return jsonify({
            "status": "success",
            "message": "Backtest API is running",
            "database_status": db_status,
            "service_status": service_status,
            "timestamp": str(datetime.now())
        }), 200
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Health check failed: {str(e)}"
        }), 500
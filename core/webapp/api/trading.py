from flask import Blueprint, jsonify, current_app
import threading
from core.webapp.services.trading_service import get_live_trader_instance, start_live_trading, stop_live_trading

trading_bp = Blueprint('trading', __name__)
trader_lock = threading.Lock()

@trading_bp.route('/start_trading', methods=['POST'])
def start_trading():
    """Start live trading."""
    with trader_lock:
        try:
            # Check if trading is already running
            if get_live_trader_instance() and get_live_trader_instance().is_running:
                return jsonify({
                    "status": "error", 
                    "message": "Live trading is already running"
                }), 400
            
            # Start trading
            start_live_trading(current_app)
            
            return jsonify({
                "status": "success", 
                "message": "Live trading started"
            }), 200
            
        except Exception as e:
            current_app.logger.error(f"Error starting trading: {e}")
            return jsonify({
                "status": "error", 
                "message": f"Error starting trading: {str(e)}"
            }), 500

@trading_bp.route('/stop_trading', methods=['POST'])
def stop_trading():
    """Stop live trading."""
    try:
        trader = get_live_trader_instance()
        if trader and getattr(trader, 'is_running', False):
            stop_live_trading()
            return jsonify({
                "status": "success", 
                "message": "Live trading stopped"
            }), 200
        
        return jsonify({
            "status": "error", 
            "message": "Live trading not running"
        }), 400
        
    except Exception as e:
        current_app.logger.error(f"Error stopping trading: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Error stopping trading: {str(e)}"
        }), 500

@trading_bp.route('/live_status', methods=['GET'])
def get_live_status():
    """Get current live trading status."""
    try:
        trader = get_live_trader_instance()
        status = "active" if trader and getattr(trader, 'is_running', False) else "inactive"
        
        return jsonify({
            "status": status, 
            "message": f"Live trading is {status}"
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in live_status: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Error getting status: {str(e)}"
        }), 500
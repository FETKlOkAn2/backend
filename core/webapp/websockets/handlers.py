# mypy: disable-error-code=import-untyped
# pylint: disable=C0114

import datetime
import jwt
from flask import request, session
from flask_socketio import emit
from core.webapp.services.trading_service import get_live_trader_instance

def register_socket_handlers(socketio):
    """Register all WebSocket event handlers."""
    
    @socketio.on("connect")
    def handle_connect(auth=None):
        """Handle client WebSocket connection"""
        print(f"Connection attempt from {request.sid}")
        
        token = request.args.get('token')
        if not token:
            print(f"Connection rejected: No token from {request.sid}")
            return False
            
        try:
            # Verify token
            decoded = jwt.decode(
                token, 
                socketio.server.flask_app.config['SECRET_KEY'], 
                algorithms=["HS256"]
            )
            
            # Store user info in session
            session['user'] = decoded['email']
            
            print(f"Client authenticated: {request.sid} (user: {decoded['email']})")
            
            # Send welcome message
            emit("log_update", {
                "message": f"Connected to server as {decoded['email']}",
                "type": "info",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Send current trading status
            try:
                trader = get_live_trader_instance()
                status = "active" if trader and getattr(trader, 'is_running', False) else "inactive"
                emit("trading_status", {"status": status})
            except Exception as e:
                print(f"Error sending initial trading status: {e}")
            
            return True
        except jwt.ExpiredSignatureError:
            print(f"Connection rejected: Token expired from {request.sid}")
            return False
        except jwt.InvalidTokenError as e:
            print(f"Connection rejected: Invalid token from {request.sid}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during connection: {e}")
            return False

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client WebSocket disconnection"""
        print(f"Client disconnected from WebSocket: {request.sid}")

    @socketio.on('get_trading_status')
    def handle_get_trading_status():
        """Return current trading status"""
        trader = get_live_trader_instance()
        status = "active" if trader and getattr(trader, 'is_running', False) else "inactive"
        emit("trading_status", {"status": status})

    @socketio.on('heartbeat')
    def handle_heartbeat():
        """Handle client heartbeat request"""
        print(f"Heartbeat received from {request.sid}")
        emit('heartbeat_ack', {
            "timestamp": datetime.datetime.now().isoformat()
        })
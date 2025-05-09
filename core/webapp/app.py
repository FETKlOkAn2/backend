from gevent import monkey
monkey.patch_all()

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO
import logging

from core.webapp.config import Config
from core.webapp.api.auth import auth_bp
from core.webapp.api.backtest import backtest_bp
from core.webapp.api.trading import trading_bp
from core.webapp.websockets.handlers import register_socket_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable CORS with more detailed configuration
    CORS(
        app,
        resources={
            r"/*": {
                "origins": "https://main.d1ygo5bg0er0wr.amplifyapp.com",
                "supports_credentials": True,
                "allow_headers": [
                    "Content-Type",
                    "Authorization",
                    "X-Requested-With"
                ],
                "methods": [
                    "GET", "HEAD", "POST", "OPTIONS", 
                    "PUT", "PATCH", "DELETE"
                ]
            }
        }
    )
    
    # Handle OPTIONS requests explicitly
    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def options_handler(path):
        response = app.make_default_options_response()
        return response
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(backtest_bp, url_prefix='/api')
    app.register_blueprint(trading_bp, url_prefix='/api')
    
    # Configure error handlers
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Unhandled Exception: {str(e)}")
        from flask import jsonify
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    
    return app

# Create Flask app
app = create_app()

# Configure Socket.IO with proper CORS settings
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",  # List of allowed origins
    async_mode='gevent', 
    logger=True, 
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=30,
    transports=['websocket']
)

app.extensions['socketio'] = socketio
# Register socket event handlers
register_socket_handlers(socketio)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
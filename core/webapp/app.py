# mypy: disable-error-code=import-untyped
# pylint: disable=C0114

from gevent import monkey
monkey.patch_all()

from flask import Flask
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
    
    # Configure CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    
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
    
    @app.before_request
    def handle_options():
        from flask import request, Response
        if request.method == 'OPTIONS':
            response = Response()
            response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.status_code = 200
            return response
    
    return app

# Create Flask app
app = create_app()

# Initialize SocketIO
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='gevent', 
    logger=True, 
    engineio_logger=True,
    ping_timeout=60000,  # 60 seconds
    ping_interval=30000,  # 30 seconds
    transports=['polling', 'websocket']
)

# Register socket event handlers
register_socket_handlers(socketio)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
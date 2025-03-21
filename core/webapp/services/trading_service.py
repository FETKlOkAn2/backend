import asyncio
import threading
import datetime
from core.livetrader import LiveTrader

# Global variable to hold the live trader instance
_live_trader_instance = None
_trading_task = None

def get_live_trader_instance():
    """Get the current live trader instance."""
    global _live_trader_instance
    return _live_trader_instance

def start_live_trading(app):
    """Start live trading."""
    global _live_trader_instance, _trading_task
    
    # Reset the trader if it exists but is not running
    if _live_trader_instance and not getattr(_live_trader_instance, 'is_running', False):
        _live_trader_instance = None
    
    if _live_trader_instance is None:
        socketio = app.extensions['socketio']
        _live_trader_instance = LiveTrader(socketio)
    
    # Start trading in a new thread
    _trading_task = run_livetrader(_live_trader_instance, socketio)
    
    return _trading_task

def stop_live_trading():
    """Stop live trading."""
    global _live_trader_instance
    
    if _live_trader_instance:
        _live_trader_instance.stop()
        # Emit status change via socketio
        from flask import current_app
        socketio = current_app.extensions['socketio']
        socketio.emit('trading_status', {"status": "inactive"})
        return True
    
    return False

def run_livetrader(trader_instance, socketio):
    """Run the live trader in a separate thread."""
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(trader_instance.main())
        except Exception as e:
            print(f"Error in live trader: {e}")
            socketio.emit('log_update', {
                "message": f"Error in live trader: {e}",
                "type": "error",
                "timestamp": datetime.datetime.now().isoformat()
            })
            socketio.emit('trading_status', {"status": "inactive"})
        finally:
            loop.close()
    
    # Start the thread
    thread = threading.Thread(target=run_async_loop, daemon=True)
    thread.start()
    
    return thread
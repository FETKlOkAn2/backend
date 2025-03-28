import datetime
import jwt
from flask import request, session
from flask_socketio import emit
from flask import current_app

from core.webapp.services.trading_service import get_live_trader_instance

def register_socket_handlers(socketio):
    """Register all WebSocket event handlers."""
    
    @socketio.on("connect")
    def handle_connect(auth=None):
        """Handle client WebSocket connection"""
        print(f"Connection attempt from {request.sid}")
        
        token = request.args.get('token')  # Fallback (legacy clients)
        if not token and auth:
            token = auth.get('token')  # Use the new auth field
        
        if not token:
            print(f"Connection rejected: No token from {request.sid}")
            emit("connection_error", {"message": "Authentication required"}, to=request.sid)
            return
            
        try:
            decoded = jwt.decode(
                token, 
                current_app.config['SECRET_KEY'], 
                algorithms=["HS256"]
            )
            session['user'] = decoded['email']
            
            print(f"Client authenticated: {request.sid} (user: {decoded['email']})")
            
            emit("log_update", {
                "message": f"Connected to server as {decoded['email']}",
                "type": "info",
                "timestamp": datetime.datetime.now().isoformat()
            })

            # Send current trading status
            try:
                trader = get_live_trader_instance()
                status = "active" if trader and getattr(trader, 'running', False) else "inactive"
                emit("trading_status", {"status": status})
            except Exception as e:
                print(f"Error sending initial trading status: {e}")
            
        except jwt.ExpiredSignatureError:
            print(f"Connection rejected: Token expired from {request.sid}")
            emit("connection_error", {"message": "Token expired"}, to=request.sid)
        except jwt.InvalidTokenError as e:
            print(f"Connection rejected: Invalid token from {request.sid}: {e}")
            emit("connection_error", {"message": "Invalid token"}, to=request.sid)
        except Exception as e:
            print(f"Unexpected error during connection: {e}")
            emit("connection_error", {"message": "Unexpected error"}, to=request.sid)

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client WebSocket disconnection"""
        print(f"Client disconnected from WebSocket: {request.sid}")

    @socketio.on('get_trading_status')
    def handle_get_trading_status():
        """Return current trading status"""
        trader = get_live_trader_instance()
        status = "active" if trader and getattr(trader, 'running', False) else "inactive"
        emit("trading_status", {"status": status})

    @socketio.on('heartbeat')
    def handle_heartbeat():
        """Handle client heartbeat request"""
        print(f"Heartbeat received from {request.sid}")
        emit('heartbeat_ack', {
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    @socketio.on('request_graph')
    def handle_request_graph(data):
        """Handle requests for symbol graphs"""
        trader = get_live_trader_instance()
        if not trader:
            emit('log_update', {
                "message": f"Error: Trading service not active",
                "type": "error",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return
            
        symbol = data.get('symbol')
        resampling_factor = data.get('resamplingFactor', '15m')  # Default to 15m if not specified
        
        if not symbol:
            emit('log_update', {
                "message": f"Error: No symbol specified for graph request",
                "type": "error",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return
            
        try:
            print(f"Graph requested for {symbol} with resampling {resampling_factor}")
            
            # Update data for the specific symbol
            trader.df_manager.data_for_live_trade(symbol=symbol, update=True, resampling=resampling_factor)
            current_dict = {symbol: trader.df_manager.dict_df[symbol]}
            
            # Use the same strategy as the trading logic
            from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
            strat = RSI_ADX_GPU(current_dict, trader.risk, with_sizing=True, hyper=False)
            
            # Use the parameters specific to this symbol
            params = trader.risk.symbol_params.get(symbol, [14, 14, 20, 80])
            strat.custom_indicator(strat.close, *params)
            
            # Define a callback to capture the graph output
            # In your graph method, ensure the callback marking is done properly
            def graph_callback(fig):
                try:
                    # Convert graph to base64 image
                    import plotly.io as pio
                    import base64
                    img_bytes = pio.to_image(fig, format="png")
                    graph_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Send the graph to the client
                    emit('graph_update', {
                        'symbol': symbol,
                        'graph': graph_base64
                    })
                    
                    print(f"Graph for {symbol} sent successfully")
                    # Mark that we've handled this graph
                    setattr(fig, '_sent_to_client', True)
                    return True
                except Exception as e:
                    print(f"Error generating graph image: {str(e)}")
                    return False
            
            # Generate and send the graph
            fig = strat.graph(graph_callback)
            
            # If graph_callback wasn't called (which can happen), try to send the graph directly
            if fig is not None and not hasattr(fig, '_sent_to_client'):
                try:
                    import plotly.io as pio
                    import base64
                    img_bytes = pio.to_image(fig, format="png")
                    graph_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    emit('graph_update', {
                        'symbol': symbol,
                        'graph': graph_base64
                    })
                    
                    print(f"Graph for {symbol} sent directly")
                except Exception as e:
                    print(f"Error sending graph directly: {str(e)}")
                    emit('log_update', {
                        "message": f"Error generating graph: {str(e)}",
                        "type": "error",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            
        except Exception as e:
            print(f"Error processing graph request for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            emit('log_update', {
                "message": f"Error generating graph for {symbol}: {str(e)}",
                "type": "error",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
    @socketio.on('change_resampling')
    def handle_change_resampling(data):
        """Handle change in resampling factor"""
        factor = data.get('factor')
        if factor:
            try:
                # Store the preferred resampling factor in the session
                session['resampling_factor'] = factor
                emit('log_update', {
                    "message": f"Resampling factor changed to {factor}",
                    "type": "info",
                    "timestamp": datetime.datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error changing resampling factor: {str(e)}")
                emit('log_update', {
                    "message": f"Error changing resampling factor: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.datetime.now().isoformat()
                })
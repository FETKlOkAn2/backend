# mypy: disable-error-code=import-untyped
# pylint: disable=C0114
import time
import asyncio
import datetime as dt
import os
import importlib.util
import inspect
import traceback
from typing import Dict, List, Callable, Any, Optional

from core.dataframe_manager import DF_Manager
from core.log import LinkedList
# from core.strategies.strategy import Strategy
# from core.strategies.single.rsi import RSI
# from core.strategies.double.rsi_adx import RSI_ADX
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
from core.trade import Trade
from core.scanner import Scanner
from core.risk import Risk_Handler
from core.kraken_wrapper import Kraken
import core.database_interaction as database_interaction


class LiveTrader:
    """
    LiveTrader class responsible for managing live trading operations,
    including data fetching, strategy execution, and communication with frontend.
    """
    def __init__(self, socketio):
        """Initialize the LiveTrader with required components and connections."""
        self.socketio = socketio
        self.counter = 0
        self.running = True  # Single status flag for consistency
        self.trades = []
        self.graph_callback = None
        
        # Initialize core components
        self._init_core_components()
        
        # Load strategy classes
        self.strat_classes = {}
        self._load_strategy_classes()
        
        # Send initial connection message
        self.log("LiveTrader initialized and ready", "info")
        
        # Initialize data and parameters
        self._initialize_data_and_params()

    #-----------------
    # Initialization Methods
    #-----------------
    
    def _init_core_components(self):
        """Initialize the core trading components."""
        self.kraken = Kraken()
        self.risk = Risk_Handler(self.kraken)
        self.scanner = Scanner(client=self.kraken, socketio=self.socketio)
        self.df_manager = DF_Manager(self.scanner)
        self.scanner.assign_attribute(df_manager=self.df_manager)
        self.logbook = LinkedList()

    def _initialize_data_and_params(self):
        """Initialize data and strategy parameters."""
        try:
            self.update_candle_data()
            self.load_strategy_params_for_strategy()
            self.log("Initial data loaded successfully", "info")
        except Exception as e:
            self.log(f"Error during initialization: {str(e)}", "error")
            self.log(traceback.format_exc(), "error")

    def _load_strategy_classes(self):
        """Load strategy classes from the strategy directory."""
        strat_path = 'core/strategies'
        try:
            for root, _, files in os.walk(strat_path):
                for file in files:
                    if file.endswith(".py") and file not in ['strategy.py', 'combined_strategy.py']:
                        self._load_strategy_from_file(os.path.join(root, file), file[:-3])
        except Exception as e:
            self.log(f"Error extracting strategy classes: {str(e)}", "error")

    def _load_strategy_from_file(self, file_path, module_name):
        """Load a strategy class from a specific file."""
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module_name:
                    self.strat_classes[name] = obj
                    
        except Exception as e:
            self.log(f"Error loading strategy {module_name}: {str(e)}", "error")

    #-----------------
    # Logging and Status Methods
    #-----------------
    
    def log(self, message, log_type="info"):
        """Log a message to the console and emit to the frontend."""
        print(f"[{log_type.upper()}] {message}")  # Console logging
        try:
            # Emit via socketio with namespace if needed
            self.socketio.emit('log_update', {
                "message": message,
                "type": log_type,
                "timestamp": dt.datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Error emitting log: {e}")

    def get_status(self):
        """Return current trading status."""
        return {
            "running": self.running,
            "symbols": list(self.df_manager.dict_df.keys()) if hasattr(self, 'df_manager') and hasattr(self.df_manager, 'dict_df') else [],
            "counter": self.counter,
        }

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        self.log("Live trading stopped", "info")
        self.socketio.emit('trading_status', {"status": "inactive"})

    #-----------------
    # Data Loading Methods
    #-----------------
    
    def load_strategy_params_for_strategy(self):
        """Load strategy parameters for each symbol."""
        try:
            for symb in self.scanner.kraken_crypto:
                self.log(f"Loading parameters for {symb}", "info")
                try:
                    strat = RSI_ADX_GPU(dict_df=None, risk_object=self.risk)
                    strat.symbol = symb
                    
                    params = database_interaction.get_best_params(
                        strat,
                        self.df_manager,
                        live_trading=True,
                        best_of_all_granularities=True,
                        minimum_trades=4
                    )
                    
                    self.log(f'Best params for {symb}: {params}', "info")
                    self.risk.symbol_params[symb] = params
                    self.df_manager.set_next_update(symb, initial=True)
                except Exception as e:
                    self.log(f"Error loading params for {symb}: {str(e)}", "error")
        except Exception as e:
            self.log(f"Error in load_strategy_params: {str(e)}", "error")

    def update_candle_data(self):
        """Update candle data for all cryptocurrencies."""
        try:
            self.log("Updating candle data, this may take a moment...", "info")
            
            # Create callback function for logging
            def send_log(message):
                self.log(message, "info")
            
            # Get candles with correct parameters
            self.scanner.coinbase.get_candles_for_db(
                self.scanner.coinbase_crypto,
                self.scanner.granularity,
                days=30,
                callback=send_log,
                socketio=self.socketio
            )
            
            self.log("Candle data updated successfully", "info")
        except Exception as e:
            self.log(f"Error updating candle data: {str(e)}", "error")
            self.log(traceback.format_exc(), "error")

    #-----------------
    # Trading Logic Methods
    #-----------------
    
    def process_symbol(self, symbol):
        """Process a single trading symbol with progress updates."""
        try:
            # Skip if not time to update
            if dt.datetime.now() <= self.df_manager.next_update_time.get(symbol, dt.datetime.now()):
                self.log(f"Skipping {symbol} - Next update at {self.df_manager.next_update_time.get(symbol, dt.datetime.now()).strftime('%H:%M:%S')}", "info")
                return False
            
            self.log(f"Processing {symbol}", "info")
            
            # Update data with progress tracking
            self._update_progress(symbol, 10, "Fetching data...")
            
            # Update data
            self.df_manager.data_for_live_trade(symbol=symbol, update=True)
            current_dict = {symbol: self.df_manager.dict_df[symbol]}
            
            self._update_progress(symbol, 30, "Processing strategy...")
            
            # Instantiate strategy
            strat = RSI_ADX_GPU(current_dict, self.risk, with_sizing=True, hyper=False)
            
            self._update_progress(symbol, 50, "Calculating indicators...")
            
            # Use the parameters specific to this symbol
            params = self.risk.symbol_params.get(symbol, [14, 14, 20, 80])
            strat.custom_indicator(strat.close, *params)
            
            self._update_progress(symbol, 70, "Analyzing signals...")
            
            # Process signals and generate graph
            signal_type = self._process_signal(strat, symbol)
            
            self._update_progress(symbol, 90, "Executing trade...")
            
            # Execute trade logic
            self.log(f"Executing {signal_type.upper()} signal for {symbol} at {float(strat.close.iloc[-1]):.2f}", signal_type)
            
            Trade(risk=self.risk, strat_object=strat, logbook=self.logbook)
            
            # Set next update time
            self.df_manager.set_next_update(symbol)
            next_update = self.df_manager.next_update_time[symbol].strftime("%H:%M:%S")
            
            self.log(f"Next update for {symbol} scheduled at {next_update}", "info")
            
            # Complete progress
            self._update_progress(symbol, 100, f"Complete. Next update: {next_update}")
            
            # Also emit trade status for other UI components
            self._emit_trade_status(symbol, signal_type, strat.close.iloc[-1], next_update)
            
            return True
            
        except Exception as e:
            self.log(f"Error processing {symbol}: {str(e)}", "error")
            self.log(traceback.format_exc(), "error")
            return False

    def _process_signal(self, strat, symbol):
        """Process signals and generate graph for a strategy."""
        # Get signal type (buy, sell, monitor)
        signal_type = "monitor"
        if hasattr(strat, 'signals') and len(strat.signals) > 0:
            if strat.signals[-1] == 1:
                signal_type = "buy"
            elif strat.signals[-1] == -1:
                signal_type = "sell"
        
        # Generate graph if callback exists
        if self.graph_callback:
            try:
                self._update_progress(symbol, 80, "Generating graph...")
                
                fig = strat.graph(self.graph_callback)
                self.log(f"Graph return value for {symbol}: {fig is not None}", "debug")
                
                # Send graph to frontend if available
                if fig:
                    try:
                        import plotly.io as pio
                        import base64
                        img_bytes = pio.to_image(fig, format="png")
                        graph_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        self.socketio.emit('graph_update', {
                            'symbol': symbol,
                            'graph': graph_base64
                        })
                        
                        self.log(f"Generated graph for {symbol}", "info")
                        return signal_type  # Return the signal type here
                    except Exception as e:
                        self.log(f"Error generating graph for {symbol}: {str(e)}", "error")
            except Exception as e:
                self.log(f"Error in graph generation for {symbol}: {str(e)}", "error")
                
        return signal_type

    def on_message(self):
        """Main trading logic executed on each cycle."""
        try:
            self.log(f"Trading cycle {self.counter} started", "info")
            self.socketio.emit('trading_status', {"status": "active"})

            # Create a list of symbols to avoid modification during iteration
            symbols = list(self.df_manager.dict_df.keys())
            
            for symbol in symbols:
                self.process_symbol(symbol)
                # Small delay between symbols to prevent rate limiting
                time.sleep(0.5)

            self.counter += 1
            self.log(f"Trading cycle {self.counter-1} completed", "info")
            
        except Exception as e:
            self.log(f"Error in trading cycle: {str(e)}", "error")
            self.log(traceback.format_exc(), "error")

    #-----------------
    # Async Methods
    #-----------------
    
    async def fetch_data_periodically(self):
        """Fetch data and execute trading logic periodically."""
        self.log("Starting periodic data fetching", "info")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Execute trading logic
                self.on_message()
                
                # Calculate sleep time based on execution time
                execution_time = time.time() - start_time
                sleep_time = max(10, self.kraken.time_to_wait - execution_time)  # At least 10 seconds
                
                self.log(f"Execution time: {execution_time:.2f}s. Sleeping for {sleep_time:.2f}s", "info")
                
                # Send heartbeat during wait time
                self._emit_heartbeat(sleep_time)
                
                # Wait before next cycle
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.log(f"Error in fetch cycle: {str(e)}", "error")
                self.log(traceback.format_exc(), "error")
                await asyncio.sleep(30)  # Wait 30 seconds before retry after error
        
        self.log("Periodic data fetching stopped", "info")

    async def main(self, graph_callback=None):
        """Main entry point for LiveTrader."""
        try:
            # Set graph callback
            self.graph_callback = graph_callback or (lambda fig, strat: None)
            if graph_callback is None:
                self.log("Warning: graph_callback is None. Assigning a default placeholder.", "info")
                
            # Update trading status
            self.socketio.emit('trading_status', {"status": "active"})
            self.log("Live trading started", "monitor")
            
            # Start periodic data fetching
            await self.fetch_data_periodically()
            
        except Exception as e:
            self.log(f"Critical error in LiveTrader: {str(e)}", "error")
            self.log(traceback.format_exc(), "error")
            self.socketio.emit('trading_status', {"status": "inactive"})
            
    #-----------------
    # Helper Methods
    #-----------------
    
    def _update_progress(self, symbol, progress, eta):
        """Update progress for a symbol."""
        self.socketio.emit('progress_update', {
            'symbol': symbol,
            'progress': progress,
            'eta': eta
        })
    
    def _emit_trade_status(self, symbol, signal_type, price, next_update):
        """Emit trade status for UI components."""
        self.socketio.emit('trade_status', {
            'symbol': symbol,
            'action': signal_type.upper(),
            'price': float(price),
            'time': dt.datetime.now().isoformat(),
            'next_update': next_update
        })
        
    def _emit_heartbeat(self, sleep_time):
        """Emit heartbeat with next cycle information."""
        self.socketio.emit('heartbeat', {
            'timestamp': dt.datetime.now().isoformat(),
            'next_cycle': (dt.datetime.now() + dt.timedelta(seconds=sleep_time)).isoformat()
        })
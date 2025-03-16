import time
import asyncio
import datetime as dt
from core.dataframe_manager import DF_Manager
from core.log import LinkedList
from core.strategies.strategy import Strategy
from core.strategies.single.rsi import RSI
from core.strategies.double.rsi_adx import RSI_ADX
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
from core.trade import Trade
from core.scanner import Scanner
from core.risk import Risk_Handler
from core.scanner import Scanner
from core.kraken_wrapper import Kraken
from flask_socketio import emit
import core.database_interaction as database_interaction
import os
import importlib.util
import inspect

class LiveTrader:
    def __init__(self, socketio):
        # self.granularity = 'ONE_MINUTE'
        # self.symbol = 'XBTUSD'
        self.socketio = socketio 
        self.counter = 0
        self.is_running = True  # Flag to stop the trading loop
        self.trades = []  # Store trade data
        # Initialize main components
        self.kraken = Kraken()
        self.risk = Risk_Handler(self.kraken)
        self.scanner = Scanner(client=self.kraken, socketio=self.socketio)
        self.df_manager = DF_Manager(self.scanner)

        self.scanner.assign_attribute(df_manager=self.df_manager)
        self.logbook = LinkedList()

        self.strat_classes = {}
        self.extract_classes_from_scripts()

        self.update_candle_data() 
        self.load_strategy_params_for_strategy()
        
    

    def get_status(self):
        """Returns the latest trade data and logs."""
        return {"status": "running" if self.is_running else "stopped", "trades": self.trades}

    def stop(self):
        """Stops live trading."""
        self.is_running = False

    async def fetch_data_periodically(self):
        """Fetch data and execute trades periodically."""
        while self.is_running:
            self.on_message()  # Runs trading logic
            await asyncio.sleep(10)  # Adjust sleep time as needed

    def extract_classes_from_scripts(self):
        strat_path = 'core/strategies'

        for root, _, files in os.walk(strat_path):
            for i, file in enumerate(files):
                if file not in ['strategy.py', 'combined_strategy.py']:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        module_name = file[:-3]

                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if obj.__module__ == module_name:
                                self.strat_classes[name] = obj


    def load_strategy_params_for_strategy(self):
        # Load strategy parameters for each symbol
        for symb in self.scanner.kraken_crypto:
            strat = RSI_ADX_GPU(dict_df=None, risk_object=self.risk)
            strat.symbol = symb
            params = database_interaction.get_best_params(
                strat,
                self.df_manager,
                live_trading=True,
                best_of_all_granularities=True,
                minimum_trades=4
            )
            print(f'Best params for {symb}: {params}')
            self.risk.symbol_params[symb] = params
            self.df_manager.set_next_update(symb, initial=True)

    def update_candle_data(self):
        try:
            def send_log(message):
                print(message)  # Keep logging in the console
                self.socketio.emit("log_update", {"message": message})  # Emit log to frontend
            
            self.scanner.coinbase.get_candles_for_db(
                self.scanner.coinbase_crypto,
                self.scanner.granularity,
                days=30,
                callback=send_log,  # Pass the logging callback
                socketio=self.socketio  # Pass socketio explicitly
            )
        except Exception as e:
            print(f"Error fetching candle data: {e}")
            self.socketio.emit("log_update", {"message": f"Error fetching candle data: {e}"})

    def on_message(self):
        self.socketio.emit('log_update', {
            "message": f"Trading cycle {self.counter} started",
            "type": "info",
            "timestamp": dt.datetime.now().isoformat()
        })

        for k in self.df_manager.dict_df.keys():
            # Skip if not time to update
            if dt.datetime.now() <= self.df_manager.next_update_time[k]:
                self.socketio.emit('log_update', {
                    "message": f"Skipping {k} - Next update at {self.df_manager.next_update_time[k].strftime('%H:%M:%S')}",
                    "type": "info",
                    "timestamp": dt.datetime.now().isoformat()
                })
                continue

            self.socketio.emit('log_update', {
                "message": f"Processing {k}",
                "type": "info",
                "timestamp": dt.datetime.now().isoformat()

            })
            
            try:
                # Update data
                self.df_manager.data_for_live_trade(symbol=k, update=True)
                current_dict = {k: self.df_manager.dict_df[k]}
                
                # Instantiate strategy
                strat = RSI_ADX_GPU(current_dict, self.risk, with_sizing=True, hyper=False)
                strat.custom_indicator(strat.close, *self.risk.symbol_params[k])
                
                # Get signal type (buy, sell, monitor)
                signal_type = "monitor"
                if strat.signals[-1] == 1:
                    signal_type = "buy"
                elif strat.signals[-1] == -1:
                    signal_type = "sell"
                    
                # Generate graph
                fig = strat.graph(self.graph_callback)
                
                # Send graph to frontend if available
                if fig:
                    try:
                        import plotly.io as pio
                        import base64
                        img_bytes = pio.to_image(fig, format="png")
                        graph_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        self.socketio.emit('graph_update', {
                            'symbol': k,
                            'graph': graph_base64
                        })
                        
                        self.socketio.emit('log_update', {
                            "message": f"Generated graph for {k}",
                            "type": "info",
                            "timestamp": dt.datetime.now().isoformat()
                        })
                    except Exception as e:
                        self.socketio.emit('log_update', {
                            "message": f"Error generating graph for {k}: {str(e)}",
                            "type": "error",
                            "timestamp": dt.datetime.now().isoformat()
                        })
                
                # Execute trade logic
                self.socketio.emit('log_update', {
                    "message": f"Executing {signal_type.upper()} signal for {k} at {float(strat.close.iloc[-1]):.2f}",
                    "type": signal_type,
                    "timestamp": dt.datetime.now().isoformat()
                })
                
                Trade(risk=self.risk, strat_object=strat, logbook=self.logbook)
                
                # Set next update time
                self.df_manager.set_next_update(k)
                next_update = self.df_manager.next_update_time[k].strftime("%H:%M:%S")
                
                self.socketio.emit('log_update', {
                    "message": f"Next update for {k} scheduled at {next_update}",
                    "type": "info",
                    "timestamp": dt.datetime.now().isoformat()
                })
                
                # Also emit trade status for other UI components
                self.socketio.emit('trade_status', {
                    'symbol': k,
                    'action': signal_type.upper(),
                    'price': float(strat.close.iloc[-1]),
                    'time': dt.datetime.now().isoformat(),
                    'next_update': next_update
                })
                
            except Exception as e:
                self.socketio.emit('log_update', {
                    "message": f"Error processing {k}: {str(e)}",
                    "type": "error",
                    "timestamp": dt.datetime.now().isoformat()
                })
            
            time.sleep(0.5)

        self.counter += 1
        
        self.socketio.emit('log_update', {
            "message": f"Trading cycle {self.counter-1} completed",
            "type": "info",
            "timestamp": dt.datetime.now().isoformat()
        })

    async def fetch_data_periodically(self):
        while True:
            start_time = time.time()

            self.on_message()

            execution_time = time.time() - start_time
            sleep_time = max(0, self.kraken.time_to_wait - execution_time)

            print(f"Execution time: {execution_time:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.\n")
            await asyncio.sleep(sleep_time)

    async def main(self, graph_callback=None):
        if graph_callback is None:
            print("Warning: graph_callback is None. Assigning a default placeholder.")
            self.graph_callback = lambda fig, strat: None  # Avoid crash
        else:
            self.graph_callback = graph_callback

        await self.fetch_data_periodically()



# trader = LiveTrader()
# asyncio.run(trader.main())

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import core.utils.utils as utils
import datetime as dt
import numpy as np
import database.database_interaction as database_interaction
import time
import gc
import gc
import io
import base64
from coinbase.rest import RESTClient
from core.strategies.strategy import Strategy
from core.strategies.single.efratio import EFratio
from core.strategies.single.vwap import Vwap
from core.strategies.single.rsi import RSI
from core.strategies.single.atr import ATR
from core.strategies.single.macd import MACD
from core.strategies.single.kama import Kama
from core.strategies.single.adx import ADX
from core.strategies.single.bollinger import BollingerBands
from core.strategies.double.rsi_adx import RSI_ADX
from core.strategies.combined_strategy import Combined_Strategy
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
from core.strategies.gpu_optimized.NP.rsi_adx_np import RSI_ADX_NP
from core.strategies.gpu_optimized.GPU.bollinger_vwap_gpu import BollingerBands_VWAP_GPU
from core.risk import Risk_Handler
from core.log import LinkedList
from core.hyper import Hyper
from concurrent.futures import ThreadPoolExecutor
import plotly
from io import BytesIO
from PIL import Image
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
import logging
# logging.basicConfig(level=logging.DEBUG)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)

# Set your own logging level if needed
logging.basicConfig(level=logging.INFO) 
import tracemalloc






class Backtest():
    def __init__(self):
        self.symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD', 'LINK-USD', 'XLM-USD'] #
        self.granularities = ['ONE_MINUTE','FIVE_MINUTE','FIFTEEN_MINUTE','THIRTY_MINUTE','ONE_HOUR','TWO_HOUR','SIX_HOUR','ONE_DAY']#,
        self.product = ['XTZ-USD']
        self.granularity = 'ONE_MINUTE'
        

    def run_multiple_strategy(self, symbol: str, granularity: str, num_days: int, sizing:bool, graph_callback =None, strategies:list = []):
        logbook = LinkedList()
        dict_df = database_interaction.get_historical_from_db(granularity=granularity, symbols=symbol, num_days=num_days)
        risk = Risk_Handler()

        #for sym, df in df_dict.items():
            #current_dict_df = {sym:df}

        combined_strat = Combined_Strategy(dict_df, risk, sizing,*strategies)
        combined_strat.generate_combined_signals()
        combined_strat.graph(graph_callback)
        combined_strat.generate_backtest()
        print(combined_strat.portfolio.stats(silence_warnings=True))

        logbook.insert_beginning(combined_strat)
        
        logbook.export_multiple_pf_to_db(is_combined=True)



    def run_basic_backtest(self, symbol, granularity, strategy_obj, num_days, sizing, best_params=True, graph_callback=None):
        print(symbol, granularity, strategy_obj, num_days, sizing, best_params, graph_callback)
        print("Getting historical data...")
        dict_df = database_interaction.get_historical_from_db(
            granularity=granularity,
            symbols=symbol,
            num_days=num_days
        )
        if not dict_df:
            raise ValueError("No historical data found for the given parameters.")
        print("Fetched historical data:", dict_df)

        stats = {}
        graph_base64 = None

        logging.info(f"Running backtest for {len(dict_df)} symbols")

        for key, value in dict_df.items():
            try:
                current_dict = {key: value}
                risk = Risk_Handler()
                strat = strategy_obj(
                    dict_df=current_dict,
                    risk_object=risk,
                    with_sizing=sizing,
                )

                
                if best_params:
                    print("getting best parameters ...")
                    logging.info(f"Getting best parameters for {strat.__class__.__name__}")
                    params = database_interaction.get_best_params(strat, minimum_trades=4)
                    logging.info(f"Best params: {params}")
                    strat.custom_indicator(None, *params)
                else:
                    try:
                        print("running custom indicator ...")
                        logging.info(f"Running custom indicator for {strat.__class__.__name__}")
                        strat.custom_indicator()
                    except Exception as e:
                        logging.error(f"Error running custom indicator: {e}")
                        print(f"Error running custom indicator: {e}")
                        continue
                try:
                    print("graphing ...")
                    logging.info(f"Graphing for {strat.__class__.__name__}")
                    strat.graph()
                except Exception as e:
                    logging.error(f"Error graphing: {e}")
                    print(f"Error graphing: {e}")    

                try:
                    logging.info(f"Generating backtest for {strat.__class__.__name__}") 
                    print("generating backtest...")
                    strat.generate_backtest()
                    logging.info(f"Portfolio after backtest: {strat.portfolio}")
                    pf = strat.portfolio
                    print("assigning portfolio")
                    stats = pf.stats().to_dict()
                    print("generating stats ...")
                    print(f"stats: ")
                    print(stats)
                except Exception as e:
                    print(f"Error in backtest iteration for {key}: {e}")

                if graph_callback:
                    try:
                        fig = pf.plot(subplots=['orders'])
                        graph_base64 = graph_callback(fig)
                    except Exception as e:
                        logging.error(f"Graph generation failed: {e}")
                        print(f"Graph generation failed: {e}")
                        graph_base64 = ""

            except Exception as e:
                print(f"Error in backtest iteration for {key}: {e}")

        if not stats:
            stats = {"error": "No stats generated"}
        if not graph_base64:
            logging.info("No graph generated")
            graph_base64 = ""

        return stats, graph_base64


            # fig = pf.plot(subplots = [
            # 'orders',
            # 'trade_pnl',
            # 'cum_returns',
            # 'drawdowns',
            # 'underwater',
            # 'gross_exposure'])
            # fig.update_layout(
            #     title={
            #         'text': f"{strat.__class__.__name__} for {strat.symbol} on {strat.granularity} timeframe",  # Replace with your desired title
            #         'x': 0.5,  # Centers the title
            #         'xanchor': 'center',
            #         'yanchor': 'top'
            #     },
            #     margin={
            #         't': 100  # Adjust the top margin to create space for the title
            #     }
            # )
                

    def run_hyper(self, strategy_obj, param_ranges):
        """Runs hyperparameter optimization for a given strategy and parameter ranges."""
        
        risk = Risk_Handler()

        for granularity in self.granularities:
            days = 25 if granularity == 'ONE_MINUTE' else \
                100 if granularity == 'FIVE_MINUTE' else \
                250 if granularity == 'FIFTEEN_MINUTE' else 365

            dict_df = database_interaction.get_historical_from_db(
                granularity=granularity,
                symbols=self.symbols,
                num_days=days  # Use `days` based on granularity
            )

            print(f'...Running hyper on {len(self.symbols)} symbols')

            start_time = time.time()
            for i, (key, value) in enumerate(dict_df.items()):
                current_dict = {key: value}

                # Initialize the provided strategy dynamically
                strat = strategy_obj(current_dict, risk_object=risk, with_sizing=True)

                # Run hyperparameter optimization with dynamic ranges
                hyper = Hyper(
                    strategy_object=strat,
                    close=strat.close,
                    **param_ranges  # Pass dynamic parameter ranges
                )

                # Export hyperparameter results to the database
                database_interaction.export_hyper_to_db(
                    strategy=strat,
                    hyper=hyper
                )

                print(f"Execution Time: {time.time() - start_time}")
                utils.progress_bar_with_eta(
                    progress=i,
                    data=dict_df.keys(),
                    start_time=start_time
                )

                # Cleanup to free memory
                del hyper
                gc.collect()

    def chunk_dataframe(slef, df, chunk_size):
        """
        Splits a DataFrame into smaller chunks of specified size.

        Args:
            df (pd.DataFrame): The DataFrame to chunk.
            chunk_size (int): The size of each chunk.

        Yields:
            pd.DataFrame: A chunk of the original DataFrame.
        """
        for start_idx in range(0, len(df), chunk_size):
            yield df.iloc[start_idx:start_idx + chunk_size]

    def process_symbol(self, symbol, granularity, num_days, train_test_split, strategy_class, param_ranges, risk, chunk_size=10000):
        try:
            logging.info(f"Fetching historical data for {symbol} at {granularity}")
            dict_df = database_interaction.get_historical_from_db(
                granularity=granularity,
                symbols=symbol,
                num_days=num_days
            )
            if not dict_df or symbol not in dict_df:
                logging.warning(f"No data available for {symbol} at {granularity}")
                return None

            data = dict_df[symbol].astype(np.float32)  # Reduce precision to optimize memory
            logging.info(f"Fetched data for {symbol}: {data.shape[0]} rows")

            results = []
            
            for chunk in self.chunk_dataframe(data, chunk_size):
                total_len = len(chunk)
                train_len = int(total_len * train_test_split)

                if train_len < 1 or total_len - train_len < 1:
                    logging.warning(f"Insufficient data for training/testing split in chunk for {symbol} at {granularity}")
                    continue

                step_size = max(1, (total_len - train_len) // 10)

                # Walk-forward analysis within the chunk
                for start_idx in range(0, total_len - train_len, step_size):
                    try:
                        logging.info(f"Processing window: start_idx={start_idx}, train_len={train_len}")
                        train_data = chunk.iloc[start_idx: start_idx + train_len]
                        test_data = chunk.iloc[start_idx + train_len: start_idx + 2 * train_len]

                        if len(test_data) < 1:
                            logging.warning(f"Skipping incomplete test window in chunk for {symbol} at {granularity}")
                            break

                        gc.collect()
                        strat_train = strategy_class(
                            dict_df={symbol: train_data},
                            risk_object=risk,
                            with_sizing=True
                        )
                        hyper = Hyper(
                            strategy_object=strat_train,
                            close=strat_train.close,
                            **param_ranges
                        )

                        database_interaction.export_hyper_to_db(
                            strategy=strat_train,
                            hyper=hyper
                        )

                        best_params = database_interaction.get_best_params(strat_train)
                        logging.info(f"Best params for {symbol} at {granularity}: {best_params}")

                        strat_test = strategy_class(
                            dict_df={symbol: test_data},
                            risk_object=risk,
                            with_sizing=True
                        )
                        strat_test.custom_indicator(None, *best_params)
                        strat_test.generate_backtest()
                        stats = strat_test.portfolio.stats(silence_warnings=True).to_dict()

                        filtered_stats = {
                            key: stats[key]
                            for key in [
                                'Total Return [%]',
                                'Total Trades',
                                'Win Rate [%]',
                                'Best Trade [%]',
                                'Worst Trade [%]',
                                'Avg Winning Trade [%]',
                                'Avg Losing Trade [%]',
                            ]
                            if key in stats
                        }

                        filtered_stats.update({
                            "symbol": symbol,
                            "granularity": granularity,
                            "start_idx": start_idx,
                            "end_idx": start_idx + train_len,
                        })

                        results.append(filtered_stats)

                        del train_data, test_data, strat_train, strat_test, hyper
                        gc.collect()

                    except MemoryError:
                        logging.error(f"MemoryError at window: start_idx={start_idx}. Skipping window.", exc_info=True)
                        continue
                    except Exception as e:
                        logging.error(f"Error during walk-forward window processing in chunk: {e}", exc_info=True)

            return results

        except Exception as e:
            logging.error(f"Error processing {symbol} at {granularity}: {e}", exc_info=True)
            return None

    def run_optimization_tests(self, strategy_class, param_ranges, train_test_split=0.7):
        tracemalloc.start()
        # print("Printing param ranges...")
        # print(param_ranges)

        results = []
        risk = Risk_Handler()

        for granularity in self.granularites:
            num_days = {
                'ONE_MINUTE': 25,
                'FIVE_MINUTE': 125,
                'FIFTEEN_MINUTE': 375,
                'THIRTY_MINUTE': 750,
                'ONE_HOUR': 1500,
                'TWO_HOUR': 3000,
                'SIX_HOUR': 3000,
                'ONE_DAY': 3000,
            }.get(granularity, 3000)

            logging.info(f"Processing granularity: {granularity}, num_days: {num_days}")

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(self.process_symbol, symbol, granularity, num_days, train_test_split, strategy_class, param_ranges, risk)
                    for symbol in self.symbols
                ]
                for future in futures:
                    result = future.result()
                    if result:
                        results.extend(result)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        if results_df.empty:
            logging.warning("Results DataFrame is empty. Nothing to export.")
            return

        # Identify the row with the best parameters based on a metric (e.g., 'Total Return [%]')
        if 'Total Return [%]' in results_df.columns:
            best_index = results_df['Total Return [%]'].idxmax()
            results_df['is_best'] = False  # Add a column to mark the best result
            results_df.loc[best_index, 'is_best'] = True
            logging.info(f"Best parameters: {results_df.loc[best_index]}")

        logging.info(f"Results DataFrame:\n{results_df.head()}")

        try:
            database_interaction.export_optimization_results(results_df)
            logging.info("Optimization results exported successfully.")
        except Exception as e:
            logging.error(f"Error exporting optimization results: {e}", exc_info=True)

        return results_df



    def run_optuna_backtest(self, symbol, granularity, strategy_obj, num_days, sizing, params):
        print(f"Running Optuna Backtest for {symbol} on {granularity}")
        print(f"Parameters: {params}")
        dict_df = database_interaction.get_historical_from_db(
            granularity=granularity,
            symbols=symbol,
            num_days=num_days
        )
        if not dict_df:
            raise ValueError("No historical data found for the given parameters.")

        for key, value in dict_df.items():
            try:
                current_dict = {key: value}
                risk = Risk_Handler()

                # Initialize strategy with the dictionary and sizing
                strat = strategy_obj(
                    dict_df=current_dict,
                    risk_object=risk,
                    with_sizing=sizing,
                )

                # Apply the Optuna parameters
                strat.custom_indicator(**params)

                # Run backtest
                strat.generate_backtest()
                pf = strat.portfolio

                # Return relevant performance metrics
                stats = pf.stats().to_dict()
                print(f"Stats for {symbol}: {stats}")
                return stats
            except Exception as e:
                print(f"Error in backtest for {key}: {e}")
                return {"error": str(e)}

        return {"error": "No data processed"}
    


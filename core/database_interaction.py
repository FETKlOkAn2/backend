import pandas as pd
import core.utils.utils as utils
import inspect
import numpy as np
import sys
import time
from datetime import timedelta
import gc
import sys
from datetime import datetime
from threading import Lock
import os
import logging
import json
import pandas as pd
import logging
import sqlite3 as sql
import datetime
import json
import logging
import os
import sys
import secrets
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid

from core.utils.encryption_utils import encrypt_api_key, decrypt_api_key
from core.utils.api_key_tester import ApiKeyTester

# Suppress debug logs from libraries
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)

# Set your own logging level if needed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ai_backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__) # Set to DEBUG, INFO, WARNING, ERROR, CRITICAL as needed


from dotenv import load_dotenv
load_dotenv(override=True)

import os
from sqlalchemy import create_engine, text
import pandas as pd

#db_path = os.getenv('DATABASE_PATH')

class MSSQLDatabase:
    def __init__(self, db_file_name):
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT", "1433")
        self.db = db_file_name
        self.driver = "ODBC+Driver+18+for+SQL+Server"
        self.engine = self.get_engine()
        self.lock = Lock()

    def table_exists(self, schema_name, table_name):
        """Check if a table exists in the specified schema."""
        sql = f"SELECT CASE WHEN OBJECT_ID('[{schema_name}].[{table_name}]', 'U') IS NOT NULL THEN 1 ELSE 0 END as table_exists"
        result = self.read_sql_query(sql)
        return result.iloc[0]['table_exists'] == 1

    def get_engine(self):
        # Fixed connection string format with properly escaped parameters
        url = (
            f"mssql+pyodbc://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}?"
            f"driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
        )
        return create_engine(url, pool_pre_ping=True)

    def execute_sql(self, sql_text, params=None):
        """Execute SQL with proper parameter handling."""
        with self.lock, self.engine.begin() as conn:
            if params:
                if isinstance(params, (list, tuple)):
                    # Convert positional parameters to named parameters
                    param_count = sql_text.count('?')
                    if len(params) != param_count:
                        raise ValueError(f"Parameter count mismatch: expected {param_count}, got {len(params)}")
                    
                    # Create named parameter dictionary
                    param_dict = {f'param_{i}': param for i, param in enumerate(params)}
                    
                    # Replace ? placeholders with named parameters
                    modified_sql = sql_text
                    for i in range(param_count):
                        modified_sql = modified_sql.replace('?', f':param_{i}', 1)
                    
                    logger.debug(f"Modified SQL: {modified_sql}")
                    logger.debug(f"Parameters: {param_dict}")
                    
                    return conn.execute(text(modified_sql), param_dict)
                else:
                    # Dictionary parameters - use as-is
                    return conn.execute(text(sql_text), params)
            else:
                return conn.execute(text(sql_text))

    def read_sql_query(self, sql_text, params=None):
        """Execute SQL query and return DataFrame with proper parameter handling."""
        with self.lock, self.engine.begin() as conn:
            if params:
                if isinstance(params, (list, tuple)):
                    # Convert positional parameters to named parameters
                    param_count = sql_text.count('?')
                    if len(params) != param_count:
                        raise ValueError(f"Parameter count mismatch: expected {param_count}, got {len(params)}")
                    
                    # Create named parameter dictionary
                    param_dict = {f'param_{i}': param for i, param in enumerate(params)}
                    
                    # Replace ? placeholders with named parameters
                    modified_sql = sql_text
                    for i in range(param_count):
                        modified_sql = modified_sql.replace('?', f':param_{i}', 1)
                    
                    logger.debug(f"Modified SQL: {modified_sql}")
                    logger.debug(f"Parameters: {param_dict}")
                    
                    return pd.read_sql_query(text(modified_sql), conn, params=param_dict)
                else:
                    # Dictionary parameters - use as-is
                    return pd.read_sql_query(text(sql_text), conn, params=params)
            else:
                return pd.read_sql_query(text(sql_text), conn)

    def to_sql(self, df, table_name, schema=None, if_exists='append', index=False):
        """Write DataFrame to SQL table."""
        with self.lock:
            df.to_sql(
                table_name, 
                self.engine, 
                schema=schema,
                if_exists=if_exists, 
                index=index, 
                chunksize=100
            )

    def close(self):
        """Close the database engine."""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()

    def create_table_if_not_exists(self, table_name, df):
        """
        Creates a table if it doesn't exist based on the DataFrame's schema.
        """
        try:
            # Check if the table exists
            table_exists_query = f"IF OBJECT_ID('{table_name}', 'U') IS NOT NULL SELECT 1 ELSE SELECT 0;"
            result = self.read_sql_query(table_exists_query)
            table_exists = result.iloc[0, 0] == 1

            if not table_exists:
                # Build the CREATE TABLE query based on DataFrame's schema
                columns = df.columns
                dtypes = df.dtypes
                sql_dtypes = []
                for col in columns:
                    dtype = dtypes[col]
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_dtype = 'INT'
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_dtype = 'FLOAT'
                    elif dtype == 'datetime64[ns]':
                        sql_dtype = 'DATETIME2'
                    else:
                        sql_dtype = 'NVARCHAR(MAX)'  # Use NVARCHAR(MAX) for TEXT

                    sql_dtypes.append(f'[{col}] {sql_dtype}')

                create_table_query = f"CREATE TABLE [{table_name}] ("
                create_table_query += ', '.join(sql_dtypes)
                create_table_query += ");"

                self.execute_sql(create_table_query)
                logger.info(f"Table {table_name} created successfully.")
            else:
                logger.info(f"Table {table_name} already exists.")
        except Exception as e:
            logger.error(f"Error occurred while creating table {table_name}: {e}")

def get_historical_from_db(granularity, symbols: list = [], num_days: int = None, convert=False):
    original_symbol = symbols

    if convert:
        symbols = utils.convert_symbols(lone_symbol=symbols)

    db_file = f'{granularity}'
    db = MSSQLDatabase(db_file)

    try:
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE';"
        tables = db.read_sql_query(query)
        tables_data = {}
        
        logger.info(f"Found {len(tables)} tables in {granularity}")
        
        if tables.empty:
            logger.warning(f"No tables found in {granularity}")
            return {}
        
        for table in tables['TABLE_NAME']:
            clean_table_name = '-'.join(table.split('_')[:2])
            
            # If symbols are provided, skip tables that are not in the symbol list
            if symbols and clean_table_name not in symbols:
                continue

            try:
                # Retrieve data from the table
                data = db.read_sql_query(f'SELECT * FROM "{table}"')
                
                if data.empty:
                    logger.warning(f"Table {table} is empty!")
                    continue
                    
                logger.info(f"Retrieved {len(data)} rows from table {table}")
                
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data.set_index('date', inplace=True)

                if num_days is not None:
                    last_date = data.index.max()  # Find the most recent date in the dataset
                    start_date = last_date - pd.Timedelta(days=num_days)
                    data = data.loc[data.index >= start_date]
                    logger.info(f"Filtered data to last {num_days} days, now have {len(data)} rows")

                # Store the data in the dictionary
                if convert:
                    tables_data[original_symbol] = data
                else:
                    tables_data[clean_table_name] = data
                    
            except Exception as e:
                logger.error(f"Error processing table {table}: {str(e)}")
                continue

        # Validate the result
        if not tables_data:
            logger.warning(f"No data found for {symbols} in {granularity}")
        else:
            for symbol, data in tables_data.items():
                logger.info(f"Successfully retrieved {len(data)} rows for {symbol}")
                
        return tables_data
        
    except Exception as e:
        logger.error(f"Error in get_historical_from_db: {str(e)}")
        return {}


def get_best_params(strategy_object, df_manager=None, live_trading=False, best_of_all_granularities=False, minimum_trades=None, with_lowest_losing_average=False):
    granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY']
    
    db = MSSQLDatabase('hyper')

    try:
        if best_of_all_granularities:
            best_results = []
            best_granularity = ''
            best_return = float('-inf')  # To track the best return
            
            for granularity in granularities:
                try:
                    table = f"RSI_ADX_GPU_{granularity}" if strategy_object.__class__.__name__ == "RSI_ADX_NP" else f"{strategy_object.__class__.__name__}_{granularity}"
                    
                    params = inspect.signature(strategy_object.custom_indicator)
                    param_keys = list(dict(params.parameters).keys())[1:]  # Exclude 'self'
                    parameters = ', '.join(param_keys)

                    symbol = (
                        utils.convert_symbols(strategy_object=strategy_object)
                        if strategy_object.risk_object.client is not None else strategy_object.symbol
                    )

                    query = f'SELECT {parameters}, MAX("Total Return [%]") AS max_return FROM {table} WHERE symbol="{symbol}"'
                    if minimum_trades is not None:
                        query += f' AND "Total Trades" >= {minimum_trades}'

                    result = db.read_sql_query(query)

                    if result.empty or all(result.iloc[0].isnull()):
                        continue

                    max_return = result['max_return'].iloc[0]
                    list_results = [
                        result[param].iloc[0] if param in result.columns else None for param in param_keys
                    ]

                    # Update the best results if this granularity has a higher return
                    if max_return > best_return:
                        print(f"New best granularity: {granularity} with return: {max_return}")
                        best_return = max_return
                        best_results = list_results
                        best_granularity = granularity

                except Exception as e:
                    print(f'Error processing granularity {granularity}:', e)

            try:
                if best_granularity and (strategy_object.granularity != best_granularity or strategy_object.granularity is None):
                    print('Granularity has changed. Updating strategy with new data.')
                    print(f"Best granularity: {best_granularity}")

                    if live_trading:
                        dict_df = get_historical_from_db(
                            granularity=best_granularity,
                            symbols=strategy_object.symbol,
                            num_days=30,
                            convert=True
                        )
                    else:
                        num_days = int((strategy_object.df.index[-1] - strategy_object.df.index[0]).total_seconds() // 86400)
                        dict_df = get_historical_from_db(
                            granularity=best_granularity,
                            symbols=strategy_object.symbol,
                            num_days=num_days
                        )

                    if hasattr(strategy_object, 'df'):
                        strategy_object.update(dict_df)
                    
                    if live_trading and df_manager:
                        df_manager.add_to_manager(dict_df)
                        df_manager.products_granularity[list(dict_df.keys())[0]] = best_granularity

                print(f"Final best results: {best_results[:-1]}")
                best_results = best_results[:-1]  # Exclude the return value for parameters
            except Exception as e:
                print('Error updating strategy or DF manager:', e)

        else:
            try:
                table = f"{strategy_object.__class__.__name__}_{strategy_object.granularity}"
                params = inspect.signature(strategy_object.custom_indicator)
                param_keys = list(dict(params.parameters).keys())[1:]  # Exclude 'self'
                parameters = ', '.join(param_keys)

                symbol = (
                    utils.convert_symbols(strategy_object=strategy_object)
                    if strategy_object.risk_object.client is not None else strategy_object.symbol
                )

                query = f'SELECT {parameters}, MAX("Total Return [%]") AS max_return FROM {table} WHERE symbol="{symbol}"'
                if minimum_trades is not None:
                    query += f' AND "Total Trades" >= {minimum_trades}'

                result = db.read_sql_query(query)

                list_results = [
                    result[param].iloc[0] for param in param_keys if param in result.columns
                ]
                print(f"Final results for single granularity: {list_results}")
            except Exception as e:
                print('Error querying for specific granularity:', e)
                return None

    finally:
       pass

    return best_results if best_of_all_granularities else list_results


def export_optimization_results(df):
    db = MSSQLDatabase('optimization')
    try:
        print("Connected to database successfully.")
        db.create_table_if_not_exists('optimization_results', df)
        
        print("Verifying DataFrame types:")
        print(df.dtypes)
        
        print("Exporting results to the database...")
        db.to_sql(df, 'optimization_results', if_exists='append', index=False)
        print("Data exported successfully.")
    except Exception as e:
        print(f"Error occurred while exporting optimization results: {e}")
    finally:
        pass

def get_best_params_without_df(strategy_name, symbol, granularity, minimum_trades=None):
    db = MSSQLDatabase('hyper')

    try:
        # Use strategy_name directly instead of strategy_object.__class__.__name__
        table = f"{strategy_name}_{granularity}"
        
        # Determine parameter names for RSI_ADX_GPU (or other strategies if needed)
        if strategy_name == 'RSI_ADX_GPU' or strategy_name == 'RSI_ADX_NP':
            parameters = 'rsi_window, buy_threshold, sell_threshold, adx_time_period, adx_buy_threshold'
        else:
            # If you have other strategies, add their parameter lists here
            logger.warning(f"Parameter list not defined for strategy {strategy_name}")
            return None

        query = f'SELECT {parameters}, MAX("Total Return [%]") AS max_return FROM {table} WHERE symbol="{symbol}"'
        if minimum_trades is not None:
            query += f' AND "Total Trades" >= {minimum_trades}'
        
        logger.debug(f"Executing query: {query}")

        result = db.read_sql_query(query)
        
        if result.empty or all(result.iloc[0].isnull()):
            logger.info(f"No valid results for strategy {strategy_name} with granularity {granularity}")
            return None

        # Extract parameter values (excluding the max_return column)
        param_columns = result.columns[:-1]  # All columns except the last one (max_return)
        list_results = [result[param].iloc[0] for param in param_columns]
        
        logger.info(f"Found best parameters for {strategy_name} ({granularity}): {list_results}")
        return list_results

    except Exception as e:
        logger.error(f'Error querying database for {strategy_name} parameters: {e}')
        return None
    finally:
        pass

import json
import datetime
import pandas as pd

def save_backtest(email, symbol, strategy, result, date):
    # Convert non-serializable objects in 'result' to serializable types
    def make_serializable(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Convert Timestamps to ISO 8601 strings
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()  # Handle datetime objects
        elif isinstance(obj, datetime.date):
            return obj.isoformat()  # Handle date objects
        return obj

    # Apply conversion to the entire 'result' dictionary
    serializable_result = {key: make_serializable(value) for key, value in result.items()}

    # Serialize the processed result into JSON
    result_json = json.dumps(serializable_result)
    
    db = MSSQLDatabase('backtest')
    df = pd.DataFrame(columns=['email', 'symbol', 'strategy', 'result', 'date'])
    db.create_table_if_not_exists('backtest',df)

    data = {'email': email, 'symbol': symbol, 'strategy': strategy, 'result': result_json, 'date': date}
    df = pd.DataFrame([data])
    db.to_sql(df, 'backtest', if_exists='append', index=False)


def export_hyper_to_db(strategy: object, hyper: object):
    stats_to_export = [
            'Total Return [%]',
            'Total Trades',
            'Win Rate [%]',
            'Best Trade [%]',
            'Worst Trade [%]',
            'Avg Winning Trade [%]',
            'Avg Losing Trade [%]'
        ]
    
    data = hyper.pf.stats(silence_warnings=True,
                          agg_func=None)
    
    # dont forget to change this when using hyper !!!
    db = MSSQLDatabase('hyper')

    symbol = strategy.symbol
    granularity = strategy.granularity
    params = inspect.signature(strategy.custom_indicator)
    
    params = list(dict(params.parameters).keys())[1:]
    combined_df = pd.DataFrame()


    for i in range(len(data)):
        stats = data.iloc[i]
        print(f"Stats: {stats}")
        print(f"Stats name: {stats.name}")
        backtest_dict = {'symbol': symbol}
        
        # If stats.name is not iterable, wrap it in a list.
        name_list = stats.name if isinstance(stats.name, (list, tuple)) else [stats.name]
        
        for j, param in enumerate(params):
            # Check if name_list has enough elements
            if j < len(name_list):
                print(j, param, name_list[j])
                backtest_dict[param] = name_list[j]
            else:
                print(f"Warning: Not enough values in stats.name for parameter '{param}'.")
        
        for key, value in stats.items():
            if key in stats_to_export:
                backtest_dict[key] = value

        combined_df = pd.concat([combined_df, pd.DataFrame([backtest_dict])])


    # Prepare table name
    table_name = f"{strategy.__class__.__name__}_{granularity}"

    # Create table if not exists
    db.create_table_if_not_exists(table_name, combined_df)

    # Check for existing data and update
    query = f'SELECT * FROM "{table_name}" WHERE symbol = ?'
    existing_data = db.read_sql_query(query, params={'symbol': symbol})

    if not existing_data.empty:
        delete_query = f'DELETE FROM "{table_name}" WHERE symbol = ?'
        db.execute_sql(delete_query, params={'symbol': symbol})
        db.to_sql(combined_df, table_name, if_exists='append', index=False)
    else:
        db.to_sql(combined_df, table_name, if_exists='append', index=False)

    return

def export_historical_to_db(dict_df, granularity):
    db = MSSQLDatabase(f'{granularity}')
    
    for symbol, df in dict_df.items():
        # Replace hyphens with underscores in symbol
        symbol_pattern = symbol.replace('-', '_')
        # Construct the new table name with the updated date range.
        first_date = df.index[0].date()
        last_date = df.index[-1].date()
        new_table_name = f'{symbol_pattern}_{first_date}_TO_{last_date}'.replace('-', '_')
        
        # Escape underscores in the symbol pattern for the LIKE query
        symbol_pattern_escaped = symbol_pattern.replace('_', r'\_')
        pattern = f'{symbol_pattern_escaped}\\_%'
        
        # Fetch all existing tables matching the pattern
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_NAME LIKE ? ESCAPE '\\'"
        existing_tables = db.read_sql_query(query, params={'pattern': pattern})
        
        # Drop all existing tables that match the pattern
        for existing_table in existing_tables['TABLE_NAME']:
            drop_table_query = f'DROP TABLE IF EXISTS \"{existing_table}\"'
            db.execute_sql(drop_table_query)
        
        # Write the DataFrame to the new table
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
        db.to_sql(df, new_table_name, if_exists='replace', index=True)
    return


def resample_dataframe_from_db(granularity='ONE_MINUTE', callback=None, socketio=None):
    """
    Resamples data from the database for different timeframes based on the granularity.
    """
    if callback:
        callback(f"...Resampling Database")
    print("\n...Resampling Database")
    
    times_to_resample = {
        'FIVE_MINUTE': '5min',
        'FIFTEEN_MINUTE': '15min',
        'THIRTY_MINUTE': '30min',
        'ONE_HOUR': '1h',
        'TWO_HOUR': '2h',
        'SIX_HOUR': '6h',
        'ONE_DAY': '1d'
    }

    dict_df = get_historical_from_db(granularity=granularity)

    resampled_dict_df = {}
    start_time = time.time()
    for i, key in enumerate(times_to_resample.keys()):
        value = times_to_resample[key]
        for symbol, df in dict_df.items():
            print(symbol, df)
            df = df.sort_index()

            df_resampled = df.resample(value).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            df_resampled.dropna(inplace=True)

            resampled_dict_df[symbol] = df_resampled
        
        export_historical_to_db(resampled_dict_df, granularity=key)

        utils.progress_bar_with_eta(i, data=times_to_resample.keys(), start_time=start_time, socketio=socketio, symbol=key, socket_invoker="resampling_progress")







############################################### for backtest;however, will need to re-do...maybe ################################################################


def get_metrics_from_backtest(strategy_object, multiple=False, multiple_dict=None):

    symbol = strategy_object.symbol
    portfolio = strategy_object.portfolio
    backtest_dict = {'symbol': symbol}
    if multiple_dict:
        backtest_dict = multiple_dict
    if not multiple:
        params = inspect.signature(strategy_object.custom_indicator)
        params = list(dict(params.parameters).keys())[1:]
        value_list = []

        for param in params:
            value = getattr(strategy_object, param, None)
            backtest_dict[param] = value
            value_list.append(value)
    

    stats_to_export = [
        'Total Return [%]',
        'Total Trades',
        'Win Rate [%]',
        'Best Trade [%]',
        'Worst Trade [%]',
        'Avg Winning Trade [%]',
        'Avg Losing Trade [%]'
    ]

    for key, value in portfolio.stats().items():
        if key in stats_to_export:
            backtest_dict[key] = value

    backtest_df = pd.DataFrame([backtest_dict])

    if not multiple:
        return backtest_df, value_list, params
    return backtest_df




def export_backtest_to_db(object, multiple_table_name=None, is_combined=False):
    """Exports strategy backtest results to the database."""
    db = MSSQLDatabase('backtest')

    if not isinstance(object, pd.DataFrame):
        # Handle Strategy object
        strategy_object = object
        granularity = strategy_object.granularity
        backtest_df, value_list, params = get_metrics_from_backtest(strategy_object)
        symbol = backtest_df['symbol'].unique()[0]

        # Determine table name
        if is_combined and hasattr(strategy_object, "strategies"):
            strat_names = "_".join([str(type(s).__name__) for s in strategy_object.strategies])
            table_name = f"{strat_names}_COMBINED_{granularity}"
        else:
            table_name = f"{strategy_object.__class__.__name__}_{granularity}"

        # Ensure the table exists
        db.create_table_if_not_exists(table_name, backtest_df)

        # Prepare the DELETE query
        delete_query = f'DELETE FROM "{table_name}" WHERE symbol = @symbol'
        param_query = {'symbol': symbol}

    else:
        # Handle DataFrame directly
        backtest_df = object
        granularity = "default_granularity"  # Fallback granularity if not provided
        table_name = f"{multiple_table_name}_{granularity}"

        # Ensure the table exists
        db.create_table_if_not_exists(table_name, backtest_df)

        # Prepare the DELETE query
        symbol = backtest_df['symbol'].unique()[0]
        delete_query = f'DELETE FROM "{table_name}" WHERE symbol = @symbol'
        param_query = {'symbol': symbol}

    # Step 1: Delete existing rows with the same symbol
    db.execute_sql(delete_query, param_query)

    # Step 2: Insert the updated data
    db.to_sql(backtest_df, table_name, if_exists='append', index=False)

    return



def trade_export(response_json, balance, order_type="spot"):
    # Extract data from the JSON response
    if response_json.get("error"):
        print("Error in response:", response_json["error"])
        return

    result = response_json.get("result", {})
    txid_list = result.get("txid", [])  # List of transaction IDs
    order_description = result.get("descr", {}).get("order", "")
    
    # Parse the order description string
    if order_description:
        order_parts = order_description.split()
        trade_type = order_parts[0]  # "buy" or "sell"
        volume = float(order_parts[1])  # e.g., "1.45"
        symbol = order_parts[2]  # e.g., "XBTUSD"
        price = float(order_parts[-1])  # e.g., "27500.0"
    else:
        print("Order description is missing.")
        return

    txid = txid_list[0] if txid_list else "Unknown"
    time_date = datetime.now().strftime('%D %H:%M:%S')

    trade_data = {
        "order_type": trade_type,
        "volume": volume,
        "amount": price,
        "symbol": symbol,
        "date_time": time_date,
        "txid": txid,
        "trade_category": order_type  # Include "futures" or "spot"
    }
    trade_df = pd.DataFrame([trade_data])


    db = MSSQLDatabase('trades')
    table_name = 'trade_data'

    db.create_table_if_not_exists(table_name, trade_df)

    db.to_sql(trade_df, table_name, if_exists='append', index=False)

    print("Trade exported successfully.")

def export_optimization_results_to_db(study, strategy_class):
        """Export Bayesian optimization results to the database."""
        db = MSSQLDatabase('hyper_optuna')
        table_name = f"OptunaOptimization_{strategy_class.__name__}"

        # Create table if it doesn't exist
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                trial_id INTEGER PRIMARY KEY,
                params NVARCHAR(MAX),
                value REAL
            )
        """
        db.execute_sql(create_table_query)

        # Insert the results
        for trial in study.trials:
            params = str(trial.params)
            value = trial.value
            
            # Check if trial_id already exists
            check_query = f"SELECT COUNT(1) FROM {table_name} WHERE trial_id = @trial_id"
            result = db.read_sql_query(check_query, params={'trial_id': trial.number})
            exists = result.iloc[0, 0]
            
            if exists == 0:
                # If trial_id doesn't exist, insert it
                insert_query = f"""
                    INSERT INTO "{table_name}" (trial_id, params, value)
                    VALUES (@trial_id, @params, @value)
                """
                db.execute_sql(insert_query, params={'trial_id': trial.number, 'params': params, 'value': value})
        
def export_optimization_summary_to_db(results_df, strategy_name):
    """Export optimization summary results to the database."""
    db = MSSQLDatabase('hyper_optuna')
    table_name = f"OptimizationSummary_{strategy_name}"

    # Create table if it doesn't exist
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY IDENTITY(1,1),
            symbol NVARCHAR(255),
            granularity NVARCHAR(255),
            best_value REAL,
            best_params NVARCHAR(MAX),
            n_trials INTEGER,
            timestamp DATETIME2 DEFAULT GETDATE()
        )
    """
    try:
        db.execute_sql(create_table_query)
        # Insert the results
        for _, row in results_df.iterrows():
            insert_query = f"""
                INSERT INTO "{table_name}" (symbol, granularity, best_value, best_params, n_trials)
                VALUES (@symbol, @granularity, @best_value, @best_params, @n_trials)
            """
            params = {
                'symbol': row['symbol'],
                'granularity': row['granularity'],
                'best_value': row['best_value'],
                'best_params': str(row['best_params']),
                'n_trials': row['n_trials']
            }
            db.execute_sql(insert_query, params)
    except Exception as e:
        logger.error(f"Error exporting optimization summary to database: {e}")

from contextlib import contextmanager
import json
import threading
from threading import Lock
import pandas as pd
import os

# Use your existing database connection logic
@contextmanager
def get_db_connection(db_name):
    db = MSSQLDatabase(db_name)
    try:
        yield db.engine
    finally:
        pass

def initialize_backtest_tables(db):
    """Create tables to store backtest results if they don't exist"""
    try:
        # Table for overall backtest results
        create_table_results = """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY IDENTITY(1,1),
            symbol NVARCHAR(255),
            granularity NVARCHAR(255),
            start_date NVARCHAR(255),
            end_date NVARCHAR(255),
            initial_capital REAL,
            final_value REAL,
            total_return REAL,
            annualized_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            win_rate REAL,
            timestamp DATETIME2 DEFAULT GETDATE()
        )
        """
        db.execute_sql(create_table_results)
        
        # Table for model metrics
        create_table_metrics = """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY IDENTITY(1,1),
            backtest_id INTEGER,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
        )
        """
        db.execute_sql(create_table_metrics)

        # Table for feature importance
        create_table_importance = """
        CREATE TABLE IF NOT EXISTS feature_importance (
            id INTEGER PRIMARY KEY IDENTITY(1,1),
            backtest_id INTEGER,
            feature_name NVARCHAR(255),
            importance_value REAL,
            FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
        )
        """
        db.execute_sql(create_table_importance)
        
        # Table for individual trades
        create_table_trades = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY IDENTITY(1,1),
            backtest_id INTEGER,
            entry_date NVARCHAR(255),
            exit_date NVARCHAR(255),
            entry_price REAL,
            exit_price REAL,
            position_size REAL,
            profit_loss REAL,
            trade_return REAL,
            FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
        )
        """
        db.execute_sql(create_table_trades)
        
        # Table for WFO results
        create_table_wfo = """
        CREATE TABLE IF NOT EXISTS wfo_results (
            id INTEGER PRIMARY KEY IDENTITY(1,1),
            backtest_id INTEGER,
            window_number INTEGER,
            train_start NVARCHAR(255),
            train_end NVARCHAR(255),
            test_start NVARCHAR(255),
            test_end NVARCHAR(255),
            accuracy REAL,
            strategy_return REAL,
            buy_hold_return REAL,
            win_rate REAL,
            trades INTEGER,
            FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
        )
        """
        db.execute_sql(create_table_wfo)
    except Exception as e:
         logger.error(f"Error initializing tables in database: {e}")


def save_backtest_results(backtest_obj):
    """Save backtest results to database"""
    db = MSSQLDatabase('ai_backtest')
    try:
            # Initialize tables if they don't exist
            initialize_backtest_tables(db)
            
            # Extract backtest results
            results = backtest_obj.backtest_results
            
            # Insert backtest results
            insert_backtest_results = """
            INSERT INTO backtest_results 
            (symbol, granularity, start_date, end_date, initial_capital, 
            final_value, total_return, annualized_return, sharpe_ratio, 
            max_drawdown, total_trades, win_rate)
            VALUES (@symbol, @granularity, @start_date, @end_date, @initial_capital, 
            @final_value, @total_return, @annualized_return, @sharpe_ratio, 
            @max_drawdown, @total_trades, @win_rate)
            """

            params = {
                'symbol': backtest_obj.symbol,
                'granularity': backtest_obj.granularity,
                'start_date': str(backtest_obj.backtest_data.index[0]),
                'end_date': str(backtest_obj.backtest_data.index[-1]),
                'initial_capital': results['Initial Capital'],
                'final_value': results['Final Value'],
                'total_return': results['Total Return (%)'],
                'annualized_return': results['Annualized Return (%)'],
                'sharpe_ratio': results['Sharpe Ratio'],
                'max_drawdown': results['Max Drawdown (%)'],
                'total_trades': results['Total Trades'],
                'win_rate': results['Win Rate (%)']
            }
            db.execute_sql(insert_backtest_results, params)
            
            # Get the ID of the inserted backtest (SQL Server uses SCOPE_IDENTITY())
            result = db.read_sql_query("SELECT SCOPE_IDENTITY() AS id")
            backtest_id = int(result['id'][0])

            
            # Save feature importance
            for _, row in backtest_obj.feature_importance.iterrows():
                insert_feature_importance = """
                INSERT INTO feature_importance (backtest_id, feature_name, importance_value)
                VALUES (@backtest_id, @feature_name, @importance_value)
                """
                params = {
                    'backtest_id': backtest_id,
                    'feature_name': row['Feature'],
                    'importance_value': row['Importance']
                }
                db.execute_sql(insert_feature_importance, params)
            
            # Extract and save individual trades
            if hasattr(backtest_obj, 'backtest_data'):
                # Find buy and sell signals
                df = backtest_obj.backtest_data
                buy_signals = df[df['position'] > df['position'].shift(1)]
                sell_signals = df[df['position'] < df['position'].shift(1)]
                
                # Match buy and sell signals to create trade records
                if len(buy_signals) > 0 and len(sell_signals) > 0:
                    buy_indices = buy_signals.index.tolist()
                    sell_indices = sell_signals.index.tolist()
                    
                    trade_idx = 0
                    for i, buy_date in enumerate(buy_indices):
                        # Find the next sell after this buy
                        matching_sells = [s for s in sell_indices if s > buy_date]
                        if matching_sells:
                            sell_date = matching_sells[0]
                            
                            # Get trade details
                            entry_price = df.loc[buy_date, 'close']
                            exit_price = df.loc[sell_date, 'close']
                            position_size = df.loc[buy_date, 'holdings']
                            profit_loss = position_size * (exit_price/entry_price - 1)
                            trade_return = (exit_price / entry_price) - 1
                            
                            # Save trade
                            insert_trade = """
                            INSERT INTO trades
                            (backtest_id, entry_date, exit_date, entry_price, exit_price, 
                            position_size, profit_loss, trade_return)
                            VALUES (@backtest_id, @entry_date, @exit_date, @entry_price, @exit_price, 
                            @position_size, @profit_loss, @trade_return)
                            """
                            params = {
                                'backtest_id': backtest_id,
                                'entry_date': str(buy_date),
                                'exit_date': str(sell_date),
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'profit_loss': profit_loss,
                                'trade_return': trade_return
                            }
                            db.execute_sql(insert_trade, params)
            
            # Save WFO results if they exist
            if hasattr(backtest_obj, 'wfo_results') and backtest_obj.wfo_results is not None:
                for _, row in backtest_obj.wfo_results.iterrows():
                    insert_wfo_results = """
                    INSERT INTO wfo_results
                    (backtest_id, window_number, train_start, train_end, test_start, test_end,
                    accuracy, strategy_return, buy_hold_return, win_rate, trades)
                    VALUES (@backtest_id, @window_number, @train_start, @train_end, @test_start, @test_end,
                    @accuracy, @strategy_return, @buy_hold_return, @win_rate, @trades)
                    """
                    params = {
                        'backtest_id': backtest_id,
                        'window_number': row['Window'],
                        'train_start': str(row['Train_Start']),
                        'train_end': str(row['Train_End']),
                        'test_start': str(row['Test_Start']),
                        'test_end': str(row['Test_End']),
                        'accuracy': row['Accuracy'],
                        'strategy_return': row['Strategy_Return'],
                        'buy_hold_return': row['Buy_Hold_Return'],
                        'win_rate': row['Win_Rate'],
                        'trades': row['Trades']
                    }
                    db.execute_sql(insert_wfo_results, params)
            
            return backtest_id
            
    except Exception as e:
        logger.error(f"Error saving backtest results to database: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_backtest_results(symbol=None, granularity=None, limit=10):
    """Retrieve backtest results from database with optional filtering"""
    db = MSSQLDatabase('ai_backtest')
    try:
        query = "SELECT * FROM backtest_results"
        params = {}
        conditions = []
        
        if symbol:
            conditions.append("symbol = @symbol")
            params['symbol'] = symbol
            
        if granularity:
            conditions.append("granularity = @granularity")
            params['granularity'] = granularity
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY timestamp DESC OFFSET 0 ROWS FETCH NEXT @limit ROWS ONLY"
        params['limit'] = limit
        
        results = db.read_sql_query(query, params=params)
        return results
            
    except Exception as e:
        logger.error(f"Error retrieving backtest results: {e}")
        return pd.DataFrame()


#
#   DB API KEY FUNCTIONS
#   ---------------------
#   These functions are used to manage API keys in the database.
#   They include creating, updating, and deleting API keys for users.
#   The API keys are stored in a database table and can be retrieved as needed.
#   The functions also include error handling and logging for better debugging.


def setup_api_keys_table(db):
    """
    Create API keys table and audit table if they don't exist
    
    Args:
        db: MSSQLDatabase instance
    """
    try:
        # Create API keys table with encrypted storage
        db.execute_sql("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'api_keys')
            BEGIN
                CREATE TABLE api_keys (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    uuid NVARCHAR(50) UNIQUE NOT NULL,
                    email NVARCHAR(255) NOT NULL,
                    platform NVARCHAR(50) NOT NULL,
                    api_key NVARCHAR(MAX) NOT NULL,
                    api_secret NVARCHAR(MAX) NOT NULL,
                    passphrase NVARCHAR(MAX),
                    created_at NVARCHAR(50) NOT NULL,
                    updated_at NVARCHAR(50) NOT NULL,
                    last_used NVARCHAR(50),
                    is_active BIT DEFAULT 1,
                    metadata NVARCHAR(MAX),
                    last_verified NVARCHAR(50),
                    verification_status NVARCHAR(50),
                    key_hash NVARCHAR(255),
                    CONSTRAINT FK_api_keys_users FOREIGN KEY (email) REFERENCES users(email),
                    CONSTRAINT UQ_email_platform UNIQUE(email, platform)
                )
            END
        """)
        
        # Create API key audit log table
        db.execute_sql("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'api_key_audit_log')
            BEGIN
                CREATE TABLE api_key_audit_log (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    api_key_id NVARCHAR(50),
                    email NVARCHAR(255) NOT NULL,
                    platform NVARCHAR(50) NOT NULL,
                    action NVARCHAR(50) NOT NULL,
                    timestamp NVARCHAR(50) NOT NULL,
                    ip_address NVARCHAR(50),
                    user_agent NVARCHAR(MAX),
                    status NVARCHAR(50),
                    details NVARCHAR(MAX),
                    CONSTRAINT FK_audit_log_api_keys FOREIGN KEY (api_key_id) REFERENCES api_keys(uuid)
                )
            END
        """)
        
        logger.info("API keys and audit tables setup complete")
    except Exception as e:
        logger.error(f"Error setting up API keys table: {str(e)}")
        raise


def generate_user_secret(email, platform):
    """
    Generate a strong, user-specific secret for encryption
    
    Args:
        email: User's email
        platform: Platform name
    
    Returns:
        str: A user-specific secret
    """
    # Create a deterministic but secure secret based on user info
    # This ensures we can regenerate the same secret for decryption
    base_secret = f"{email}:{platform}:{os.environ.get('USER_SECRET_SALT', 'DEFAULT_SALT')}"
    return hashlib.sha256(base_secret.encode()).hexdigest()


def calculate_key_hash(api_key, platform):
    """
    Calculate a hash of the API key to identify changes without storing the key
    
    Args:
        api_key: The API key
        platform: The platform
    
    Returns:
        str: A one-way hash of the API key
    """
    key_data = f"{api_key}:{platform}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def log_api_key_action(
    db,
    email,
    platform,
    action,
    status,
    api_key_id=None,
    details=None,
    ip_address=None,
    user_agent=None
):
    """
    Log an API key related action for audit purposes
    
    Args:
        db: MSSQLDatabase instance
        email: User's email
        platform: Platform name
        action: Action performed (save, get, delete, etc.)
        status: Outcome status (success, failed, etc.)
        api_key_id: Optional UUID of the API key
        details: Optional additional details
        ip_address: Optional IP address of the requestor
        user_agent: Optional user agent of the requestor
    """
    try:
        timestamp = datetime.datetime.now().isoformat()
        
        db.execute_sql("""
            INSERT INTO api_key_audit_log
            (api_key_id, email, platform, action, timestamp, ip_address, user_agent, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            api_key_id,
            email,
            platform,
            action,
            timestamp,
            ip_address,
            user_agent,
            status,
            details
        ))
        
    except Exception as e:
        logger.error(f"Error logging API key action: {str(e)}")


def save_api_keys(
    db,
    email, 
    platform, 
    api_key, 
    api_secret, 
    passphrase=None, 
    metadata=None
):
    """
    Save encrypted API keys for a user's trading platform with verification
    
    Args:
        db: MSSQLDatabase instance
        email: User's email
        platform: Trading platform name (e.g., 'coinbase', 'kraken')
        api_key: API key to encrypt and store
        api_secret: API secret to encrypt and store
        passphrase: Optional passphrase for platforms that require it
        metadata: Optional JSON metadata about the API key
    
    Returns:
        bool: Success or failure
    """
    try:
        # Create tables if they don't exist
        setup_api_keys_table(db)
        
        # Verify API keys before saving
        verification = ApiKeyTester.test_api_connection(
            platform, 
            api_key, 
            api_secret, 
            passphrase
        )
        
        if verification["status"] != "success":
            logger.warning(f"API key verification failed for {email} on {platform}: {verification['message']}")
            log_api_key_action(
                db,
                email=email,
                platform=platform,
                action="save",
                status="failed",
                details=f"Verification failed: {verification['message']}",
                ip_address=metadata.get("ip_address") if metadata else None,
                user_agent=metadata.get("user_agent") if metadata else None
            )
            return False
        
        # Generate a user-specific secret
        user_secret = generate_user_secret(email, platform)
        
        # Calculate key hash for change detection
        key_hash = calculate_key_hash(api_key, platform)
        
        # Encrypt sensitive data
        encrypted_api_key = encrypt_api_key(api_key, user_secret)
        encrypted_api_secret = encrypt_api_key(api_secret, user_secret)
        encrypted_passphrase = None
        if passphrase:
            encrypted_passphrase = encrypt_api_key(passphrase, user_secret)
        
        # Current timestamp
        now = datetime.datetime.now().isoformat()
        
        # Begin transaction
        # Check if this user already has keys for this platform
        result = db.read_sql_query(
            "SELECT uuid, key_hash FROM api_keys WHERE email = ? AND platform = ?",
            (email, platform)
        )
        
        api_key_id = None
        
        if not result.empty:
            # Update existing keys
            existing = result.iloc[0]
            api_key_id = existing['uuid']
            
            # Only update if key has changed (based on hash)
            if existing['key_hash'] != key_hash:
                db.execute_sql("""
                    UPDATE api_keys
                    SET api_key = ?, api_secret = ?, passphrase = ?, 
                        updated_at = ?, is_active = 1, metadata = ?,
                        last_verified = ?, verification_status = ?, key_hash = ?
                    WHERE email = ? AND platform = ?
                """, (
                    encrypted_api_key, 
                    encrypted_api_secret, 
                    encrypted_passphrase,
                    now,
                    json.dumps(metadata) if metadata else None,
                    now,
                    "verified",
                    key_hash,
                    email, 
                    platform
                ))
                logger.info(f"Updated API keys for user {email} on platform {platform}")
            else:
                # Just update verification status
                db.execute_sql("""
                    UPDATE api_keys
                    SET last_verified = ?, verification_status = ?, is_active = 1
                    WHERE email = ? AND platform = ?
                """, (
                    now,
                    "verified",
                    email, 
                    platform
                ))
                logger.info(f"Verified existing API keys for user {email} on platform {platform}")
        else:
            # Create a new UUID for this API key
            api_key_id = str(uuid.uuid4())
            
            # Insert new keys
            db.execute_sql("""
                INSERT INTO api_keys 
                (uuid, email, platform, api_key, api_secret, passphrase, created_at, updated_at, 
                 metadata, is_active, last_verified, verification_status, key_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """, (
                api_key_id,
                email, 
                platform, 
                encrypted_api_key, 
                encrypted_api_secret, 
                encrypted_passphrase,
                now, 
                now,
                json.dumps(metadata) if metadata else None,
                now,
                "verified",
                key_hash
            ))
            logger.info(f"Inserted new API keys for user {email} on platform {platform}")
        
        # Update platform connection if exists
        result = db.read_sql_query(
            "SELECT id FROM platform_connections WHERE email = ?", 
            (email,)
        )
        
        if not result.empty:
            db.execute_sql(
                """UPDATE platform_connections 
                   SET platform = ?, timestamp = ?, api_key = ?, api_secret = ? 
                   WHERE email = ?""", 
                (platform, now, encrypted_api_key, encrypted_api_secret, email)
            )
        else:
            db.execute_sql(
                """INSERT INTO platform_connections (email, platform, timestamp, api_key, api_secret)
                   VALUES (?, ?, ?, ?, ?)""",
                (email, platform, now, encrypted_api_key, encrypted_api_secret)
            )
        
        # Update user setup status 
        db.execute_sql(
            """UPDATE users 
               SET platform_connected = ?, keys_configured = ? 
               WHERE email = ?""",
            (1, 1, email)
        )
        
        # Log successful action
        log_api_key_action(
            db,
            email=email,
            platform=platform,
            action="save",
            status="success",
            api_key_id=api_key_id,
            ip_address=metadata.get("ip_address") if metadata else None,
            user_agent=metadata.get("user_agent") if metadata else None
        )
        
        logger.info(f"API keys saved successfully for user {email} on platform {platform}")
        return True
    except Exception as e:
        logger.error(f"Error saving API keys for {email} on {platform}: {str(e)}")
        
        # Log failed action
        log_api_key_action(
            db,
            email=email,
            platform=platform,
            action="save",
            status="error",
            details=str(e),
            ip_address=metadata.get("ip_address") if metadata else None,
            user_agent=metadata.get("user_agent") if metadata else None
        )
        
        return False


def get_api_keys(db, email, platform):
    """
    Get decrypted API keys for a user's trading platform
    
    Args:
        db: MSSQLDatabase instance
        email: User's email
        platform: Trading platform name
    
    Returns:
        dict: Decrypted API credentials or None if not found
    """
    try:
        # Generate the user-specific secret
        user_secret = generate_user_secret(email, platform)
        
        result = db.read_sql_query(
            """SELECT uuid, api_key, api_secret, passphrase, is_active, verification_status 
               FROM api_keys 
               WHERE email = ? AND platform = ?""",
            (email, platform)
        )
        
        if result.empty or not result.iloc[0]['is_active']:
            logger.warning(f"No active API keys found for {email} on {platform}")
            return None
        
        row = result.iloc[0]
        api_key_id = row['uuid']
        encrypted_api_key = row['api_key']
        encrypted_api_secret = row['api_secret']
        encrypted_passphrase = row['passphrase']
        
        # Update last used timestamp
        now = datetime.datetime.now().isoformat()
        db.execute_sql(
            "UPDATE api_keys SET last_used = ? WHERE uuid = ?",
            (now, api_key_id)
        )
        
        # Log access
        log_api_key_action(
            db,
            email=email,
            platform=platform,
            action="access",
            status="success",
            api_key_id=api_key_id
        )
        
        # Decrypt the keys
        try:
            credentials = {
                "apiKey": decrypt_api_key(encrypted_api_key, user_secret),
                "apiSecret": decrypt_api_key(encrypted_api_secret, user_secret)
            }
            
            if encrypted_passphrase:
                credentials["passphrase"] = decrypt_api_key(encrypted_passphrase, user_secret)
            
            return credentials
        except Exception as e:
            logger.error(f"Error decrypting API keys for {email} on {platform}: {str(e)}")
            
            # Log decryption failure
            log_api_key_action(
                db,
                email=email,
                platform=platform,
                action="access",
                status="error",
                api_key_id=api_key_id,
                details=f"Decryption error: {str(e)}"
            )
            
            return None
    except Exception as e:
        logger.error(f"Error getting API keys for {email} on {platform}: {str(e)}")
        
        # Log general error
        log_api_key_action(
            db,
            email=email,
            platform=platform,
            action="access",
            status="error",
            details=str(e)
        )
        
        return None


def verify_api_keys(db, email, platform):
    """
    Verify that stored API keys for a platform are still valid
    
    Args:
        db: MSSQLDatabase instance
        email: User's email
        platform: Trading platform name
    
    Returns:
        dict: Verification result with status and message
    """
    try:
        # Get the decrypted API keys
        credentials = get_api_keys(db, email, platform)
        
        if not credentials:
            return {
                "status": "error",
                "message": f"No API keys found for {platform}"
            }
        
        # Test the API connection
        verification = ApiKeyTester.test_api_connection(
            platform, 
            credentials.get("apiKey"), 
            credentials.get("apiSecret"), 
            credentials.get("passphrase")
        )
        
        # Update verification status in database
        now = datetime.datetime.now().isoformat()
        
        db.execute_sql(
            """UPDATE api_keys 
               SET last_verified = ?, verification_status = ? 
               WHERE email = ? AND platform = ?""",
            (
                now,
                "verified" if verification["status"] == "success" else "failed",
                email,
                platform
            )
        )
        
        # Log verification result
        log_api_key_action(
            db,
            email=email,
            platform=platform,
            action="verify",
            status=verification["status"],
            details=verification.get("message", "")
        )
        
        return verification
    except Exception as e:
        logger.error(f"Error verifying API keys for {email} on {platform}: {str(e)}")
        return {
            "status": "error",
            "message": f"Error verifying API keys: {str(e)}"
        }


def delete_api_keys(db, email, platform, metadata=None):
    """
    Delete API keys for a user's trading platform
    
    Args:
        db: MSSQLDatabase instance
        email: User's email
        platform: Trading platform name
        metadata: Optional metadata about the request
    
    Returns:
        bool: Success or failure
    """
    try:
        # Get API key ID before deleting
        result = db.read_sql_query(
            "SELECT uuid FROM api_keys WHERE email = ? AND platform = ?",
            (email, platform)
        )
        api_key_id = result.iloc[0]['uuid'] if not result.empty else None
        
        # Mark as inactive instead of deleting
        db.execute_sql(
            """UPDATE api_keys 
               SET is_active = 0, updated_at = ? 
               WHERE email = ? AND platform = ?""",
            (datetime.datetime.now().isoformat(), email, platform)
        )
        
        # Also update platform connection if needed
        db.execute_sql(
            """UPDATE platform_connections 
               SET api_key = NULL, api_secret = NULL 
               WHERE email = ? AND platform = ?""", 
            (email, platform)
        )
        
        # Update user keys_configured status if this was their only platform
        result = db.read_sql_query(
            """SELECT COUNT(*) as count
               FROM api_keys 
               WHERE email = ? AND is_active = 1""",
            (email,)
        )
        remaining_keys = result.iloc[0]['count']
        
        if remaining_keys == 0:
            db.execute_sql(
                """UPDATE users 
                   SET keys_configured = 0 
                   WHERE email = ?""",
                (email,)
            )
        
        # Log successful deletion
        log_api_key_action(
            db,
            email=email,
            platform=platform,
            action="delete",
            status="success",
            api_key_id=api_key_id,
            ip_address=metadata.get("ip_address") if metadata else None,
            user_agent=metadata.get("user_agent") if metadata else None
        )
        
        logger.info(f"API keys deleted for user {email} on platform {platform}")
        return True
    except Exception as e:
        logger.error(f"Error deleting API keys for {email} on {platform}: {str(e)}")
        
        # Log failed deletion
        log_api_key_action(
            db,
            email=email,
            platform=platform,
            action="delete",
            status="error",
            details=str(e),
            ip_address=metadata.get("ip_address") if metadata else None,
            user_agent=metadata.get("user_agent") if metadata else None
        )
        
        return False


def get_api_key_info(db, email, platform=None):
    """
    Get information about stored API keys without revealing secrets
    
    Args:
        db: MSSQLDatabase instance
        email: User's email
        platform: Optional platform name (if None, get for all platforms)
    
    Returns:
        list: List of API key information dictionaries
    """
    try:
        if platform:
            result = db.read_sql_query(
                """SELECT uuid, platform, created_at, updated_at, last_used, 
                          last_verified, verification_status, metadata 
                   FROM api_keys 
                   WHERE email = ? AND platform = ? AND is_active = 1""",
                (email, platform)
            )
        else:
            result = db.read_sql_query(
                """SELECT uuid, platform, created_at, updated_at, last_used, 
                          last_verified, verification_status, metadata
                   FROM api_keys 
                   WHERE email = ? AND is_active = 1""",
                (email,)
            )
            
        key_info = []
        for _, row in result.iterrows():
            platform_name = row["platform"]
            
            info = {
                "id": row["uuid"],
                "platform": platform_name,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_used": row["last_used"],
                "last_verified": row["last_verified"],
                "verification_status": row["verification_status"],
                "has_passphrase": False,  # Will be updated below
            }
            
            # Check if this platform has a passphrase
            passphrase_result = db.read_sql_query(
                """SELECT passphrase 
                   FROM api_keys 
                   WHERE email = ? AND platform = ? AND is_active = 1""",
                (email, platform_name)
            )
            if not passphrase_result.empty and passphrase_result.iloc[0]['passphrase']:
                info["has_passphrase"] = True
                
            # Add any additional metadata
            if row["metadata"]:
                try:
                    meta_dict = json.loads(row["metadata"])
                    # Remove sensitive data from metadata
                    if 'ip_address' in meta_dict:
                        meta_dict['ip_address'] = meta_dict['ip_address'].split('.')
                        meta_dict['ip_address'][-1] = 'xxx'
                        meta_dict['ip_address'] = '.'.join(meta_dict['ip_address'])
                    
                    info["metadata"] = meta_dict
                except json.JSONDecodeError:
                    pass
            
            # Get recent activity
            activity_result = db.read_sql_query(
                """SELECT TOP 5 action, timestamp, status
                   FROM api_key_audit_log
                   WHERE api_key_id = ?
                   ORDER BY timestamp DESC""",
                (row["uuid"],)
            )
            
            if not activity_result.empty:
                recent_activity = activity_result.to_dict('records')
                info["recent_activity"] = recent_activity
                
            key_info.append(info)
            
        # Log access to this information
        log_api_key_action(
            db,
            email=email,
            platform=platform or "all",
            action="info",
            status="success"
        )
            
        return key_info
    except Exception as e:
        logger.error(f"Error getting API key info for {email}: {str(e)}")
        
        # Log error
        log_api_key_action(
            db,
            email=email,
            platform=platform or "all",
            action="info",
            status="error",
            details=str(e)
        )
        
        return []


def check_api_keys_exist(db, email, platform):
    """
    Check if a user has API keys for a specific platform
    
    Args:
        db: MSSQLDatabase instance
        email: User's email
        platform: Trading platform name
    
    Returns:
        bool: True if keys exist, False otherwise
    """
    try:
        result = db.read_sql_query(
            """SELECT COUNT(*) as count
               FROM api_keys 
               WHERE email = ? AND platform = ? AND is_active = 1""",
            (email, platform)
        )
        count = result.iloc[0]['count']
        
        return count > 0
    except Exception as e:
        logger.error(f"Error checking API keys for {email} on {platform}: {str(e)}")
        return False


def migrate_existing_keys(db):
    """
    Migrate existing keys to the new format with UUID and verification
    
    Args:
        db: MSSQLDatabase instance
    """
    try:
        # Check if any keys need migration (missing UUID)
        result = db.read_sql_query("SELECT COUNT(*) as count FROM api_keys WHERE uuid IS NULL")
        count = result.iloc[0]['count']
        
        if count == 0:
            logger.info("No API keys need migration")
            return
            
        logger.info(f"Found {count} API keys to migrate")
        
        # Get all keys needing migration
        keys_to_migrate = db.read_sql_query(
            """SELECT id, email, platform, api_key, api_secret 
               FROM api_keys 
               WHERE uuid IS NULL"""
        )
        
        for _, key in keys_to_migrate.iterrows():
            # Generate UUID
            key_uuid = str(uuid.uuid4())
            
            # Generate key hash
            # We can't calculate the actual key hash since we can't decrypt
            # Just use a placeholder
            key_hash = hashlib.sha256(f"{key['id']}:{key['platform']}".encode()).hexdigest()
            
            # Update the key
            db.execute_sql(
                """UPDATE api_keys
                   SET uuid = ?, key_hash = ?, verification_status = ?
                   WHERE id = ?""",
                (key_uuid, key_hash, "needs_verification", key['id'])
            )
            
            logger.info(f"Migrated API key {key['id']} for {key['email']} on {key['platform']}")
        
        logger.info(f"Successfully migrated {len(keys_to_migrate)} API keys")
    except Exception as e:
        logger.error(f"Error migrating API keys: {str(e)}")


def get_verification_needed_keys(db):
    """
    Get a list of API keys that need verification
    
    Args:
        db: MSSQLDatabase instance
    
    Returns:
        list: List of keys needing verification
    """
    try:
        # Get keys that have never been verified or haven't been verified in 7 days
        seven_days_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
        
        result = db.read_sql_query(
            """SELECT email, platform, uuid
               FROM api_keys
               WHERE is_active = 1 AND (
                   last_verified IS NULL OR
                   last_verified < ? OR
                   verification_status = 'needs_verification'
               )""",
            (seven_days_ago,)
        )
        
        return result.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting keys needing verification: {str(e)}")
        return []
    

#
#   DB USER FUNCTIONS
#   ---------------------
#   These functions are used to manage user accounts in the database.
#   They include creating, updating, and deleting user accounts.
#   The functions also include error handling and logging for better debugging.

def get_users():
    """Get all users and their passwords from the database."""
    db = MSSQLDatabase('users')
    try:
        query = "SELECT email, password FROM users;"
        users = db.read_sql_query(query)
        users_dict = dict(zip(users['email'], users['password']))
    except Exception as e:
        # Handle case where table might not exist
        if "Invalid object name" in str(e):
            users_dict = {}  # No users yet
        else:
            raise  # Reraise other errors
    return users_dict

def activate_2fa(email, totp_secret):
    """Activate 2FA for user."""
    db = MSSQLDatabase('users')
    try:
        # Create 2FA table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'totp_secrets')
            CREATE TABLE totp_secrets (
                email NVARCHAR(255) PRIMARY KEY,
                secret NVARCHAR(255),
                activated_at NVARCHAR(50)
            )"""
        )
        
        # Insert or update 2FA secret
        db.execute_sql(
            """MERGE INTO totp_secrets AS target
               USING (SELECT ? AS email) AS source
               ON target.email = source.email
               WHEN MATCHED THEN
                   UPDATE SET secret = ?, activated_at = ?
               WHEN NOT MATCHED THEN
                   INSERT (email, secret, activated_at)
                   VALUES (?, ?, ?);""",
            (email, totp_secret, datetime.now().isoformat(), email, totp_secret, datetime.now().isoformat())
        )
        
        # Update user 2FA status
        db.execute_sql(
            """UPDATE users SET twofa_enabled = 1 WHERE email = ?""",
            (email,)
        )
        
        # Delete the temporary secret
        db.execute_sql(
            """DELETE FROM totp_setup WHERE email = ?""",
            (email,)
        )
        
        logger.info(f"2FA activated for {email}")
    except Exception as e:
        logger.error(f"Error activating 2FA for {email}: {str(e)}")
        raise

def save_user(email, privacy_policy_accepted=False, api_key=None, 
              risk_percent=0.02, max_drawdown=0.15, max_open_trades=3):
    """Save a new user with default risk settings and setup state."""
    db = MSSQLDatabase('users')
    try:
        # Create users table if not exists
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'users')
            CREATE TABLE users (
                email NVARCHAR(255) PRIMARY KEY,
                password NVARCHAR(255),
                privacy_policy_accepted BIT,
                api_key NVARCHAR(255),
                risk_percent DECIMAL(10,6),
                max_drawdown DECIMAL(10,6),
                max_open_trades INT,
                platform_connected BIT,
                email_verified BIT,
                twofa_enabled BIT,
                setup_complete BIT,
                setup_completion_time NVARCHAR(50),
                created_at NVARCHAR(50)
            )"""
        )
        
        logger.info(f"(Creating new table USERS) : Executing query for user {email} with default risk settings")
        current_time = datetime.datetime.now().isoformat()
        
        # Insert new user with proper parameter binding
        params = (
            email, 
            '-', 
            1 if privacy_policy_accepted else 0, 
            api_key or '', 
            float(risk_percent), 
            float(max_drawdown), 
            int(max_open_trades),
            0, 0, 0, 0, '', 
            current_time
        )
        
        db.execute_sql(
            """INSERT INTO users 
               (email, password, privacy_policy_accepted, api_key, 
                risk_percent, max_drawdown, max_open_trades,
                platform_connected, email_verified, twofa_enabled, setup_complete,
                setup_completion_time, created_at) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""", 
            params
        )
        
        logger.info(f"User {email} registered with default risk settings")
    except Exception as e:
        logger.error(f"Error saving user {email}: {str(e)}")
        raise
    finally:
        db.close()

def save_user_platform(email, platform, timestamp):
    """Save user platform connection information."""
    db = MSSQLDatabase('users')
    try:
        # Create platform connections table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'platform_connections')
            CREATE TABLE platform_connections (
                email NVARCHAR(255) PRIMARY KEY,
                platform NVARCHAR(50),
                timestamp NVARCHAR(50),
                api_key NVARCHAR(255),
                api_secret NVARCHAR(255)
            )"""
        )
        
        # Check if a platform connection exists
        result = db.read_sql_query(
            "SELECT email FROM platform_connections WHERE email = ?", 
            (email,)
        )
        
        if not result.empty:
            # Update existing platform record
            db.execute_sql(
                """UPDATE platform_connections 
                   SET platform = ?, timestamp = ? 
                   WHERE email = ?""", 
                (platform, timestamp, email)
            )
        else:
            # Insert new platform record
            db.execute_sql(
                """INSERT INTO platform_connections (email, platform, timestamp)
                   VALUES (?, ?, ?)""",
                (email, platform, timestamp)
            )
        
        # Update user setup status
        db.execute_sql(
            """UPDATE users 
               SET platform_connected = 1 
               WHERE email = ?""",
            (email,)
        )
        
        logger.info(f"User {email} connected to platform {platform}")
    except Exception as e:
        logger.error(f"Error saving platform for {email}: {str(e)}")
        raise

def save_verification_code(email, verification_code, expires_at):
    """Save email verification code."""
    db = MSSQLDatabase('users')
    try:
        # Create verification codes table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'verification_codes')
            CREATE TABLE verification_codes (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email NVARCHAR(255),
                code NVARCHAR(50),
                expires_at NVARCHAR(50),
                created_at NVARCHAR(50),
                used BIT DEFAULT 0
            )"""
        )
        
        # Insert new verification code
        db.execute_sql(
            """INSERT INTO verification_codes (email, code, expires_at, created_at)
               VALUES (?, ?, ?, ?)""",
            (email, verification_code, expires_at.isoformat(), datetime.now().isoformat())
        )
        
        logger.info(f"Verification code created for {email}")
    except Exception as e:
        logger.error(f"Error saving verification code for {email}: {str(e)}")
        raise

def get_recent_verification_attempts(email, minutes=60):
    """Get count of recent verification attempts for rate limiting."""
    db = MSSQLDatabase('users')
    try:
        # Check if table exists
        result = db.read_sql_query(
            """SELECT COUNT(*) AS table_count FROM INFORMATION_SCHEMA.TABLES 
               WHERE TABLE_NAME = 'verification_codes'"""
        )
        
        if result.iloc[0]['table_count'] == 0:
            return 0
            
        # Get verification attempts in the last X minutes
        time_threshold = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        result = db.read_sql_query(
            """SELECT COUNT(*) AS attempt_count FROM verification_codes
               WHERE email = ? AND created_at > ?""",
            (email, time_threshold)
        )
        
        return result.iloc[0]['attempt_count']
    except Exception as e:
        logger.error(f"Error getting verification attempts for {email}: {str(e)}")
        return 0  # Return 0 on error to avoid blocking legitimate users

def check_verification_code(email, verification_code):
    """Check if verification code is valid and not expired."""
    db = MSSQLDatabase('users')
    try:
        # Check if table exists
        result = db.read_sql_query(
            """SELECT COUNT(*) AS table_count FROM INFORMATION_SCHEMA.TABLES 
               WHERE TABLE_NAME = 'verification_codes'"""
        )
        
        if result.iloc[0]['table_count'] == 0:
            return {
                "valid": False,
                "message": "No verification code found. Please request a new code."
            }
        
        # Get the most recent verification code for this email
        result = db.read_sql_query(
            """SELECT TOP 1 code, expires_at, used FROM verification_codes
               WHERE email = ? ORDER BY created_at DESC""",
            (email,)
        )
        
        if result.empty:
            return {
                "valid": False,
                "message": "No verification code found. Please request a new code."
            }
        
        db_code = result.iloc[0]['code']
        expires_at_str = result.iloc[0]['expires_at']
        used = result.iloc[0]['used']
        
        # Check if code is used
        if used:
            return {
                "valid": False,
                "message": "Verification code already used. Please request a new code."
            }
        
        # Check if code is expired
        expires_at = datetime.fromisoformat(expires_at_str)
        if expires_at < datetime.now():
            return {
                "valid": False,
                "message": "Verification code expired. Please request a new code."
            }
        
        # Check if code matches
        if verification_code != db_code:
            return {
                "valid": False,
                "message": "Invalid verification code. Please try again."
            }
        
        # Mark code as used
        db.execute_sql(
            """UPDATE verification_codes SET used = 1 WHERE email = ? AND code = ?""",
            (email, verification_code)
        )
        
        return {
            "valid": True,
            "message": "Verification successful."
        }
    except Exception as e:
        logger.error(f"Error checking verification code for {email}: {str(e)}")
        return {
            "valid": False,
            "message": "An error occurred while verifying the code."
        }

def log_failed_verification(email, verification_code):
    """Log failed verification attempt for security monitoring."""
    db = MSSQLDatabase('users')
    try:
        # Create failed verifications table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'failed_verifications')
            CREATE TABLE failed_verifications (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email NVARCHAR(255),
                code NVARCHAR(50),
                attempt_time NVARCHAR(50),
                ip_address NVARCHAR(50)
            )"""
        )
        
        # Insert failed verification record
        db.execute_sql(
            """INSERT INTO failed_verifications (email, code, attempt_time)
               VALUES (?, ?, ?)""",
            (email, verification_code, datetime.now().isoformat())
        )
        
        logger.warning(f"Failed verification attempt for {email}")
    except Exception as e:
        logger.error(f"Error logging failed verification for {email}: {str(e)}")

def count_recent_failed_verifications(email, minutes=30):
    """Count recent failed verification attempts."""
    db = MSSQLDatabase('users')
    try:
        # Check if table exists
        result = db.read_sql_query(
            """SELECT COUNT(*) AS table_count FROM INFORMATION_SCHEMA.TABLES 
               WHERE TABLE_NAME = 'failed_verifications'"""
        )
        
        if result.iloc[0]['table_count'] == 0:
            return 0
            
        # Get failed verification attempts in the last X minutes
        time_threshold = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        result = db.read_sql_query(
            """SELECT COUNT(*) AS failed_count FROM failed_verifications
               WHERE email = ? AND attempt_time > ?""",
            (email, time_threshold)
        )
        
        return result.iloc[0]['failed_count']
    except Exception as e:
        logger.error(f"Error counting failed verifications for {email}: {str(e)}")
        return 0

def mark_email_verified(email):
    """Mark user's email as verified."""
    db = MSSQLDatabase('users')
    try:
        # Update user email verification status
        db.execute_sql(
            """UPDATE users SET email_verified = 1 WHERE email = ?""",
            (email,)
        )
        
        logger.info(f"Email verified for user {email}")
    except Exception as e:
        logger.error(f"Error marking email as verified for {email}: {str(e)}")
        raise

def save_temp_2fa_secret(email, totp_secret):
    """Save temporary 2FA secret during setup."""
    db = MSSQLDatabase('users')
    try:
        # Create 2FA setup table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'totp_setup')
            CREATE TABLE totp_setup (
                email NVARCHAR(255) PRIMARY KEY,
                secret NVARCHAR(255),
                created_at NVARCHAR(50)
            )"""
        )
        
        # Check if there's an existing entry
        result = db.read_sql_query(
            """SELECT email FROM totp_setup WHERE email = ?""",
            (email,)
        )
        
        if not result.empty:
            # Update existing entry
            db.execute_sql(
                """UPDATE totp_setup 
                   SET secret = ?, created_at = ? 
                   WHERE email = ?""",
                (totp_secret, datetime.now().isoformat(), email)
            )
        else:
            # Insert new entry
            db.execute_sql(
                """INSERT INTO totp_setup (email, secret, created_at)
                   VALUES (?, ?, ?)""",
                (email, totp_secret, datetime.now().isoformat())
            )
        
        logger.info(f"Temporary 2FA secret saved for {email}")
    except Exception as e:
        logger.error(f"Error saving temporary 2FA secret for {email}: {str(e)}")
        raise

def get_temp_2fa_secret(email):
    """Get temporary 2FA secret."""
    db = MSSQLDatabase('users')
    try:
        # Get the 2FA secret
        result = db.read_sql_query(
            """SELECT secret FROM totp_setup WHERE email = ?""",
            (email,)
        )
        
        if not result.empty:
            return result.iloc[0]['secret']
        else:
            return None
    except Exception as e:
        logger.error(f"Error getting temporary 2FA secret for {email}: {str(e)}")
        return None

def log_failed_2fa_attempt(email):
    """Log failed 2FA verification attempt."""
    db = MSSQLDatabase('users')
    try:
        # Create failed 2FA attempts table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'failed_2fa_attempts')
            CREATE TABLE failed_2fa_attempts (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email NVARCHAR(255),
                attempt_time NVARCHAR(50),
                ip_address NVARCHAR(50)
            )"""
        )
        
        # Insert failed attempt record
        db.execute_sql(
            """INSERT INTO failed_2fa_attempts (email, attempt_time)
               VALUES (?, ?)""",
            (email, datetime.now().isoformat())
        )
        
        logger.warning(f"Failed 2FA attempt for {email}")
    except Exception as e:
        logger.error(f"Error logging failed 2FA attempt for {email}: {str(e)}")

def count_recent_failed_2fa_attempts(email, minutes=30):
    """Count recent failed 2FA attempts."""
    db = MSSQLDatabase('users')
    try:
        # Check if table exists
        result = db.read_sql_query(
            """SELECT COUNT(*) AS table_count FROM INFORMATION_SCHEMA.TABLES 
               WHERE TABLE_NAME = 'failed_2fa_attempts'"""
        )
        
        if result.iloc[0]['table_count'] == 0:
            return 0
            
        # Get failed 2FA attempts in the last X minutes
        time_threshold = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        result = db.read_sql_query(
            """SELECT COUNT(*) AS failed_count FROM failed_2fa_attempts
               WHERE email = ? AND attempt_time > ?""",
            (email, time_threshold)
        )
        
        return result.iloc[0]['failed_count']
    except Exception as e:
        logger.error(f"Error counting failed 2FA attempts for {email}: {str(e)}")
        return 0

def save_backup_codes(email, backup_codes):
    """Save backup codes for 2FA recovery."""
    db = MSSQLDatabase('users')
    try:
        # Create backup codes table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'backup_codes')
            CREATE TABLE backup_codes (
                email NVARCHAR(255) PRIMARY KEY,
                codes NVARCHAR(MAX),
                created_at NVARCHAR(50)
            )"""
        )
        
        # Convert codes list to JSON string
        codes_json = json.dumps(backup_codes)
        
        # Insert or update backup codes
        db.execute_sql(
            """MERGE INTO backup_codes AS target
               USING (SELECT ? AS email) AS source
               ON target.email = source.email
               WHEN MATCHED THEN
                   UPDATE SET codes = ?, created_at = ?
               WHEN NOT MATCHED THEN
                   INSERT (email, codes, created_at)
                   VALUES (?, ?, ?);""",
            (email, codes_json, datetime.now().isoformat(), email, codes_json, datetime.now().isoformat())
        )
        
        logger.info(f"Backup codes saved for {email}")
    except Exception as e:
        logger.error(f"Error saving backup codes for {email}: {str(e)}")
        raise

def get_user_setup_status(email):
    """Get user's setup status."""
    db = MSSQLDatabase('users')
    try:
        # Get user setup status
        result = db.read_sql_query(
            """SELECT platform_connected, email_verified, twofa_enabled, setup_complete
               FROM users WHERE email = ?""",
            (email,)
        )
        
        if result.empty:
            return {
                "platform_connected": False,
                "email_verified": False,
                "2fa_enabled": False,
                "setup_complete": False
            }
        
        # Convert bit/boolean values to Python booleans
        platform_connected = bool(result.iloc[0]['platform_connected'])
        email_verified = bool(result.iloc[0]['email_verified'])
        twofa_enabled = bool(result.iloc[0]['twofa_enabled'])
        setup_complete = bool(result.iloc[0]['setup_complete'])
        
        return {
            "platform_connected": platform_connected,
            "email_verified": email_verified,
            "2fa_enabled": twofa_enabled,
            "setup_complete": setup_complete
        }
    except Exception as e:
        logger.error(f"Error getting setup status for {email}: {str(e)}")
        return {
            "platform_connected": False,
            "email_verified": False,
            "2fa_enabled": False,
            "setup_complete": False
        }

def mark_setup_complete(email, completion_time):
    """Mark user setup as complete."""
    db = MSSQLDatabase('users')
    try:
        # Update user setup status
        db.execute_sql(
            """UPDATE users 
               SET setup_complete = 1, setup_completion_time = ? 
               WHERE email = ?""",
            (completion_time.isoformat(), email)
        )
        
        logger.info(f"Setup marked complete for {email}")
    except Exception as e:
        logger.error(f"Error marking setup complete for {email}: {str(e)}")
        raise

def get_risk_settings(email):
    """Get user's risk settings with enhanced error handling."""
    logger.info(f"=== get_risk_settings called for email: {email} ===")
    
    if not email:
        logger.error("Email parameter is required")
        return None
    
    db = None
    try:
        db = MSSQLDatabase('users')
        logger.info("Database connection established")
    except Exception as db_init_error:
        logger.error(f"Failed to initialize database connection: {str(db_init_error)}")
        return None
        
    try:
        # First, ensure the user exists
        logger.info(f"Checking if user exists: {email}")
        user_check = db.read_sql_query(
            "SELECT COUNT(*) AS user_count FROM users WHERE email = ?", 
            params=(email,)
        )
        
        if user_check.empty:
            logger.error("User check query returned empty result")
            return None
            
        user_count = user_check.iloc[0]['user_count']
        logger.info(f"User count for {email}: {user_count}")
        
        if user_count == 0:
            logger.warning(f"User not found: {email}")
            return None
        
        # Check if the table has risk columns
        logger.info("Checking table schema for risk columns")
        try:
            columns_result = db.read_sql_query(
                """SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
                   WHERE TABLE_NAME = 'users' AND TABLE_SCHEMA = 'dbo'"""
            )
        except Exception as schema_error:
            logger.error(f"Error checking table schema: {str(schema_error)}")
            try:
                columns_result = db.read_sql_query(
                    "SELECT name FROM sys.columns WHERE object_id = OBJECT_ID('users')"
                )
                columns_result = columns_result.rename(columns={'name': 'COLUMN_NAME'})
            except Exception as alt_schema_error:
                logger.error(f"Alternative schema check also failed: {str(alt_schema_error)}")
                return None
        
        if columns_result.empty:
            logger.error("Could not retrieve table schema information")
            return None
            
        columns = [col.lower() for col in columns_result['COLUMN_NAME'].tolist()]
        logger.info(f"Available columns: {columns}")
        
        # Define risk columns with their data types
        risk_columns = {
            'risk_percent': 'DECIMAL(10,6)',
            'max_drawdown': 'DECIMAL(10,6)', 
            'max_open_trades': 'INT'
        }
        
        # Add missing columns
        missing_columns = []
        for col_name, col_type in risk_columns.items():
            if col_name.lower() not in columns:
                missing_columns.append((col_name, col_type))
        
        if missing_columns:
            logger.info(f"Adding missing risk columns: {[col[0] for col in missing_columns]}")
            for col_name, col_type in missing_columns:
                try:
                    alter_sql = f"ALTER TABLE users ADD {col_name} {col_type}"
                    logger.info(f"Executing: {alter_sql}")
                    db.execute_sql(alter_sql)
                    logger.info(f"Successfully added column {col_name}")
                except Exception as alter_e:
                    logger.error(f"Failed to add column {col_name}: {str(alter_e)}")
            
            # Set default values for the user
            try:
                update_sql = """UPDATE users 
                               SET risk_percent = COALESCE(risk_percent, 0.02),
                                   max_drawdown = COALESCE(max_drawdown, 0.15),
                                   max_open_trades = COALESCE(max_open_trades, 3)
                               WHERE email = ?"""
                logger.info(f"Setting default values for user: {email}")
                db.execute_sql(update_sql, params=(email,))
                logger.info("Default values set successfully")
            except Exception as default_error:
                logger.error(f"Error setting default values: {str(default_error)}")
        
        # Query the risk settings
        logger.info(f"Querying risk settings for user: {email}")
        try:
            result = db.read_sql_query(
                """SELECT 
                    COALESCE(risk_percent, 0.02) as risk_percent,
                    COALESCE(max_drawdown, 0.15) as max_drawdown,
                    COALESCE(max_open_trades, 3) as max_open_trades
                   FROM users WHERE email = ?""", 
                params=(email,)
            )
        except Exception as query_error:
            logger.error(f"Error in main query: {str(query_error)}")
            try:
                logger.info("Attempting fallback query")
                result = db.read_sql_query(
                    "SELECT * FROM users WHERE email = ?", 
                    params=(email,)
                )
                if not result.empty:
                    risk_data = {}
                    for col in ['risk_percent', 'max_drawdown', 'max_open_trades']:
                        if col in result.columns:
                            risk_data[col] = result.iloc[0][col]
                        else:
                            defaults = {'risk_percent': 0.02, 'max_drawdown': 0.15, 'max_open_trades': 3}
                            risk_data[col] = defaults[col]
                    
                    import pandas as pd
                    result = pd.DataFrame([risk_data])
                else:
                    logger.error("Fallback query also returned empty result")
                    return None
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {str(fallback_error)}")
                return None
        
        if result.empty:
            logger.warning(f"No risk settings found for user: {email}")
            return None
        
        logger.info(f"Raw query result: {result.to_dict()}")
        
        # Convert to appropriate types with validation
        try:
            risk_percent = float(result.iloc[0]['risk_percent']) if result.iloc[0]['risk_percent'] is not None else 0.02
            max_drawdown = float(result.iloc[0]['max_drawdown']) if result.iloc[0]['max_drawdown'] is not None else 0.15
            max_open_trades = int(result.iloc[0]['max_open_trades']) if result.iloc[0]['max_open_trades'] is not None else 3
            
            logger.info(f"Raw values - risk_percent: {risk_percent}, max_drawdown: {max_drawdown}, max_open_trades: {max_open_trades}")
            
            # Validate ranges and apply bounds
            risk_percent = max(0.001, min(1.0, risk_percent))
            max_drawdown = max(0.01, min(1.0, max_drawdown))
            max_open_trades = max(1, min(50, max_open_trades))
            
            result_dict = {
                "risk_percent": risk_percent,
                "max_drawdown": max_drawdown,
                "max_open_trades": max_open_trades
            }
            
            logger.info(f"Returning validated risk settings: {result_dict}")
            return result_dict
            
        except (ValueError, TypeError) as conv_e:
            logger.error(f"Error converting risk settings data types for {email}: {str(conv_e)}")
            logger.info("Returning default values due to conversion error")
            return {
                "risk_percent": 0.02,
                "max_drawdown": 0.15,
                "max_open_trades": 3
            }
            
    except Exception as e:
        logger.error(f"Unexpected error getting risk settings for {email}: {str(e)}")
        logger.exception("Full error traceback:")
        return None
    finally:
        if db:
            try:
                db.close()
                logger.info("Database connection closed")
            except Exception as close_error:
                logger.error(f"Error closing database connection: {str(close_error)}")


def update_risk_settings(email, risk_percent, max_drawdown, max_open_trades):
    """Update user's risk settings."""
    logger.info(f"=== update_risk_settings called ===")
    logger.info(f"Parameters: email={email}, risk_percent={risk_percent}, max_drawdown={max_drawdown}, max_open_trades={max_open_trades}")
    
    if not email:
        logger.error("Email parameter is required")
        return False
        
    # Validate input parameters
    try:
        risk_percent = float(risk_percent)
        max_drawdown = float(max_drawdown)
        max_open_trades = int(max_open_trades)
        
        logger.info(f"Converted parameters: risk_percent={risk_percent}, max_drawdown={max_drawdown}, max_open_trades={max_open_trades}")
        
        # Validate ranges
        if not (0.001 <= risk_percent <= 1.0):
            logger.error(f"Invalid risk_percent value: {risk_percent}")
            return False
        if not (0.01 <= max_drawdown <= 1.0):
            logger.error(f"Invalid max_drawdown value: {max_drawdown}")
            return False
        if not (1 <= max_open_trades <= 50):
            logger.error(f"Invalid max_open_trades value: {max_open_trades}")
            return False
            
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid parameter types for risk settings: {str(e)}")
        return False
    
    db = None
    try:
        db = MSSQLDatabase('users')
        
        # First, ensure the user exists
        logger.info(f"Checking if user exists: {email}")
        user_check = db.read_sql_query(
            "SELECT COUNT(*) AS user_count FROM users WHERE email = ?", 
            params=(email,)
        )
        
        if user_check.empty or user_check.iloc[0]['user_count'] == 0:
            logger.error(f"Cannot update risk settings: User not found: {email}")
            return False
        
        logger.info(f"User exists, proceeding with update")
        
        # Check if the table has risk columns
        try:
            columns_result = db.read_sql_query(
                """SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
                   WHERE TABLE_NAME = 'users' AND TABLE_SCHEMA = 'dbo'"""
            )
        except Exception as schema_error:
            logger.warning(f"INFORMATION_SCHEMA query failed, trying alternative: {str(schema_error)}")
            try:
                columns_result = db.read_sql_query(
                    "SELECT name as COLUMN_NAME FROM sys.columns WHERE object_id = OBJECT_ID('users')"
                )
            except Exception as alt_error:
                logger.error(f"Could not get table schema: {str(alt_error)}")
                return False
        
        if columns_result.empty:
            logger.error("Could not retrieve table schema information")
            return False
            
        columns = [col.lower() for col in columns_result['COLUMN_NAME'].tolist()]
        logger.info(f"Available columns: {columns}")
        
        # Define risk columns with their data types
        risk_columns = {
            'risk_percent': 'DECIMAL(10,6)',
            'max_drawdown': 'DECIMAL(10,6)', 
            'max_open_trades': 'INT'
        }
        
        # Add missing columns
        missing_columns = []
        for col_name, col_type in risk_columns.items():
            if col_name.lower() not in columns:
                missing_columns.append((col_name, col_type))
        
        if missing_columns:
            logger.info(f"Adding missing risk columns: {[col[0] for col in missing_columns]}")
            for col_name, col_type in missing_columns:
                try:
                    alter_sql = f"ALTER TABLE users ADD {col_name} {col_type}"
                    logger.info(f"Executing: {alter_sql}")
                    db.execute_sql(alter_sql)
                    logger.info(f"Successfully added column: {col_name}")
                except Exception as alter_e:
                    logger.error(f"Failed to add column {col_name}: {str(alter_e)}")
                    return False
        
        # Execute the update
        update_sql = """UPDATE users 
                       SET risk_percent = ?, max_drawdown = ?, max_open_trades = ? 
                       WHERE email = ?"""
        
        params = (float(risk_percent), float(max_drawdown), int(max_open_trades), str(email))
        logger.info(f"Update SQL: {update_sql}")
        logger.info(f"Parameters: {params}")
        logger.info(f"Parameter types: {[type(p).__name__ for p in params]}")
        
        try:
            logger.info("Executing update query...")
            update_result = db.execute_sql(update_sql, params=params)
            logger.info(f"Update executed successfully, affected rows: {update_result.rowcount}")
        except Exception as update_error:
            logger.error(f"Update execution failed: {str(update_error)}")
            return False
        
        # Verify the update was successful
        logger.info("Verifying update...")
        try:
            verification = db.read_sql_query(
                """SELECT risk_percent, max_drawdown, max_open_trades 
                   FROM users WHERE email = ?""", 
                params=(email,)
            )
            
            if not verification.empty:
                updated_risk = float(verification.iloc[0]['risk_percent'])
                updated_drawdown = float(verification.iloc[0]['max_drawdown'])
                updated_trades = int(verification.iloc[0]['max_open_trades'])
                
                logger.info(f"Verification - DB values: risk={updated_risk}, drawdown={updated_drawdown}, trades={updated_trades}")
                logger.info(f"Verification - Expected: risk={risk_percent}, drawdown={max_drawdown}, trades={max_open_trades}")
                
                if (abs(updated_risk - risk_percent) < 0.000001 and 
                    abs(updated_drawdown - max_drawdown) < 0.000001 and 
                    updated_trades == max_open_trades):
                    
                    logger.info(f"Successfully updated and verified risk settings for user {email}")
                    return True
                else:
                    logger.error(f"Risk settings update verification failed for user {email}")
                    return False
            else:
                logger.error(f"Could not verify risk settings update for user {email} - no data returned")
                return False
                
        except Exception as verify_error:
            logger.error(f"Verification query failed: {str(verify_error)}")
            logger.warning("Returning True despite verification failure - update may have succeeded")
            return True
            
    except Exception as e:
        logger.error(f"Unexpected error updating risk settings for {email}: {str(e)}")
        logger.exception("Full error traceback:")
        return False
    finally:
        if db:
            try:
                db.close()
                logger.info("Database connection closed")
            except Exception as close_error:
                logger.error(f"Error closing database connection: {str(close_error)}")

def get_backtest_history(email):
    """Get user's backtest history."""
    db = MSSQLDatabase('backtest')
    try:
        # Check if backtests table exists
        result = db.read_sql_query(
            """SELECT COUNT(*) AS table_count FROM INFORMATION_SCHEMA.TABLES 
               WHERE TABLE_NAME = 'backtests'"""
        )
        
        if result.iloc[0]['table_count'] == 0:
            logger.warning("backtests table does not exist")
            return []
        
        # Use string formatting directly since it's working
        # Proper SQL escaping for email
        escaped_email = email.replace("'", "''")
        history = db.read_sql_query(f"SELECT * FROM backtests WHERE email = '{escaped_email}'")
        
        if history is None or history.empty:
            logger.info(f"No backtest history found for {email}")
            return []
        
        result_list = []
        for _, row in history.iterrows():
            result_list.append({
                'email': str(row['email']) if pd.notna(row['email']) else None,
                'symbol': str(row['symbol']) if pd.notna(row['symbol']) else None,
                'strategy': str(row['strategy']) if pd.notna(row['strategy']) else None,
                'result': str(row['result']) if pd.notna(row['result']) else None,
                'date': str(row['date']) if pd.notna(row['date']) else None
            })
        
        logger.info(f"Found {len(result_list)} backtest records for {email}")
        return result_list
        
    except Exception as e:
        logger.error(f"Error getting backtest history for {email}: {str(e)}")
        return []
    finally:
        try:
            db.close()
        except:
            pass

def save_password_reset_token(email, reset_token, expires_at):
    """Save a password reset token for a user."""
    db = MSSQLDatabase('users')
    try:
        # Create password reset tokens table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'password_reset_tokens')
            CREATE TABLE password_reset_tokens (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email NVARCHAR(255),
                token NVARCHAR(255),
                expires_at NVARCHAR(50),
                created_at NVARCHAR(50),
                used BIT DEFAULT 0
            )"""
        )
        
        # Insert the new token
        db.execute_sql(
            """INSERT INTO password_reset_tokens (email, token, expires_at, created_at, used)
               VALUES (?, ?, ?, ?, ?)""",
            (email, reset_token, expires_at.isoformat(), datetime.now().isoformat(), 0)
        )
        
        logger.info(f"Password reset token created for {email}")
    except Exception as e:
        logger.error(f"Error saving password reset token for {email}: {str(e)}")
        raise

def get_recent_password_reset_requests(email, minutes=60):
    """Count recent password reset requests for rate limiting."""
    db = MSSQLDatabase('users')
    try:
        # Check if table exists
        result = db.read_sql_query(
            """SELECT COUNT(*) AS table_count FROM INFORMATION_SCHEMA.TABLES 
               WHERE TABLE_NAME = 'password_reset_tokens'"""
        )
        
        if result.iloc[0]['table_count'] == 0:
            return 0
        
        # Calculate the cutoff time
        cutoff_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        # Count recent requests
        result = db.read_sql_query(
            """SELECT COUNT(*) AS request_count FROM password_reset_tokens 
               WHERE email = ? AND created_at > ?""",
            (email, cutoff_time)
        )
        
        return result.iloc[0]['request_count']
    except Exception as e:
        logger.error(f"Error counting recent password reset requests for {email}: {str(e)}")
        return 0

def validate_password_reset_token(email, token):
    """Validate if a password reset token is valid and not expired."""
    db = MSSQLDatabase('users')
    try:
        # Check if table exists
        result = db.read_sql_query(
            """SELECT COUNT(*) AS table_count FROM INFORMATION_SCHEMA.TABLES 
               WHERE TABLE_NAME = 'password_reset_tokens'"""
        )
        
        if result.iloc[0]['table_count'] == 0:
            return False
        
        # Get the token
        result = db.read_sql_query(
            """SELECT TOP 1 expires_at, used FROM password_reset_tokens 
               WHERE email = ? AND token = ?
               ORDER BY created_at DESC""",
            (email, token)
        )
        
        # Token not found
        if result.empty:
            return False
        
        expires_at = result.iloc[0]['expires_at']
        used = result.iloc[0]['used']
        
        # Check if token is used
        if used:
            return False
        
        # Check if token is expired
        expires_at_dt = datetime.fromisoformat(expires_at)
        if datetime.now() > expires_at_dt:
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating password reset token for {email}: {str(e)}")
        return False

def invalidate_password_reset_token(email, token):
    """Mark a password reset token as used to prevent reuse."""
    db = MSSQLDatabase('users')
    try:
        # Mark the token as used
        db.execute_sql(
            """UPDATE password_reset_tokens SET used = 1
               WHERE email = ? AND token = ?""",
            (email, token)
        )
        
        logger.info(f"Password reset token invalidated for {email}")
    except Exception as e:
        logger.error(f"Error invalidating password reset token for {email}: {str(e)}")

def log_failed_password_reset(email, token):
    """Log failed password reset attempt for security monitoring."""

    db = MSSQLDatabase('users')

    try:
        # Create failed attempts table if it doesn't exist
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'failed_password_resets')
            CREATE TABLE failed_password_resets (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email NVARCHAR(255),
                token NVARCHAR(255),
                attempt_time NVARCHAR(50),
                ip_address NVARCHAR(50)
            )"""
        )
        
        # Log the failed attempt
        db.execute_sql(
            """INSERT INTO failed_password_resets (email, token, attempt_time, ip_address)
               VALUES (?, ?, ?, ?)""",
            (email, token, datetime.now().isoformat(), "N/A")
        )
        
        logger.warning(f"Failed password reset attempt logged for {email}")
    except Exception as e:
        logger.error(f"Error logging failed password reset for {email}: {str(e)}")

def update_user_password(email, hashed_password):
    """Update a user's password."""
    
    db = MSSQLDatabase('users')

    try:
        # Update the password
        db.execute_sql(
            """UPDATE users SET password = ? WHERE email = ?""",
            (hashed_password, email)
        )
        
        # Check if update was successful
        result = db.read_sql_query(
            "SELECT COUNT(*) AS row_count FROM users WHERE email = ?",
            (email,)
        )
        
        if result.iloc[0]['row_count'] == 0:
            logger.error(f"User {email} not found when trying to update password")
            raise Exception(f"User {email} not found")
        
        # Log the password change
        db.execute_sql(
            """IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'password_changes')
            CREATE TABLE password_changes (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email NVARCHAR(255),
                changed_at NVARCHAR(50)
            )"""
        )
        
        db.execute_sql(
            """INSERT INTO password_changes (email, changed_at)
               VALUES (?, ?)""",
            (email, datetime.now().isoformat())
        )
        
        logger.info(f"Password updated for {email}")
    except Exception as e:
        logger.error(f"Error updating password for {email}: {str(e)}")
        raise

def invalidate_password_reset_token(email, token):
    """
    Mark a password reset token as used to prevent reuse
    
    Args:
        email: User email
        token: Token to invalidate
    """
    try:
        # Create database connection using the MSSQLDatabase class
        db = MSSQLDatabase('users')
        
        # SQL query to mark the token as used
        sql_query = """
            UPDATE password_reset_tokens 
            SET used = 1
            WHERE email = :email AND token = :token
        """
        
        # Execute the update query with parameters
        params = {"email": email, "token": token}
        db.execute_sql(sql_query, params)
        
        logger.info(f"Password reset token invalidated for {email}")
    except Exception as e:
        logger.error(f"Error invalidating password reset token for {email}: {str(e)}")
    finally:
        # No need to explicitly close connection as it's handled by the context manager in execute_sql
        pass
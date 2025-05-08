import pandas as pd
import core.utils as utils
import inspect
import numpy as np
import sys
import time
import gc
import sys
from datetime import datetime
from threading import Lock
import os
import logging
import json
import pandas as pd
import logging

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
        self.db = db_file_name # Retaining the original logic of string interpolation
        self.driver = "ODBC+Driver+18+for+SQL+Server"
        self.engine = self.get_engine() # Initialize engine in the constructor
        self.lock = Lock() # Thread lock for database access

    def get_engine(self):
        url = (
            f"mssql+pyodbc://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}?driver={self.driver}"
            "&TrustServerCertificate=yes"
        )
        return create_engine(url, pool_pre_ping=True, fast_executemany=True)

    def execute_sql(self, sql_text, params=None):
         with self.lock, self.engine.begin() as conn:
             if params:
                 return conn.execute(text(sql_text), params)
             else:
                 return conn.execute(text(sql_text))
    def read_sql_query(self, sql_text, params=None):
        """Execute a SQL query and return the result as a pandas DataFrame."""
        with self.lock, self.engine.begin() as conn:
            if params:
                return pd.read_sql_query(text(sql_text), conn, params=params)
            else:
                return pd.read_sql_query(text(sql_text), conn)

    def to_sql(self, df, table_name, if_exists='append', index=False):
        """Write records stored in a DataFrame to a SQL database."""
        with self.lock, self.engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=index, method='multi', chunksize=20000)
            
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

    # Ensure database path exists
    if not os.path.exists(db_path):
        logger.error(f"Database path {db_path} does not exist!")
        return {}
        
    # Ensure the database file exists
    db_file_path = f'{db_path}/{granularity}'
    if not os.path.exists(db_file_path):
        logger.error(f"Database file {db_file_path} does not exist!")
        return {}

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
# def _create_table_if_not_exists(table_name, df, conn):
#     """ Helper function to create table if it doesn't exist """
#     try:
#         # Check if the table exists
#         print(f"Checking if table {table_name} exists...")
#         table_exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
#         table_exists = pd.read_sql(table_exists_query, conn)
#         print("Table existence check completed.")
        
#         if table_exists.empty:
#             # Create table if it does not exist
#             print(f"Table {table_name} doesn't exist. Creating table...")
#             columns = df.columns
#             dtypes = df.dtypes  
#             sql_dtypes = []
#             for col in columns:
#                 dtype = dtypes[col]
#                 if pd.api.types.is_integer_dtype(dtype):
#                     sql_dtype = 'INTEGER'
#                 elif pd.api.types.is_float_dtype(dtype):
#                     sql_dtype = 'REAL'
#                 else:
#                     sql_dtype = 'TEXT'
#                 sql_dtypes.append(f'"{col}" {sql_dtype}')
            
#             create_table_query = f"CREATE TABLE {table_name} ("
#             create_table_query += ', '.join(sql_dtypes)
#             create_table_query += ");"
#             print(f"Creating table with query:\n{create_table_query}")
            
#             cursor = conn.cursor()
#             cursor.execute(create_table_query)
#             conn.commit()
#             print(f"Table {table_name} created successfully.")
#     except Exception as e:
#         print(f"Error occurred while creating table {table_name}: {e}")

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
def get_users():
    db = MSSQLDatabase('users')
    query = "SELECT email, password FROM users;"
    users = db.read_sql_query(query)
    users_dict = dict(zip(users['email'], users['password']))
    return users_dict
def save_user(email, password, privacy_policy_accepted=False, api_key=None):
    db = MSSQLDatabase('users')
    df = pd.DataFrame(columns=['email', 'password', 'privacy_policy_accepted', 'api_key'])
    db.create_table_if_not_exists('users',df)
    
    data = {'email': email, 'password': password, 'privacy_policy_accepted': privacy_policy_accepted, 'api_key': api_key}
    df = pd.DataFrame([data])
    db.to_sql(df, 'users', if_exists='append', index=False)


#save_user("test_user", "test_password")
def get_backtest_history(email):
    db = MSSQLDatabase('backtests')
    df = pd.DataFrame(columns=['email', 'symbol', 'strategy', 'result', 'date'])
    db.create_table_if_not_exists('backtests',df)
    query = f"SELECT * FROM backtests WHERE email = ?"
    history = db.read_sql_query(query, params={'email': email})
    return history.to_dict(orient="records")

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
    
    db = MSSQLDatabase('backtests')
    df = pd.DataFrame(columns=['email', 'symbol', 'strategy', 'result', 'date'])
    db.create_table_if_not_exists('backtests',df)

    data = {'email': email, 'symbol': symbol, 'strategy': strategy, 'result': result_json, 'date': date}
    df = pd.DataFrame([data])
    db.to_sql(df, 'backtests', if_exists='append', index=False)


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

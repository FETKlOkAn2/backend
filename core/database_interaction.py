import pandas as pd
import core.utils as utils
import sqlite3 as sql
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

db_path = os.getenv('DATABASE_PATH')
print("DATABASE_PATH : ", db_path)
def get_historical_from_db(granularity, symbols: list = [], num_days: int = None, convert=False):
    original_symbol = symbols

    if convert:
        symbols = utils.convert_symbols(lone_symbol=symbols)
    db_lock = Lock()

    def get_connection():
        with db_lock:
            return sql.connect(f'{db_path}/{granularity}.db')    
    conn = get_connection()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    tables_data = {}
    
    for table in tables['name']:
        clean_table_name = '-'.join(table.split('_')[:2])

        # If symbols are provided, skip tables that are not in the symbol list
        if symbols and clean_table_name not in symbols:
            continue

        # Retrieve data from the table
        data = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data.set_index('date', inplace=True)


        if num_days is not None:
            last_date = data.index.max()  # Find the most recent date in the dataset
            start_date = last_date - pd.Timedelta(days=num_days)
            data = data.loc[data.index >= start_date]

        # Store the data in the dictionary
        if convert:
            tables_data[original_symbol] = data
        else:
            tables_data[clean_table_name] = data

    
    conn.close()
    return tables_data


def get_best_params(strategy_object, df_manager=None, live_trading=False, best_of_all_granularities=False, minimum_trades=None, with_lowest_losing_average=False):
    granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY']
    
    try:
        conn = sql.connect(f'{db_path}/hyper.db') #or test_hyper.db
        #print('Connected to the database successfully.')
    except Exception as e:
        #print('Failed to connect to the database:', e)
        return None

    #print('Getting best params...')
    #print(f'DATABASE_PATH (best params): {db_path}/hyper.db')

    try:
        if best_of_all_granularities:
            best_results = []
            best_granularity = ''
            best_return = float('-inf')  # To track the best return
            
            for granularity in granularities:
                try:
                    #print(f'Processing granularity: {granularity}')
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
                    #print(f"Executing query: {query}")

                    result = pd.read_sql_query(query, conn)
                    #print(f"Query result for {granularity}:", result)

                    if result.empty or all(result.iloc[0].isnull()):
                        #print(f"No valid results for granularity: {granularity}")
                        continue

                    max_return = result['max_return'].iloc[0]
                    list_results = [
                        result[param].iloc[0] if param in result.columns else None for param in param_keys
                    ]

                    #print(f"Results for {granularity}: max_return={max_return}, parameters={list_results}")

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
                #print(f"Executing query: {query}")

                result = pd.read_sql_query(query, conn)
                #print(f"Query result:", result)

                list_results = [
                    result[param].iloc[0] for param in param_keys if param in result.columns
                ]
                print(f"Final results for single granularity: {list_results}")
            except Exception as e:
                print('Error querying for specific granularity:', e)
                return None

    finally:
        try:
            conn.close()
            #print('Database connection closed successfully.')
        except Exception as e:
            print('Failed to close the database connection:', e)

    return best_results if best_of_all_granularities else list_results


def export_optimization_results(df):
    try:
        conn = sql.connect(f'{db_path}/optimization.db')
        print("Connected to database successfully.")
        _create_table_if_not_exists('optimization_results', df, conn)
        
        # Check for unsupported types
        print("Verifying DataFrame types:")
        print(df.dtypes)
        
        print("Exporting results to the database...")
        df.to_sql('optimization_results', conn, if_exists='append', index=False)
        print("Data exported successfully.")
    except Exception as e:
        print(f"Error occurred while exporting optimization results: {e}")
    finally:
        conn.close()
        print("Database connection closed.")

def _create_table_if_not_exists(table_name, df, conn):
    """ Helper function to create table if it doesn't exist """
    try:
        # Check if the table exists
        print(f"Checking if table {table_name} exists...")
        table_exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        table_exists = pd.read_sql(table_exists_query, conn)
        print("Table existence check completed.")
        
        if table_exists.empty:
            # Create table if it does not exist
            print(f"Table {table_name} doesn't exist. Creating table...")
            columns = df.columns
            dtypes = df.dtypes  
            sql_dtypes = []
            for col in columns:
                dtype = dtypes[col]
                if pd.api.types.is_integer_dtype(dtype):
                    sql_dtype = 'INTEGER'
                elif pd.api.types.is_float_dtype(dtype):
                    sql_dtype = 'REAL'
                else:
                    sql_dtype = 'TEXT'
                sql_dtypes.append(f'"{col}" {sql_dtype}')
            
            create_table_query = f"CREATE TABLE {table_name} ("
            create_table_query += ', '.join(sql_dtypes)
            create_table_query += ");"
            print(f"Creating table with query:\n{create_table_query}")
            
            cursor = conn.cursor()
            cursor.execute(create_table_query)
            conn.commit()
            print(f"Table {table_name} created successfully.")
    except Exception as e:
        print(f"Error occurred while creating table {table_name}: {e}")

def get_users():
    conn = sql.connect(f'{db_path}/users.db')
    query = "SELECT email, password FROM users;"
    users = pd.read_sql_query(query, conn)
    conn.close()
    users_dict = dict(zip(users['email'], users['password']))
    return users_dict
def save_user(email, password):
    _create_table_if_not_exists('users', pd.DataFrame(columns=['email', 'password']), sql.connect(f'{db_path}/users.db'))
    conn = sql.connect(f'{db_path}/users.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (email, password) VALUES (?, ?);", (email, password))
    conn.commit()
    conn.close()
def get_backtest_history(email):
    _create_table_if_not_exists('backtests', pd.DataFrame(columns=['email', 'symbol', 'strategy', 'result', 'date']), sql.connect(f'{db_path}/backtests.db'))
    conn = sql.connect(f'{db_path}/backtests.db')
    query = f"SELECT * FROM backtests WHERE email = ?;"
    history = pd.read_sql_query(query, conn, params=(email,))
    conn.close()
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

    _create_table_if_not_exists(
        'backtests',
        pd.DataFrame(columns=['email', 'symbol', 'strategy', 'result', 'date']),
        sql.connect(f'{db_path}/backtests.db')
    )

    conn = sql.connect(f'{db_path}/backtests.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO backtests (email, symbol, strategy, result, date)
        VALUES (?, ?, ?, ?, ?);
    """, (email, symbol, strategy, result_json, date))
    conn.commit()
    conn.close()


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
    db_lock = Lock()
    def get_connection():
        with db_lock:
            return sql.connect(f'{db_path}/hyper.db') #test_hyper

    conn = get_connection()

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
    _create_table_if_not_exists(table_name, combined_df, conn=conn)

    # Check for existing data and update
    query = f'SELECT * FROM "{table_name}" WHERE symbol = ?;'
    delete_query = f'DELETE FROM "{table_name}" WHERE symbol = ?;'
    existing_data = pd.read_sql(query, conn, params=(symbol,))

    if not existing_data.empty:
        with conn:
            conn.execute(delete_query, (symbol,))
        combined_df.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        combined_df.to_sql(table_name, conn, if_exists='append', index=False)

    conn.close()
    return

def export_historical_to_db(dict_df, granularity):
    conn = sql.connect(f'{db_path}/{granularity}.db')
    cursor = conn.cursor()
    
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
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ESCAPE '\\'", (pattern,))
        existing_tables = cursor.fetchall()
        
        # Drop all existing tables that match the pattern
        for existing_table in existing_tables:
            cursor.execute(f'DROP TABLE IF EXISTS \"{existing_table[0]}\"')
            #print(f"Dropped table: {existing_table[0]}")
        
        # Write the DataFrame to the new table
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
        df.to_sql(new_table_name, conn, if_exists='replace', index=True)
        #print(f"\nCreated table: {new_table_name}")
    
    conn.commit()
    conn.close()


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
    conn = sql.connect(f'{db_path}/backtest.db')

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
        _create_table_if_not_exists(table_name, backtest_df, conn)

        # Prepare the DELETE query
        delete_query = f'DELETE FROM "{table_name}" WHERE symbol = ?'
        param_query = (symbol,)

    else:
        # Handle DataFrame directly
        backtest_df = object
        granularity = "default_granularity"  # Fallback granularity if not provided
        table_name = f"{multiple_table_name}_{granularity}"

        # Ensure the table exists
        _create_table_if_not_exists(table_name, backtest_df, conn)

        # Prepare the DELETE query
        symbol = backtest_df['symbol'].unique()[0]
        delete_query = f'DELETE FROM "{table_name}" WHERE symbol = ?'
        param_query = (symbol,)

    # Step 1: Delete existing rows with the same symbol
    cursor = conn.cursor()
    cursor.execute(delete_query, param_query)
    conn.commit()

    # Step 2: Insert the updated data
    backtest_df.to_sql(table_name, conn, if_exists='append', index=False)

    conn.close()
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

    db_path = f'core/database/trades.db'
    table_name = 'trade_data'

    conn = sql.connect(db_path)
    _create_table_if_not_exists(table_name, trade_df, conn)

    trade_df.to_sql(table_name, conn, if_exists='append', index=False)

    conn.commit()
    conn.close()

    print("Trade exported successfully.")

def export_optimization_results_to_db(study, strategy_class):
        """Export Bayesian optimization results to the database."""
        conn = sql.connect(f'{db_path}/hyper_optuna.db')
        table_name = f"OptunaOptimization_{strategy_class.__name__}"

        # Create table if it doesn't exist
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                trial_id INTEGER PRIMARY KEY,
                params TEXT,
                value REAL
            )
        """)

        # Insert the results
        for trial in study.trials:
            params = str(trial.params)
            value = trial.value
            
            # Check if trial_id already exists
            cursor = conn.execute(f"SELECT COUNT(1) FROM {table_name} WHERE trial_id = ?", (trial.number,))
            exists = cursor.fetchone()[0]
            
            if exists == 0:
                # If trial_id doesn't exist, insert it
                conn.execute(f"""
                    INSERT INTO "{table_name}" (trial_id, params, value)
                    VALUES (?, ?, ?)
                """, (trial.number, params, value))

        conn.commit()
        conn.close()    

# def export_optimization_results(df):
#     try:
#         conn = sql.connect(f'{db_path}/ai_optimization.db')
#         print("Connected to database successfully.")
#         _create_table_if_not_exists('ai_optimization_results', df, conn)
        
#         # Check for unsupported types
#         print("Verifying DataFrame types:")
#         print(df.dtypes)
        
#         print("Exporting results to the database...")
#         df.to_sql('optimization_results', conn, if_exists='append', index=False)
#         print("Data exported successfully.")
#     except Exception as e:
#         print(f"Error occurred while exporting optimization results: {e}")
#     finally:
#         conn.close()
#         print("Database connection closed.")

from contextlib import contextmanager
import json
import threading
from threading import Lock
import sqlite3 as sql
import pandas as pd
import os

# Use your existing database connection logic
@contextmanager
def get_db_connection(db_name):
    db_lock = Lock()
    with db_lock:
        conn = sql.connect(f'{db_path}/{db_name}.db')
        try:
            yield conn
        finally:
            conn.close()

def initialize_backtest_tables(conn):
    """Create tables to store backtest results if they don't exist"""
    cursor = conn.cursor()
    
    # Table for overall backtest results
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        granularity TEXT,
        start_date TEXT,
        end_date TEXT,
        initial_capital REAL,
        final_value REAL,
        total_return REAL,
        annualized_return REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        total_trades INTEGER,
        win_rate REAL,
        timestamp TEXT
    )
    ''')
    
    # Table for model metrics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id INTEGER,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
    )
    ''')
    
    # Table for feature importance
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feature_importance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id INTEGER,
        feature_name TEXT,
        importance_value REAL,
        FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
    )
    ''')
    
    # Table for individual trades
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id INTEGER,
        entry_date TEXT,
        exit_date TEXT,
        entry_price REAL,
        exit_price REAL,
        position_size REAL,
        profit_loss REAL,
        trade_return REAL,
        FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
    )
    ''')
    
    # Table for WFO results
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS wfo_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id INTEGER,
        window_number INTEGER,
        train_start TEXT,
        train_end TEXT,
        test_start TEXT,
        test_end TEXT,
        accuracy REAL,
        strategy_return REAL,
        buy_hold_return REAL,
        win_rate REAL,
        trades INTEGER,
        FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
    )
    ''')
    
    conn.commit()

def save_backtest_results(backtest_obj):
    """Save backtest results to database"""
    try:
        with get_db_connection('ai_backtest') as conn:
            # Initialize tables if they don't exist
            initialize_backtest_tables(conn)
            
            # Extract backtest results
            results = backtest_obj.backtest_results
            
            # Insert backtest results
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO backtest_results 
            (symbol, granularity, start_date, end_date, initial_capital, 
            final_value, total_return, annualized_return, sharpe_ratio, 
            max_drawdown, total_trades, win_rate, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                backtest_obj.symbol,
                backtest_obj.granularity,
                str(backtest_obj.backtest_data.index[0]),
                str(backtest_obj.backtest_data.index[-1]),
                results['Initial Capital'],
                results['Final Value'],
                results['Total Return (%)'],
                results['Annualized Return (%)'],
                results['Sharpe Ratio'],
                results['Max Drawdown (%)'],
                results['Total Trades'],
                results['Win Rate (%)']
            ))
            
            # Get the ID of the inserted backtest
            backtest_id = cursor.lastrowid
            
            # Save feature importance
            for _, row in backtest_obj.feature_importance.iterrows():
                cursor.execute('''
                INSERT INTO feature_importance (backtest_id, feature_name, importance_value)
                VALUES (?, ?, ?)
                ''', (
                    backtest_id,
                    row['Feature'],
                    row['Importance']
                ))
            
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
                            cursor.execute('''
                            INSERT INTO trades
                            (backtest_id, entry_date, exit_date, entry_price, exit_price, 
                            position_size, profit_loss, trade_return)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                backtest_id,
                                str(buy_date),
                                str(sell_date),
                                entry_price,
                                exit_price,
                                position_size,
                                profit_loss,
                                trade_return
                            ))
            
            # Save WFO results if they exist
            if hasattr(backtest_obj, 'wfo_results') and backtest_obj.wfo_results is not None:
                for _, row in backtest_obj.wfo_results.iterrows():
                    cursor.execute('''
                    INSERT INTO wfo_results
                    (backtest_id, window_number, train_start, train_end, test_start, test_end,
                    accuracy, strategy_return, buy_hold_return, win_rate, trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        backtest_id,
                        row['Window'],
                        str(row['Train_Start']),
                        str(row['Train_End']),
                        str(row['Test_Start']),
                        str(row['Test_End']),
                        row['Accuracy'],
                        row['Strategy_Return'],
                        row['Buy_Hold_Return'],
                        row['Win_Rate'],
                        row['Trades']
                    ))
            
            conn.commit()
            return backtest_id
            
    except Exception as e:
        logger.error(f"Error saving backtest results to database: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_backtest_results(symbol=None, granularity=None, limit=10):
    """Retrieve backtest results from database with optional filtering"""
    try:
        with get_db_connection('ai_backtest') as conn:
            query = "SELECT * FROM backtest_results"
            params = []
            
            if symbol or granularity:
                query += " WHERE"
                
                if symbol:
                    query += " symbol = ?"
                    params.append(symbol)
                    
                if symbol and granularity:
                    query += " AND"
                    
                if granularity:
                    query += " granularity = ?"
                    params.append(granularity)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            results = pd.read_sql_query(query, conn, params=params)
            return results
            
    except Exception as e:
        logger.error(f"Error retrieving backtest results: {e}")
        return pd.DataFrame()
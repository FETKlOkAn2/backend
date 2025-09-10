import os
import logging
import time
from sqlalchemy import create_engine, text
import pandas as pd
from threading import Lock

logger = logging.getLogger(__name__)

class FlexibleDatabase:
    def __init__(self, db_file_name, db_type="sqlite"):
        self.db_type = db_type.lower()
        self.db_file_name = db_file_name
        self.engine = self.get_engine()
        self.lock = Lock()
    
    def get_engine(self):
        if self.db_type == "sqlite":
            # For local SQLite database in E:/database directory
            db_directory = "E:/database"
            # Ensure the directory exists
            os.makedirs(db_directory, exist_ok=True)
            # Construct full path
            full_path = os.path.join(db_directory, self.db_file_name)
            db_path = f"sqlite:///{full_path}"
            return create_engine(db_path, pool_pre_ping=True)
        
        elif self.db_type == "mssql":
            # Your existing MSSQL logic
            self.user = os.getenv("DB_USER")
            self.password = os.getenv("DB_PASSWORD")
            self.host = os.getenv("DB_HOST")
            self.port = os.getenv("DB_PORT", "1433")
            self.driver = "ODBC+Driver+18+for+SQL+Server"
            
            connection_string = (
                f"mssql+pyodbc://{self.user}:{self.password}@{self.host}:{self.port}/"
                f"{self.db_file_name}?driver={self.driver}&TrustServerCertificate=yes"
            )
            return create_engine(connection_string, pool_pre_ping=True)
        
        elif self.db_type == "postgresql":
            # PostgreSQL option
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{self.db_file_name}"
            return create_engine(connection_string, pool_pre_ping=True)
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _convert_positional_to_named_params(self, sql_text, params):
        """Convert positional parameters (?) to named parameters (:param_0, :param_1, etc.)"""
        if not isinstance(params, (list, tuple)):
            return sql_text, params
            
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
        
        return modified_sql, param_dict

    def read_sql_query(self, query, params=None):
        """Execute a SELECT query and return results as DataFrame"""
        with self.lock:
            try:
                if params:
                    if isinstance(params, dict):
                        # For named parameters
                        return pd.read_sql_query(text(query), self.engine, params=params)
                    elif isinstance(params, (list, tuple)):
                        # Convert positional parameters to named parameters
                        modified_query, param_dict = self._convert_positional_to_named_params(query, params)
                        return pd.read_sql_query(text(modified_query), self.engine, params=param_dict)
                else:
                    return pd.read_sql_query(text(query), self.engine)
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                print(f"Error executing query: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error
    
    def execute_sql(self, query, params=None):
        """Execute a SQL command (INSERT, UPDATE, DELETE, DROP, etc.)"""
        with self.lock:
            try:
                with self.engine.begin() as conn:
                    if params:
                        if isinstance(params, dict):
                            # For named parameters
                            result = conn.execute(text(query), params)
                        elif isinstance(params, (list, tuple)):
                            # Convert positional parameters to named parameters
                            modified_query, param_dict = self._convert_positional_to_named_params(query, params)
                            result = conn.execute(text(modified_query), param_dict)
                    else:
                        result = conn.execute(text(query))
                    return result
            except Exception as e:
                logger.error(f"Error executing SQL: {e}")
                print(f"Error executing SQL: {e}")
                raise
    
    def execute_many(self, query, params_list):
        """Execute the same SQL statement multiple times with different parameters"""
        with self.lock:
            try:
                with self.engine.begin() as conn:
                    if params_list and len(params_list) > 0:
                        first_params = params_list[0]
                        if isinstance(first_params, (list, tuple)):
                            # Convert the query once for all parameter sets
                            modified_query, _ = self._convert_positional_to_named_params(query, first_params)
                            
                            # Convert all parameter sets to dictionaries
                            param_dicts = []
                            for params in params_list:
                                param_dict = {f'param_{i}': param for i, param in enumerate(params)}
                                param_dicts.append(param_dict)
                            
                            # Execute with all parameter sets
                            result = conn.execute(text(modified_query), param_dicts)
                        else:
                            # Assume dictionary parameters
                            result = conn.execute(text(query), params_list)
                    else:
                        result = conn.execute(text(query))
                    return result
            except Exception as e:
                logger.error(f"Error executing SQL with multiple parameters: {e}")
                print(f"Error executing SQL with multiple parameters: {e}")
                raise
    
    def to_sql(self, dataframe, table_name, if_exists='replace', index=True, chunksize=1000):
        """Write DataFrame to SQL table"""
        with self.lock:
            try:
                dataframe.to_sql(
                    table_name, 
                    self.engine, 
                    if_exists=if_exists, 
                    index=index,
                    chunksize=chunksize,  # Process in chunks to avoid memory issues
                    method='multi'  # For better performance with large datasets
                )
            except Exception as e:
                logger.error(f"Error writing to SQL: {e}")
                print(f"Error writing to SQL: {e}")
                raise
    
    def get_table_names(self):
        """Get list of all table names in the database"""
        if self.db_type == "sqlite":
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        else:
            query = "SELECT TABLE_NAME as name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        
        result = self.read_sql_query(query)
        return result['name'].tolist() if not result.empty else []
    
    def table_exists(self, table_name, schema_name=None):
        """Check if a table exists in the database"""
        if self.db_type == "sqlite":
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
            params = [table_name]
        elif self.db_type == "mssql":
            if schema_name:
                query = "SELECT CASE WHEN OBJECT_ID(?, 'U') IS NOT NULL THEN 1 ELSE 0 END as table_exists"
                params = [f"[{schema_name}].[{table_name}]"]
            else:
                query = "SELECT CASE WHEN OBJECT_ID(?, 'U') IS NOT NULL THEN 1 ELSE 0 END as table_exists"
                params = [table_name]
        else:  # PostgreSQL and others
            if schema_name:
                query = "SELECT tablename FROM pg_tables WHERE schemaname = ? AND tablename = ?"
                params = [schema_name, table_name]
            else:
                query = "SELECT tablename FROM pg_tables WHERE tablename = ?"
                params = [table_name]
        
        result = self.read_sql_query(query, params)
        
        if self.db_type == "mssql" and 'table_exists' in result.columns:
            return not result.empty and result.iloc[0]['table_exists'] == 1
        else:
            return not result.empty
    
    def create_table_if_not_exists(self, table_name, df, schema=None):
        """Creates a table if it doesn't exist based on the DataFrame's schema."""
        try:
            # Check if the table exists
            if self.table_exists(table_name, schema):
                logger.info(f"Table {table_name} already exists.")
                return

            # Build the CREATE TABLE query based on DataFrame's schema
            columns = df.columns
            dtypes = df.dtypes
            sql_dtypes = []
            
            for col in columns:
                dtype = dtypes[col]
                if self.db_type == "sqlite":
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_dtype = 'INTEGER'
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_dtype = 'REAL'
                    elif dtype == 'datetime64[ns]':
                        sql_dtype = 'TIMESTAMP'
                    else:
                        sql_dtype = 'TEXT'
                    sql_dtypes.append(f'"{col}" {sql_dtype}')
                    
                elif self.db_type == "mssql":
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_dtype = 'INT'
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_dtype = 'FLOAT'
                    elif dtype == 'datetime64[ns]':
                        sql_dtype = 'DATETIME2'
                    else:
                        sql_dtype = 'NVARCHAR(MAX)'
                    sql_dtypes.append(f'[{col}] {sql_dtype}')
                    
                else:  # PostgreSQL
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_dtype = 'INTEGER'
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_dtype = 'DOUBLE PRECISION'
                    elif dtype == 'datetime64[ns]':
                        sql_dtype = 'TIMESTAMP'
                    else:
                        sql_dtype = 'TEXT'
                    sql_dtypes.append(f'"{col}" {sql_dtype}')

            # Create the table
            if self.db_type == "sqlite":
                create_table_query = f'CREATE TABLE "{table_name}" ('
            elif self.db_type == "mssql":
                table_ref = f"[{schema}].[{table_name}]" if schema else f"[{table_name}]"
                create_table_query = f"CREATE TABLE {table_ref} ("
            else:  # PostgreSQL
                table_ref = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
                create_table_query = f"CREATE TABLE {table_ref} ("
                
            create_table_query += ', '.join(sql_dtypes)
            create_table_query += ");"

            self.execute_sql(create_table_query)
            logger.info(f"Table {table_name} created successfully.")
            
        except Exception as e:
            logger.error(f"Error occurred while creating table {table_name}: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()


def get_historical_from_db(granularity, symbols: list = [], num_days: int = None, convert=False, db_type="sqlite"):
    """
    Retrieve historical data from database with improved error handling and performance.
    
    Args:
        granularity (str): The granularity/database name
        symbols (list): List of symbols to retrieve (empty list means all)
        num_days (int): Number of days to retrieve from the most recent date
        convert (bool): Whether to convert symbols using utils.convert_symbols
        db_type (str): Database type ('sqlite', 'mssql', 'postgresql')
    
    Returns:
        dict: Dictionary with symbol names as keys and DataFrames as values
    """
    original_symbol = symbols

    if convert:
        try:
            import utils
            symbols = utils.convert_symbols(lone_symbol=symbols)
        except ImportError:
            logger.warning("utils module not found, skipping symbol conversion")
        except Exception as e:
            logger.warning(f"Error converting symbols: {e}")

    # Database file naming based on type
    if db_type == "sqlite":
        db_file = f'{granularity}.db'
    else:
        db_file = granularity
        
    db = FlexibleDatabase(db_file, db_type)

    try:
        # Get all tables in the database
        tables = db.get_table_names()
        tables_data = {}
        
        logger.info(f"Found {len(tables)} tables in {granularity}")
        
        if not tables:
            logger.warning(f"No tables found in {granularity}")
            return {}
        
        # Filter tables based on symbols if provided
        tables_to_process = []
        for table in tables:
            # Extract symbol from table name (assuming format like 'BTC-USD_1m' or 'BTC_USD_1m')
            clean_table_name = '-'.join(table.split('_')[:2])
            if not symbols or clean_table_name in symbols:
                tables_to_process.append((table, clean_table_name))
        
        if not tables_to_process:
            logger.warning(f"No matching tables found for symbols: {symbols}")
            return {}
        
        logger.info(f"Processing {len(tables_to_process)} tables")
        
        for table, clean_table_name in tables_to_process:
            try:
                # Retrieve data from the table - use appropriate quoting for each DB type
                if db.db_type == "sqlite":
                    query_data = f'SELECT * FROM "{table}" ORDER BY date DESC'
                elif db.db_type == "mssql":
                    query_data = f'SELECT * FROM [{table}] ORDER BY date DESC'
                else:  # PostgreSQL
                    query_data = f'SELECT * FROM "{table}" ORDER BY date DESC'
                
                # Add LIMIT if num_days is specified and we can calculate approximate rows needed
                if num_days is not None and granularity:
                    # Rough estimation of rows per day based on granularity
                    rows_per_day_map = {
                        'ONE_MINUTE': 1440,  # 24*60
                        'FIVE_MINUTE': 288,  # 24*60/5
                        'FIFTEEN_MINUTE': 96,  # 24*60/15
                        'THIRTY_MINUTE': 48,  # 24*60/30
                        'ONE_HOUR': 24,
                        'TWO_HOUR': 12,
                        'SIX_HOUR': 4,
                        'ONE_DAY': 1
                    }
                    
                    estimated_rows = rows_per_day_map.get(granularity.upper(), 1440) * num_days
                    
                    if db.db_type == "sqlite":
                        query_data += f' LIMIT {estimated_rows * 2}'  # Get extra for safety
                    elif db.db_type == "mssql":
                        query_data = query_data.replace('SELECT *', f'SELECT TOP {estimated_rows * 2} *')
                    else:  # PostgreSQL
                        query_data += f' LIMIT {estimated_rows * 2}'
                    
                data = db.read_sql_query(query_data)
                
                if data.empty:
                    logger.warning(f"Table {table} is empty!")
                    continue
                    
                logger.info(f"Retrieved {len(data)} rows from table {table}")
                
                # Handle date column - look for common date column names
                date_columns = ['date', 'timestamp', 'datetime', 'time']
                date_col = None
                for col in date_columns:
                    if col.lower() in [c.lower() for c in data.columns]:
                        # Find the actual column name (case-sensitive)
                        date_col = next(c for c in data.columns if c.lower() == col.lower())
                        break
                
                if date_col:
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                    data = data.dropna(subset=[date_col])  # Remove rows with invalid dates
                    data.set_index(date_col, inplace=True)
                else:
                    # If no date column found, try to convert index
                    if not isinstance(data.index, pd.DatetimeIndex):
                        logger.warning(f"No date column found in table {table}, attempting to convert index")
                        try:
                            data.index = pd.to_datetime(data.index, errors='coerce')
                            data = data.dropna()  # Remove rows with invalid index dates
                        except Exception as e:
                            logger.error(f"Could not convert index to datetime for table {table}: {e}")
                            continue

                # Sort by date (most recent first, then reverse for chronological order)
                data = data.sort_index()
                
                # Apply date filtering if specified
                if num_days is not None:
                    last_date = data.index.max()
                    start_date = last_date - pd.Timedelta(days=num_days)
                    data = data.loc[data.index >= start_date]
                    logger.info(f"Filtered data to last {num_days} days, now have {len(data)} rows")

                # Store the data in the dictionary
                if convert and original_symbol:
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
    finally:
        # Always close the database connection
        try:
            db.close()
        except:
            pass


def export_historical_to_db(data_dict, granularity, db_type="sqlite"):
    """
    Export historical data to database with improved error handling.
    
    Args:
        data_dict (dict): Dictionary with symbol names as keys and DataFrames as values
        granularity (str): The granularity/database name
        db_type (str): Database type ('sqlite', 'mssql', 'postgresql')
    """
    # Database file naming based on type
    if db_type == "sqlite":
        db_file = f'{granularity}.db'
    else:
        db_file = granularity
        
    db = FlexibleDatabase(db_file, db_type)
    
    try:
        for symbol, df in data_dict.items():
            if df.empty:
                logger.warning(f"Skipping empty DataFrame for symbol {symbol}")
                continue
                
            # Create table name (replace hyphens with underscores for SQL compatibility)
            table_name = f"{symbol.replace('-', '_')}_{granularity.lower()}"
            
            try:
                # Use the improved to_sql method with chunking
                db.to_sql(df, table_name, if_exists='replace', index=True, chunksize=1000)
                logger.info(f"Successfully exported {len(df)} rows for {symbol} to table {table_name}")
                
            except Exception as e:
                logger.error(f"Error exporting data for symbol {symbol}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in export_historical_to_db: {e}")
        raise
    finally:
        try:
            db.close()
        except:
            pass


def resample_dataframe_from_db(granularity='ONE_MINUTE', callback=None, socketio=None, db_type="sqlite"):
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

    dict_df = get_historical_from_db(granularity=granularity, db_type=db_type)

    start_time = time.time()
    for i, key in enumerate(times_to_resample.keys()):
        value = times_to_resample[key]
        resampled_dict_df = {}  # Move this inside the loop
        
        for symbol, df in dict_df.items():
            print(f"Processing {symbol} with {len(df)} rows")
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
        
        # Use the improved export function
        export_historical_to_db(resampled_dict_df, granularity=key, db_type=db_type)

        # Progress tracking (assuming utils module exists)
        try:
            import utils
            utils.progress_bar_with_eta(i, data=times_to_resample.keys(), start_time=start_time, 
                                      socketio=socketio, symbol=key, socket_invoker="resampling_progress")
        except ImportError:
            print(f"Completed resampling for {key} ({i+1}/{len(times_to_resample)})")


# Usage examples:
# For local development with SQLite
# db = FlexibleDatabase("local_database.db", "sqlite")

# For production with MSSQL (your existing setup)
# db = FlexibleDatabase("your_database_name", "mssql")

# For PostgreSQL
# db = FlexibleDatabase("your_database_name", "postgresql")
import os
from sqlalchemy import create_engine, text
import pandas as pd
from threading import Lock

class FlexibleDatabase:  # Renamed from MSSQLDatabase
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
            return create_engine(db_path)
        
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
            return create_engine(connection_string)
        
        elif self.db_type == "postgresql":
            # PostgreSQL option
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{self.db_file_name}"
            return create_engine(connection_string)
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def read_sql_query(self, query, params=None):
        """Execute a SELECT query and return results as DataFrame"""
        with self.lock:
            try:
                if params:
                    if isinstance(params, dict):
                        # For named parameters (MSSQL style)
                        return pd.read_sql_query(text(query), self.engine, params=params)
                    elif isinstance(params, (list, tuple)):
                        # For positional parameters (SQLite style)
                        return pd.read_sql_query(query, self.engine, params=params)
                else:
                    return pd.read_sql_query(query, self.engine)
            except Exception as e:
                print(f"Error executing query: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error
    
    def execute_sql(self, query, params=None):
        """Execute a SQL command (INSERT, UPDATE, DELETE, DROP, etc.)"""
        with self.lock:
            try:
                with self.engine.connect() as conn:
                    if params:
                        if isinstance(params, dict):
                            result = conn.execute(text(query), params)
                        elif isinstance(params, (list, tuple)):
                            result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    conn.commit()
                    return result
            except Exception as e:
                print(f"Error executing SQL: {e}")
                raise
    
    def to_sql(self, dataframe, table_name, if_exists='replace', index=True):
        """Write DataFrame to SQL table"""
        with self.lock:
            try:
                dataframe.to_sql(
                    table_name, 
                    self.engine, 
                    if_exists=if_exists, 
                    index=index,
                    method='multi'  # For better performance with large datasets
                )
            except Exception as e:
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
    
    def table_exists(self, table_name):
        """Check if a table exists in the database"""
        tables = self.get_table_names()
        return table_name in tables
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()

# Usage examples:
# For local development with SQLite
# db = FlexibleDatabase("local_database.db", "sqlite")

# For production with MSSQL (your existing setup)
# db = FlexibleDatabase("your_database_name", "mssql")

# For PostgreSQL
# db = FlexibleDatabase("your_database_name", "postgresql")
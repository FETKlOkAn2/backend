import os
import sys
import logging
import time
import datetime as dt
import pandas as pd

# Add your project's core directory to sys.path
sys.path.append('/opt/python')

# Import your existing modules
import core.database_interaction as database_interaction
import core.utils as utils

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class CandleFetcher:
    def __init__(self, db_path):
        self.db_path = db_path
        self.socketio = None  # No socketio in Lambda

    def _get_unix_times(self, granularity, days=1):
        # Your existing implementation
        pass
        
    def _get_data_from_db(self, symbol, granularity):
        # Your existing implementation
        pass
        
    def _get_missing_unix_range(self, desired_start, desired_end, existing_start, existing_end):
        # Your existing implementation
        pass
        
    def _fetch_data(self, symbol, start_unix, end_unix, granularity):
        # Your existing implementation
        pass

    def get_candles_for_db(self, symbols, granularity, days=1):
        """Fetch candles for database storage without socket.io"""
        if not symbols:
            logger.info("No symbols provided.")
            return
            
        # Make sure database directory exists
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            logger.info(f"Created database directory: {self.db_path}")
            
        # Generate time ranges to fetch
        timestamps = self._get_unix_times(granularity, days=days)
        logger.info(f"Generated {len(timestamps)} time ranges to fetch")
        
        # Print example time range for debugging
        if timestamps:
            start_time = dt.datetime.fromtimestamp(timestamps[0][0])
            end_time = dt.datetime.fromtimestamp(timestamps[0][1])
            logger.info(f"Example time range: {start_time} to {end_time}")

        for symbol in symbols:
            logger.info(f"\nFetching data for {symbol}")
            
            # Initialize combined dataframe
            combined_df = pd.DataFrame()
            
            # Get existing data from database
            existing_data = self._get_data_from_db(symbol, granularity)

            if not existing_data.empty:
                logger.info(f"Found existing data for {symbol} with {len(existing_data)} rows")
                existing_start_unix = int(existing_data.index.min().timestamp())
                existing_end_unix = int(existing_data.index.max().timestamp())
                logger.info(f"Existing data range: {dt.datetime.fromtimestamp(existing_start_unix)} to {dt.datetime.fromtimestamp(existing_end_unix)}")
            else:
                logger.info(f"No existing data found for {symbol}")
                existing_start_unix = None
                existing_end_unix = None

            # Determine which time ranges we need to fetch
            missing_date_ranges = []
            for desired_start_unix, desired_end_unix in timestamps:
                # Find what data is missing
                if existing_start_unix is not None and existing_end_unix is not None:
                    missing_ranges = self._get_missing_unix_range(
                        desired_start_unix, desired_end_unix, existing_start_unix, existing_end_unix
                    )
                else:
                    missing_ranges = [(desired_start_unix, desired_end_unix)]

                missing_date_ranges.extend(missing_ranges)

            if not missing_date_ranges:
                logger.info(f"All data for {symbol} is already up to date.")
                continue

            logger.info(f"Need to fetch {len(missing_date_ranges)} time ranges for {symbol}")
            
            # Fetch the missing data
            data_found = False
            start_time = time.time()

            for i, missing_range in enumerate(missing_date_ranges):
                start_unix, end_unix = missing_range
                
                # Print human-readable dates for debugging
                start_date = dt.datetime.fromtimestamp(start_unix)
                end_date = dt.datetime.fromtimestamp(end_unix)
                logger.info(f"Fetching range {i+1}/{len(missing_date_ranges)}: {start_date} to {end_date}")
                
                # Fetch the data
                df = self._fetch_data(symbol, start_unix, end_unix, granularity)

                if not df.empty:
                    rows_fetched = len(df)
                    data_found = True
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                else:
                    logger.info(f"No data found for {symbol} in time range")

                # Update progress (simplified for Lambda)
                progress = ((i + 1) / len(missing_date_ranges)) * 100
                eta = (time.time() - start_time) / (i + 1) * (len(missing_date_ranges) - (i + 1))
                if i % 5 == 0:  # Log every 5th update to avoid excessive logging
                    logger.info(f"Progress for {symbol}: {progress:.1f}%, ETA: {eta:.1f} seconds")

            # Process and save the fetched data
            if data_found and not combined_df.empty:
                # Sort, convert numeric columns, and remove duplicates
                sorted_df = combined_df.sort_values(by="date", ascending=True).reset_index(drop=True)
                columns_to_convert = ['low', 'high', 'open', 'close', 'volume']
                for col in columns_to_convert:
                    sorted_df[col] = pd.to_numeric(sorted_df[col], errors='coerce')
                
                # Handle potential NaN values
                if sorted_df.isna().any().any():
                    sorted_df = sorted_df.interpolate(method='linear')
                
                sorted_df.set_index("date", inplace=True)
                sorted_df = sorted_df[~sorted_df.index.duplicated(keep="first")]
                
                # Add data from existing database if available
                if not existing_data.empty:
                    sorted_df = pd.concat([existing_data, sorted_df])
                    sorted_df = sorted_df[~sorted_df.index.duplicated(keep="last")]  # Keep latest data
                    sorted_df = sorted_df.sort_index()
                
                combined_data = {symbol: sorted_df}
                
                logger.info(f"Saving {len(sorted_df)} rows of data for {symbol}")
                
                # Save to database
                database_interaction.export_historical_to_db(combined_data, granularity=granularity)
                logger.info(f"Data saved successfully for {symbol}")
            else:
                logger.info(f"No new data found for {symbol}")

        # Resample data in the database to create higher timeframes
        logger.info("\nResampling data to higher timeframes...")
        database_interaction.resample_dataframe_from_db(granularity=granularity)
        logger.info("Resampling completed.")

def lambda_handler(event, context):
    """AWS Lambda handler for candle data fetching"""
    try:
        # Get configuration from environment variables
        db_path = os.environ.get('DB_PATH', '/tmp/db')
        
        # Get symbols and granularity from event or use defaults
        symbols = event.get('symbols', ['BTC-USD', 'ETH-USD'])  # Default symbols
        granularity = event.get('granularity', 'ONE_MINUTE')    # Default granularity
        days = event.get('days', 30)                           # Default days
        
        logger.info(f"Starting candle fetch for {len(symbols)} symbols, granularity: {granularity}, days: {days}")
        
        # Initialize fetcher and get candles
        fetcher = CandleFetcher(db_path)
        fetcher.get_candles_for_db(symbols=symbols, granularity=granularity, days=days)
        
        return {
            'statusCode': 200,
            'body': f"Successfully fetched candle data for {len(symbols)} symbols"
        }
        
    except Exception as e:
        logger.error(f"Error fetching candle data: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Error fetching candle data: {str(e)}"
        }
import datetime as dt
import pandas as pd
import core.utils as utils
import sqlite3 as sql
import os
import core.database_interaction as database_interaction
import time
from coinbase.rest import RESTClient
import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv
import os
import sys

class Coinbase_Wrapper():
    def __init__(self, socketio=None):
        load_dotenv(override=True)
        
        self.socketio = socketio

        self.db_path = os.getenv('DATABASE_PATH')
        self.api_key = os.getenv('DOTENV_API_KEY_COINBASE')
        self.api_secret = os.getenv('DOTENV_API_PRIVATE_KEY_COINBASE')

        print("API KEY COINBASE :",self.api_key)
        print("API SECRET COINBASE :",self.api_secret)
        self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
        self.coinbase_robin_crypto = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD']

    
    def _get_unix_times(self, granularity: str, days: int = None):
        # Mapping each timeframe to its equivalent seconds value
        timeframe_seconds = {
            'ONE_MINUTE': 60,
            'FIVE_MINUTE': 300,
            'FIFTEEN_MINUTE': 900,
            'THIRTY_MINUTE': 1800,
            'ONE_HOUR': 3600,
            'TWO_HOUR': 7200,
            'SIX_HOUR': 21600,
            'ONE_DAY': 86400
        }

        # Check if the granularity provided is valid
        if granularity not in timeframe_seconds:
            raise ValueError(f"Invalid granularity '{granularity}'. Must be one of {list(timeframe_seconds.keys())}")

        # Get the current timestamp
        now = int(dt.datetime.now().timestamp())
        limit = 350  # Max number of candles we can fetch
        granularity_seconds = timeframe_seconds[granularity]  # Get the seconds per timeframe unit

        # Calculate max time range for the given granularity
        max_time_range_seconds = limit * granularity_seconds

        # If days are specified, we need to generate pairs of (now, timestamp_max_range) until the number of days is covered
        if days:
            results = []
            seconds_in_day = 86400  # 1 day in seconds
            total_seconds_to_cover = days * seconds_in_day
            remaining_seconds = total_seconds_to_cover

            # Loop until we cover the requested number of days
            while remaining_seconds > 0:
                # Calculate how much time we can cover in this iteration
                current_time_range_seconds = min(max_time_range_seconds, remaining_seconds)

                # Calculate the new timestamp range
                timestamp_max_range = now - current_time_range_seconds

                # Append the pair (timestamp_max_range, now) to the results
                results.append((timestamp_max_range, now))

                # Update 'now' and the remaining seconds
                now = timestamp_max_range
                remaining_seconds -= current_time_range_seconds  # Corrected decrement

            return results[::-1]

        # If no days are specified, return a single pair of (now - max_time_range_seconds, now)
        timestamp_max_range = now - max_time_range_seconds
        return [(timestamp_max_range, now)]


    def _get_data_from_db(self, symbol, granularity):
        """Retrieve existing data for a symbol from the database."""
        conn = sql.connect(f'{self.db_path}/{granularity}.db')
        cursor = conn.cursor()
        symbol_for_table = symbol.replace('-', '_')
        # Get the list of tables that contain the symbol
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f'{symbol_for_table}_%',))
        tables = cursor.fetchall()
        combined_df = pd.DataFrame()
        for table in tables:
            table_name = table[0]
            data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            #print(data.head())
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            combined_df = pd.concat([combined_df, data])
        conn.close()
        if not combined_df.empty:
            combined_df = combined_df.sort_index()
        return combined_df

    def _fetch_data(self, symbol, start_unix, end_unix, granularity):
        max_retries = 4

        for attempt in range(10):
            try:
                btc_candles = self.client.get_candles(
                    product_id=symbol,
                    start=start_unix,
                    end=end_unix,
                    granularity=granularity
                )
                # Convert the response to a DataFrame
                df = utils.to_df(btc_candles)
                return df

            except RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("All retry attempts failed.")
                    raise

            except Exception as e:
                print(f"Unexpected error: {e}")
                raise


    def _get_missing_unix_range(self, desired_start_unix, desired_end_unix, existing_start_unix, existing_end_unix):
        """Determine missing unix time ranges not covered by existing data."""
        missing_ranges = []

        # If desired range is entirely before existing data
        if desired_end_unix < existing_start_unix:
            missing_ranges.append((desired_start_unix, desired_end_unix))

        # If desired range is entirely after existing data
        elif desired_start_unix > existing_end_unix:
            missing_ranges.append((desired_start_unix, desired_end_unix))

        else:
            # Missing range before existing data
            if desired_start_unix < existing_start_unix:
                missing_ranges.append((desired_start_unix, existing_start_unix - 1))

            # Missing range after existing data
            if desired_end_unix > existing_end_unix:
                missing_ranges.append((existing_end_unix + 1, desired_end_unix))

        return missing_ranges

    def get_candles_for_db(self, symbols: list, granularity: str, days: int = None, callback=None, socketio=None):
        """Fetch candles and send real-time progress to frontend."""
        timestamps = self._get_unix_times(granularity, days=days)

        for symbol in symbols:
            if callback:
                callback(f"..getting data for {symbol}")
            if socketio:
                socketio.emit("log_update", {"message": f"Fetching data for {symbol}"})
            print(f"\nFetching data for {symbol}")
            combined_df = pd.DataFrame()
            existing_data = self._get_data_from_db(symbol, granularity)

            if not existing_data.empty:
                existing_start_unix = int(existing_data.index.min().timestamp())
                existing_end_unix = int(existing_data.index.max().timestamp())
            else:
                existing_start_unix = None
                existing_end_unix = None

            missing_date_ranges = []
            for desired_start_unix, desired_end_unix in timestamps:
                missing_ranges = self._get_missing_unix_range(
                    desired_start_unix, desired_end_unix, existing_start_unix, existing_end_unix
                ) if existing_start_unix is not None and existing_end_unix is not None else [(desired_start_unix, desired_end_unix)]

                missing_date_ranges.extend(missing_ranges)

            if not missing_date_ranges:
                if socketio:
                    socketio.emit("log_update", {"message": f"All data for {symbol} is already up to date."})
                continue

            data_found = False
            start_time = time.time()

            for i, missing_range in enumerate(missing_date_ranges):
                start_unix, end_unix = missing_range
                df = self._fetch_data(symbol, start_unix, end_unix, granularity)

                if not df.empty:
                    data_found = True
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

                utils.progress_bar_with_eta(i, missing_date_ranges, start_time, self.socketio, symbol)
                percent = int(((i + 1) / len(missing_date_ranges)) * 100)
                eta = (time.time() - start_time) / (i + 1) * (len(missing_date_ranges) - (i + 1))
                eta_minutes, eta_seconds = divmod(int(eta), 60)
                if socketio:
                    socketio.emit("progress_update", {
                        "symbol": symbol,
                        "progress": percent,
                        "eta": f"{eta_minutes:02d}:{eta_seconds:02d}"
                    })

            if data_found:
                sorted_df = combined_df.sort_values(by="date", ascending=True).reset_index(drop=True)
                columns_to_convert = ['low', 'high', 'open', 'close', 'volume']
                for col in columns_to_convert:
                    sorted_df[col] = pd.to_numeric(sorted_df[col], errors='coerce')
                sorted_df.set_index("date", inplace=True)
                sorted_df = sorted_df[~sorted_df.index.duplicated(keep="first")]
                combined_data = {symbol: sorted_df}

                if socketio:
                    socketio.emit("log_update", {"message": f"Saving data for {symbol}"})
                database_interaction.export_historical_to_db(combined_data, granularity=granularity)
            else:
                if socketio:
                    socketio.emit("log_update", {"message": f"No new data available for {symbol}."})

        # Resample data in the database
        database_interaction.resample_dataframe_from_db(granularity=granularity, callback=callback, socketio=self.socketio)
        if socketio:
            socketio.emit("log_update", {"message": "Resampling database completed."})


    def _get_existing_data(self, symbol: str, granularity: str):
        """Retrieve existing data from the database and get its date range."""
        existing_data = self._get_data_from_db(symbol, granularity)
        if not existing_data.empty:
            existing_start_unix = int(existing_data.index.min().timestamp())
            existing_end_unix = int(existing_data.index.max().timestamp())
        else:
            existing_start_unix = None
            existing_end_unix = None
        return existing_data, existing_start_unix, existing_end_unix

    def _determine_missing_date_ranges(self, timestamps, existing_start_unix, existing_end_unix, fetch_older_data):
        """Determine which date ranges are missing from the existing data."""
        missing_date_ranges = []
        for desired_start_unix, desired_end_unix in timestamps:
            if existing_start_unix is not None and existing_end_unix is not None:
                missing_ranges = self._get_missing_unix_range(
                    desired_start_unix,
                    desired_end_unix,
                    existing_start_unix,
                    existing_end_unix,
                    fetch_older_data=fetch_older_data
                )
            else:
                missing_ranges = [(desired_start_unix, desired_end_unix)]
            missing_date_ranges.extend(missing_ranges)
        return missing_date_ranges

    def _fetch_missing_data(self, symbol: str, missing_date_ranges: list, granularity: str):
        """Fetch data for the missing date ranges."""
        combined_df = pd.DataFrame()
        data_found = False
        start_time = time.time()
        for i, (start_unix, end_unix) in enumerate(missing_date_ranges):
            df = self._fetch_data(symbol, start_unix, end_unix, granularity)
            if not df.empty:
                data_found = True
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            utils.progress_bar_with_eta(
                progress=i,
                data=missing_date_ranges,
                start_time=start_time)
        if data_found:
            return combined_df
        else:
            return pd.DataFrame()

    def _combine_and_process_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame):
        """Combine existing and new data, sort, clean, and remove duplicates."""
        if not new_data.empty:
            if not existing_data.empty:
                combined_df = pd.concat([new_data, existing_data.reset_index()], ignore_index=True)
            else:
                combined_df = new_data
            sorted_df = combined_df.sort_values(by='date', ascending=True).reset_index(drop=True)
            columns_to_convert = ['low', 'high', 'open', 'close', 'volume']
            for col in columns_to_convert:
                sorted_df[col] = pd.to_numeric(sorted_df[col], errors='coerce')
            sorted_df.set_index('date', inplace=True)
            # Remove duplicates based on index
            sorted_df = sorted_df[~sorted_df.index.duplicated(keep='first')]
            return sorted_df
        else:
            return existing_data

    def _export_data_to_db(self, combined_data: dict, granularity: str):
        """Export the combined data to the database."""
        database_interaction.export_historical_to_db(combined_data, granularity=granularity)

    def _resample_data_in_db(self, granularity: str):
        """Resample data in the database."""
        database_interaction.resample_dataframe_from_db(granularity=granularity)

    def get_basic_candles(self, symbols:list,timestamps,granularity):
        combined_data = {}
        for symbol in symbols:
            combined_df = pd.DataFrame()
            for start, end in timestamps:
                df = self._fetch_data(symbol, start, end, granularity)

                combined_df = pd.concat([combined_df, df])

            combined_df = combined_df.sort_values(by='date', ascending=True).reset_index(drop=True)
            columns_to_convert = ['low', 'high', 'open', 'close', 'volume']

            for col in columns_to_convert:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

            combined_df.set_index('date', inplace=True)
            combined_data[symbol] = combined_df
        return combined_data
                



# granularity = 'ONE_MINUTE'
# coinbase = Coinbase_Wrapper()

# coinbase.get_candles_for_db(
#     symbols=coinbase.coinbase_robin_crypto,
#     granularity=granularity,
#     days=365
#     )
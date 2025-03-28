import pandas as pd
import datetime as dt
import numpy as np
import sqlite3 as sql
import inspect
import re
import sys
import time
from numba import njit
pd.set_option('future.no_silent_downcasting', True)


def convert_symbols(strategy_object: object = None, lone_symbol: str = None, to_kraken: bool = False, to_robinhood: bool = False):
    # Define symbol mappings
    coinbase_crypto = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD']
    robinhood_crypto = ['BTC', 'ETH', 'DOGE', 'SHIB', 'AVAX', 'BCH', 'LINK', 'UNI', 'LTC', 'XLM', 'ETC', 'AAVE', 'XTZ', 'COMP']
    kraken_crypto = ['XXBTZUSD', 'XETHZUSD', 'XDGUSD', 'SHIBUSD', 'AVAXUSD', 'BCHUSD', 'LINKUSD', 'UNIUSD', 'XLTCZUSD', 'XXLMZUSD', 'XETCZUSD', 'AAVEUSD', 'XTZUSD', 'COMPUSD']

    # Determine the current symbol
    if strategy_object is not None:
        current_symbol = strategy_object.symbol
    elif lone_symbol is not None:
        current_symbol = lone_symbol
    else:
        raise ValueError("Either a strategy_object or lone_symbol must be provided.")

    # Determine the source symbol list
    if current_symbol in coinbase_crypto:
        source_list = coinbase_crypto
    elif current_symbol in robinhood_crypto:
        source_list = robinhood_crypto
    elif current_symbol in kraken_crypto:
        source_list = kraken_crypto
    else:
        raise ValueError(f"Symbol '{current_symbol}' not found in any known lists.")

    # Get the index of the current symbol
    try:
        symbol_index = source_list.index(current_symbol)
    except ValueError:
        raise ValueError(f"Symbol '{current_symbol}' not found in the source list.")

    # Convert to the target symbol
    if to_kraken:
        return kraken_crypto[symbol_index]
    elif to_robinhood:
        return robinhood_crypto[symbol_index]
    else:
        return coinbase_crypto[symbol_index]

@njit
def format_signals(signals):
    """
    Formats signals to avoid double buys or sells.
    Optimized using NumPy arrays with a loop.
    """
    formatted_signals = np.zeros_like(signals)
    in_position = False

    for i in range(len(signals)):
        if signals[i] == 1 and not in_position:
            formatted_signals[i] = 1
            in_position = True
        elif signals[i] == -1 and in_position:
            formatted_signals[i] = -1
            in_position = False
        # Else, no change; formatted_signals[i] remains 0
        # in_position remains the same

    return formatted_signals

@njit
def calculate_with_sizing_numba(signal, close, percent_to_size):
    n = len(signal)
    new_signal = signal.copy()
    saved_close = 0.0
    tracking = False

    for i in range(n):
        if signal[i] == 1 and not tracking:
            saved_close = close[i]
            tracking = True
        elif tracking:
            target_close = saved_close * (1 + percent_to_size)

            if signal[i] == 0:
                if close[i] >= target_close:
                    new_signal[i] = 1
                    saved_close = close[i]
                elif close[i] <= target_close * (1 - (percent_to_size * 2)):
                    saved_close = close[i]

            if signal[i] == -1:
                tracking = False
                saved_close = 0.0
    return new_signal

import time
import sys

def progress_bar_with_eta(progress, data, start_time, socketio=None, symbol=None, socket_invoker="progress_update"):
    total = len(data)
    progress += 1  # Increment progress

    elapsed_time = time.time() - start_time  
    avg_time_per_item = elapsed_time / progress if progress > 0 else 0
    eta = avg_time_per_item * (total - progress)

    eta_minutes = int(eta // 60)
    eta_seconds = int(eta % 60)

    percent = int((progress / total) * 100) if total > 0 else 100

    # ✅ Only emit if socketio is not None
    if socketio:
        socketio.emit(socket_invoker, {
            "symbol": symbol,
            "progress": percent,
            "eta": f"{eta_minutes:02d}:{eta_seconds:02d}"
        })
    
    # Still print to console if needed
    sys.stdout.write(f'\r[{symbol}] {percent}% Complete | ETA: {eta_minutes:02d}:{eta_seconds:02d}')
    sys.stdout.flush()


def find_unix(days_ago: int):
    now = dt.datetime.now()
    start = now - dt.timedelta(days=days_ago)
    unix_start_time = int(round(start.timestamp(),0))
    return unix_start_time

def heikin_ashi_transform(dict_df):
    ha_dict = {}

    for symbol, df in dict_df.items():
        ha_df = pd.DataFrame(index=df.index)

        ha_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        ha_df.loc[ha_df.index[0], 'open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2

        ha_df['high'] = pd.NA
        ha_df['low'] = pd.NA


        for i in range(1, len(df)):
            # Heikin-Ashi Open: midpoint of previous HA_Open and HA_Close
            ha_df.loc[ha_df.index[i], 'open'] = (ha_df.loc[ha_df.index[i-1], 'open'] + ha_df.loc[ha_df.index[i-1], 'close']) / 2

            # Heikin-Ashi High: maximum of the High, HA_Open, and HA_Close of the current bar
            ha_df.loc[ha_df.index[i], 'high'] = max(df['high'].iloc[i], ha_df.loc[ha_df.index[i], 'open'], ha_df.loc[ha_df.index[i], 'close'])

            # Heikin-Ashi Low: minimum of the Low, HA_Open, and HA_Close of the current bar
            ha_df.loc[ha_df.index[i], 'low'] = min(df['low'].iloc[i], ha_df.loc[ha_df.index[i], 'open'], ha_df.loc[ha_df.index[i], 'close'])

        ha_df['volume'] = df['volume']
        ha_df.dropna(inplace=True)


        ha_dict[symbol] = ha_df

    return ha_dict

def to_df(data_dict: dict):
    if not data_dict:
        return pd.DataFrame()  # Return empty DataFrame if data_dict is empty
    key = next(iter(data_dict))  # gets the first key
    df = pd.DataFrame(data_dict[key])  # converts to df
    if key == 'candles':
        if not df.empty:
            df.columns = ['start', 'low', 'high', 'open', 'close', 'volume']
            df['start'] = pd.to_datetime(df['start'].astype(float), unit='s')
            df = df.rename(columns={'start': 'date'})  # renames the start column to date
            df.loc[:, df.columns != 'date'] = df.loc[:, df.columns != 'date'].apply(pd.to_numeric, errors='coerce')
            df = df.ffill()

        else:
            pass
            #print(f"No candle data available.")
    return df
 

# In utils.py
def convert_resampling_to_granularity(resampling):
    """Convert resampling string to granularity format"""
    # Map resampling formats to granularity formats
    resampling_map = {
        '1m': 'ONE_MINUTE',
        '5m': 'FIVE_MINUTE',
        '15m': 'FIFTEEN_MINUTE',
        '30m': 'THIRTY_MINUTE',
        '1h': 'ONE_HOUR',
        '2h': 'TWO_HOUR',
        '6h': 'SIX_HOUR',
        '1d': 'ONE_DAY'
    }
    
    return resampling_map.get(resampling, 'FIFTEEN_MINUTE')  # Default to 15 minutes
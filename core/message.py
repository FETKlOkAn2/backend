import os
import pandas as pd
import numpy as np
import cupy as cp
import vectorbt as vbt
import time
import gc
from core.strategies.strategy import Strategy
from core.risk import Risk_Handler
import core.database_interaction
import core.utils.utils as utils

# Define global functions

def calculate_rsi_gpu(close_gpu, window):
    delta = close_gpu[1:] - close_gpu[:-1]
    gain = cp.maximum(delta, 0)
    loss = cp.maximum(-delta, 0)

    # Calculate rolling averages (simple moving average for RSI)
    avg_gain = cp.convolve(gain, cp.ones(window) / window, mode='valid')
    avg_loss = cp.convolve(loss, cp.ones(window) / window, mode='valid')
    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Align output size
    pad_length = close_gpu.shape[0] - rsi.shape[0]
    rsi = cp.concatenate([cp.full(pad_length, cp.nan), rsi])
    return rsi

def calculate_adx_gpu(high_gpu, low_gpu, close_gpu, time_period):
    # True Range
    tr1 = cp.abs(high_gpu[1:] - low_gpu[1:])
    tr2 = cp.abs(high_gpu[1:] - close_gpu[:-1])
    tr3 = cp.abs(low_gpu[1:] - close_gpu[:-1])
    true_range = cp.maximum(tr1, cp.maximum(tr2, tr3))

    # Directional Movement
    plus_dm = cp.maximum(high_gpu[1:] - high_gpu[:-1], 0)
    minus_dm = cp.maximum(low_gpu[:-1] - low_gpu[1:], 0)

    plus_dm = cp.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = cp.where(minus_dm > plus_dm, minus_dm, 0)

    # Smoothed averages
    atr = cp.convolve(true_range, cp.ones(time_period) / time_period, mode='valid')
    plus_di = 100 * cp.convolve(plus_dm, cp.ones(time_period) / time_period, mode='valid') / atr
    minus_di = 100 * cp.convolve(minus_dm, cp.ones(time_period) / time_period, mode='valid') / atr

    # ADX calculation
    dx = cp.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = cp.convolve(dx, cp.ones(time_period) / time_period, mode='valid')

    # Align output size
    pad_length = high_gpu.shape[0] - adx.shape[0]
    adx = cp.concatenate([cp.full(pad_length, cp.nan), adx])
    return adx

def combine_signals(signals1, signals2):
    combined_signals = signals1.copy()
    combined_signals[signals2 == 0] = 0
    return combined_signals

def format_signals(signals):
    """
    Formats signals to avoid double buys or sells.
    Vectorized implementation using NumPy.
    """
    # Initialize formatted signals array
    formatted_signals = np.zeros_like(signals)

    # Identify where signals change (entries/exits)
    signal_changes = signals != 0

    # Compute cumulative sum of signal changes
    cumulative_signals = np.cumsum(signal_changes)

    # Compute previous in_position status
    prev_in_position = (np.concatenate([[0], cumulative_signals[:-1]]) % 2) == 1

    # Allowed entries: signals == 1 and not in position
    allowed_entries = (signals == 1) & (~prev_in_position)

    # Allowed exits: signals == -1 and in position
    allowed_exits = (signals == -1) & prev_in_position

    # Set formatted signals
    formatted_signals[allowed_entries] = 1
    formatted_signals[allowed_exits] = -1

    return formatted_signals

def custom_indicator(close, high, low, rsi_window=20, buy_threshold=15, sell_threshold=70,
                     adx_buy_threshold=20, adx_time_period=20):
    # Convert data to CuPy arrays
    close_gpu = cp.array(close)
    high_gpu = cp.array(high)
    low_gpu = cp.array(low)

    # Calculate RSI
    rsi = calculate_rsi_gpu(close_gpu, rsi_window)
    rsi_np = cp.asnumpy(rsi)

    # Generate RSI signals
    buy_signal = rsi_np < buy_threshold
    sell_signal = rsi_np > sell_threshold

    # Create initial signals
    signals = np.zeros_like(close, dtype=int)
    signals[buy_signal] = 1
    signals[sell_signal] = -1

    # Calculate ADX
    adx = calculate_adx_gpu(high_gpu, low_gpu, close_gpu, adx_time_period)
    adx_np = cp.asnumpy(adx)

    # Generate ADX signals
    buy_signal_adx = adx_np > adx_buy_threshold
    sell_signal_adx = ~buy_signal_adx

    # Create ADX signals
    signals_adx = np.zeros_like(close, dtype=int)
    signals_adx[buy_signal_adx] = 1
    signals_adx[sell_signal_adx] = -1

    # Combine RSI and ADX signals
    final_signals = combine_signals(signals, signals_adx)

    # Format signals to avoid double entries/exits
    final_signals = format_signals(final_signals)

    return final_signals

# Now, adjust the Hyper class

class Hyper(Strategy):
    """A class to handle Hyper Optimization backtests"""
    def __init__(self, strategy_object, **kwargs):
        """Initiates strategy resources"""
        dict_df = {strategy_object.symbol: strategy_object.df}
        super().__init__(dict_df=dict_df)
        self.strategy = strategy_object
        if self.strategy.risk_object is not None:
            self.risk_object = self.strategy.risk_object
        self.with_sizing = self.strategy.with_sizing

        possible_inputs = ['open', 'high', 'low', 'close', 'volume']
        self.input_names = []
        self.inputs = []
        self.params = {}
        for key, value in kwargs.items():
            if key in possible_inputs:
                self.input_names.append(key)
                self.inputs.append(value)
                setattr(self, key, value)
            else:
                self.params[key] = value

        self.ind = self.build_indicator_factory()
        self.res = self.generate_signals()
        self.entries, self.exits = self.convert_signals()
        self.pf = self.run_portfolio()
        self.returns = self.pf.total_return()
        self.max = self.returns.max()

    def build_indicator_factory(self):
        """Builds the Indicator Factory"""
        param_names = list(self.params.keys())

        ind = vbt.IndicatorFactory(
            class_name="Custom",
            short_name="cust",
            input_names=self.input_names,
            param_names=param_names,
            output_names=['value']
        ).from_apply_func(
            custom_indicator,  # Use the standalone function
            to_2d=False
        )
        return ind

    def generate_signals(self):
        """Generates the entries/exits signals"""
        try:
            # Convert inputs and parameters to NumPy arrays
            inputs_numpy = [cp.asnumpy(inp) if isinstance(inp, cp.ndarray) else inp for inp in self.inputs]
            params_numpy = {key: cp.asnumpy(value) if isinstance(value, cp.ndarray) else value for key, value in self.params.items()}

            res = self.ind.run(
                *inputs_numpy,
                **params_numpy,
                param_product=True
            )
        except Exception as e:
            print(f"Error during signal generation: {e}\n")
            res = None
        return res

    def convert_signals(self):
        """Converts signals to entries and exits."""
        if self.res is None:
            raise ValueError("Signal generation failed; self.res is None.")

        signals = self.res.value

        # Determine entries and exits
        entries = signals == 1
        exits = signals == -1

        # Use the index from signals
        time_index = signals.index

        self.entries = entries
        self.exits = exits

        return self.entries, self.exits

    def run_portfolio(self):
        """Performs backtest"""
        close_data = getattr(self, 'close', self.close)
        open_data = getattr(self, 'open', self.open)
        pf = vbt.Portfolio.from_signals(
            close_data,
            self.entries,
            self.exits
        )
        return pf

# Adjust the RSI_ADX class

class RSI_ADX(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=False):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)
        self.close = self.close
        self.high = self.high
        self.low = self.low

# Now, update the TestBed class

class TestBed():
    def __init__(self):
        risk = Risk_Handler()

        dict_df = core.database_interaction.get_historical_from_db(
            granularity='ONE_MINUTE',
            symbols=symbols,
            num_days=12
        )
        print(f'...Running hyper on {len(symbols)} symbols')

        start_time = time.time()
        for i, items in enumerate(dict_df.items()):
            key, value = items
            current_dict = {key: value}

            strat = RSI_ADX(current_dict, risk_object=risk, with_sizing=True)

            hyper = Hyper(
                strategy_object=strat,
                close=strat.close,
                high=strat.high,
                low=strat.low,
                rsi_window=np.arange(10, 30, step=10),
                buy_threshold=np.arange(5, 51, step=10),
                sell_threshold=np.arange(50, 96, step=10),
                adx_buy_threshold=np.arange(20, 60, step=20),
                adx_time_period=np.arange(10, 30, step=10)
            )

            # Optional: Save results to the database
            # database_interaction.export_hyper_to_db(strategy=strat, hyper=hyper)
            print(f"Execution Time: {time.time() - start_time}")
            utils.progress_bar_with_eta(
                progress=i,
                data=dict_df.keys(),
                start_time=start_time
            )
            del hyper
            gc.collect()

        print(f"Execution Time: {time.time() - start_time}")

# Run the test
if __name__ == "__main__":
    symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD',
               'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD',
               'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD']

    test = TestBed()

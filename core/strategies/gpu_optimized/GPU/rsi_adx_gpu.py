from core.strategies.strategy import Strategy
import cupy as cp
import numpy as np
import pandas as pd
import core.utils.utils as utils

class RSI_ADX_GPU(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=True, hyper=None):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)
        self.hyper = hyper

    def custom_indicator(self, close=None, rsi_window=20, buy_threshold=15, sell_threshold=70,
                            adx_buy_threshold=30, adx_time_period=20):
        if not self.hyper:
            self.rsi_window = rsi_window
            self.buy_threshold = buy_threshold
            self.sell_threshold = sell_threshold
            self.adx_buy_threshold = adx_buy_threshold
            self.adx_time_period = adx_time_period

        # Calculate RSI
        rsi = self.calculate_rsi_gpu(self.close_gpu, rsi_window)
        rsi_np = cp.asnumpy(rsi)
        rsi_np = np.nan_to_num(rsi_np, nan=0.0)  # Replace NaN with 0

        # Generate RSI Signals
        buy_signal = rsi_np < buy_threshold
        sell_signal = rsi_np > sell_threshold

        signals = np.zeros_like(self.close, dtype=int)
        signals[buy_signal] = 1
        signals[sell_signal] = -1

        signals = utils.format_signals(signals)

        # Calculate ADX
        adx = self.calculate_adx_gpu(self.high_gpu, self.low_gpu, self.close_gpu, adx_time_period)
        adx_np = cp.asnumpy(adx)

        # Generate ADX signals
        buy_signal_adx = adx_np > adx_buy_threshold
        sell_signal_adx = ~buy_signal_adx

        signals_adx = np.zeros_like(self.close, dtype=int)
        signals_adx[buy_signal_adx] = 1
        signals_adx[sell_signal_adx] = -1

        # Combine RSI and ADX signals
        final_signals = self.combine_signals(signals, signals_adx)
        final_signals = utils.format_signals(final_signals)

        if self.with_sizing:
            percent_to_size = self.risk_object.percent_to_size
            close_array = self.close.to_numpy(dtype=np.float64)
            signal_array = np.array(final_signals)
            final_signals = utils.calculate_with_sizing_numba(signal_array, close_array, percent_to_size)
 

        if not self.hyper:
            self.osc1_data = ('RSI', rsi_np)
            self.osc2_data = ('ADX', adx_np)
            self.signals = final_signals
            self.entries = np.zeros_like(self.signals, dtype=bool)
            self.exits = np.zeros_like(self.signals, dtype=bool)

            self.entries[self.signals == 1] = True
            self.exits[self.signals == -1] = True

            self.entries = pd.Series(self.entries, index=self.close.index)
            self.exits = pd.Series(self.exits, index=self.close.index)

        return final_signals


    def calculate_rsi_gpu(self, close_gpu, rsi_window):
        delta = close_gpu[1:] - close_gpu[:-1]
        gain = cp.maximum(delta, 0)
        loss = cp.maximum(-delta, 0)

        avg_gain = cp.convolve(gain, cp.ones(rsi_window) / rsi_window, mode='valid')
        avg_loss = cp.convolve(loss, cp.ones(rsi_window) / rsi_window, mode='valid')

        # Handle zero division and calculate RS
        rs = cp.where(avg_loss == 0, cp.inf, avg_gain / avg_loss)  # Handle division by zero
        rsi = 100 - (100 / (1 + rs))

        # If RSI calculation results in NaN (due to division by zero), set those to 100 (no loss)
        rsi = cp.where(cp.isnan(rsi), 100, rsi)

        pad_length = close_gpu.shape[0] - rsi.shape[0]
        rsi = cp.concatenate([cp.full(pad_length, cp.nan), rsi])
        return rsi

    
    def calculate_adx_gpu(self, high_gpu, low_gpu, close_gpu, adx_time_period):
        tr1 = cp.abs(high_gpu[1:] - low_gpu[1:])
        tr2 = cp.abs(high_gpu[1:] - close_gpu[:-1])
        tr3 = cp.abs(low_gpu[1:] - close_gpu[:-1])
        true_range = cp.maximum(tr1, cp.maximum(tr2, tr3))

        plus_dm = cp.maximum(high_gpu[1:] - high_gpu[:-1], 0)
        minus_dm = cp.maximum(low_gpu[:-1] - low_gpu[1:], 0)

        plus_dm = cp.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = cp.where(minus_dm > plus_dm, minus_dm, 0)

        atr = cp.convolve(true_range, cp.ones(adx_time_period) / adx_time_period, mode='valid')
        plus_di = 100 * cp.convolve(plus_dm, cp.ones(adx_time_period) / adx_time_period, mode='valid') / atr
        minus_di = 100 * cp.convolve(minus_dm, cp.ones(adx_time_period) / adx_time_period, mode='valid') / atr

        # ADX calculation
        dx = cp.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = cp.convolve(dx, cp.ones(adx_time_period) / adx_time_period, mode='valid')

        # Align output size
        pad_length = high_gpu.shape[0] - adx.shape[0]
        adx = cp.concatenate([cp.full(pad_length, cp.nan), adx])
        return adx

    def combine_signals(self, *signals):
        signals_array = np.array(signals)  
        all_ones = np.all(signals_array == 1, axis=0)
        all_neg_ones = np.all(signals_array == -1, axis=0)
        combined_signals = np.zeros(signals_array.shape[1], dtype=int)
        combined_signals[all_ones] = 1
        combined_signals[all_neg_ones] = -1
        return combined_signals

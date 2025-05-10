from core.strategies.strategy import Strategy
import numpy as np
import pandas as pd
import backend.core.utils.utils as utils

class RSI_ADX_NP(Strategy):
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
        rsi = self.calculate_rsi(self.close_np, rsi_window)
        rsi_np = np.array(rsi)
        #np.savetxt("output_rsi_array_numpy.txt", rsi_np.tolist(), fmt="%d")

        # Generate RSI Signals
        buy_signal = rsi_np < buy_threshold
        sell_signal = rsi_np > sell_threshold


        signals = np.zeros_like(self.close_np, dtype=int)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1

        signals = utils.format_signals(signals)

        # Calculate ADX
        adx = self.calculate_adx(self.high_np, self.low_np, self.close_np, adx_time_period)
        adx_np = np.array(adx)

        # Generate ADX signals
        buy_signal_adx = adx_np > adx_buy_threshold
        sell_signal_adx = ~buy_signal_adx

        signals_adx = np.zeros_like(self.close_np, dtype=int)
        signals_adx[buy_signal_adx] = 1
        signals_adx[sell_signal_adx] = -1

        # Combine RSI and ADX signals
        final_signals = self.combine_signals(signals, signals_adx)
        final_signals = utils.format_signals(final_signals)


        if self.with_sizing:
            percent_to_size = self.risk_object.percent_to_size
            # close_array = self.close_np.to_numpy(dtype=np.float64)
            signal_array = np.array(final_signals)
            final_signals = utils.calculate_with_sizing_numba(signal_array, self.close_np, percent_to_size)

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

    def calculate_rsi(self, close, rsi_window):
        
        delta = close[1:] - close[:-1]
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(rsi_window) / rsi_window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(rsi_window) / rsi_window, mode='valid')
        
        # Handle zero division with a small epsilon instead of np.inf
        epsilon = 1e-10  # Small number to avoid division by zero
        rs = np.where(avg_loss == 0, 0, avg_gain / (avg_loss + epsilon))  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # If RSI calculation results in NaN (due to division by zero), set those to 100 (no loss)
        rsi = np.where(np.isnan(rsi), 100, rsi)

        # Align output size with padding (like CuPy)
        pad_length = close.shape[0] - rsi.shape[0]
        rsi = np.concatenate([np.full(pad_length, np.nan), rsi])


        return rsi


    def calculate_adx(self, high, low, close, adx_time_period):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))

        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)

        plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)


        atr = np.convolve(true_range, np.ones(adx_time_period) / adx_time_period, mode='valid')
        plus_dm_avg = np.convolve(plus_dm, np.ones(adx_time_period) / adx_time_period, mode='valid')
        minus_dm_avg = np.convolve(minus_dm, np.ones(adx_time_period) / adx_time_period, mode='valid')



        # Adjust lengths to ensure alignment
        length = min(len(atr), len(plus_dm_avg), len(minus_dm_avg))
        atr = atr[:length]
        plus_dm_avg = plus_dm_avg[:length]
        minus_dm_avg = minus_dm_avg[:length]

        # np.savetxt("output_ADX.txt", atr.tolist())

        plus_di = 100 * plus_dm_avg / atr
        minus_di = 100 * minus_dm_avg / atr

        dx = np.where((plus_di + minus_di) != None, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 
                  0)

        adx = np.convolve(dx, np.ones(adx_time_period) / adx_time_period, mode='valid')

        # Pad the result to match the original input length
        pad_length = high.shape[0] - adx.shape[0]
        adx = np.concatenate([np.full(pad_length, np.nan), adx])

        return adx


    def combine_signals(self, *signals):
        signals_array = np.array(signals)  
        all_ones = np.all(signals_array == 1, axis=0)
        all_neg_ones = np.all(signals_array == -1, axis=0)
        combined_signals = np.zeros(signals_array.shape[1], dtype=int)
        combined_signals[all_ones] = 1
        combined_signals[all_neg_ones] = -1
        return combined_signals

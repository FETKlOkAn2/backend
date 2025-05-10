from core.strategies.strategy import Strategy
import cupy as cp
import numpy as np
import pandas as pd
import backend.core.utils.utils as utils


class BollingerBands_VWAP_GPU(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=True, hyper=None):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)
        self.hyper = hyper

    def custom_indicator(self, close=None, volume=None, bb_period=20, bb_dev=2, vwap_window=None):
        if not self.hyper:
            self.bb_period = bb_period
            self.bb_dev = bb_dev

        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands_gpu(
            self.close_gpu, bb_period, bb_dev
        )
        bb_buy_signal = self.close_gpu < lower_band
        bb_sell_signal = self.close_gpu > upper_band

        # Calculate VWAP with optional window
        vwap = self.calculate_vwap_gpu(self.close_gpu, self.volume_gpu, vwap_window)
        vwap_buy_signal = self.close_gpu < vwap
        vwap_sell_signal = self.close_gpu > vwap

        # Combine Signals
        bb_signals = cp.zeros_like(self.close_gpu, dtype=int)
        bb_signals[bb_buy_signal] = 1
        bb_signals[bb_sell_signal] = -1

        vwap_signals = cp.zeros_like(self.close_gpu, dtype=int)
        vwap_signals[vwap_buy_signal] = 1
        vwap_signals[vwap_sell_signal] = -1

        final_signals = self.combine_signals(bb_signals, vwap_signals)
        final_signals = cp.asnumpy(final_signals)  # Convert to NumPy for further processing
        final_signals = utils.format_signals(final_signals)

        # Handle risk sizing
        if self.with_sizing:
            percent_to_size = self.risk_object.percent_to_size
            close_array = self.close.to_numpy(dtype=np.float64)  # Ensure NumPy array
            final_signals = utils.calculate_with_sizing_numba(
                final_signals, close_array, percent_to_size
            )

        # Save state
        if not self.hyper:
            self.osc1_data = (
                "BollingerBands",
                cp.asnumpy(upper_band),
                cp.asnumpy(middle_band),
                cp.asnumpy(lower_band),
            )
            self.osc2_data = ("VWAP", cp.asnumpy(vwap))
            self.signals = final_signals
            self.entries = np.zeros_like(self.signals, dtype=bool)
            self.exits = np.zeros_like(self.signals, dtype=bool)

            self.entries[self.signals == 1] = True
            self.exits[self.signals == -1] = True

            self.entries = pd.Series(self.entries, index=self.close.index)
            self.exits = pd.Series(self.exits, index=self.close.index)

        return final_signals

    def calculate_bollinger_bands_gpu(self, close_gpu, period, dev):
        """
        Calculate Bollinger Bands using CuPy.
        """
        # Calculate the moving average (middle band)
        middle_band = cp.convolve(close_gpu, cp.ones(period) / period, mode="valid")

        # Calculate rolling standard deviation
        rolling_std = cp.empty_like(close_gpu)
        rolling_std[: period - 1] = cp.nan
        for i in range(period - 1, len(close_gpu)):
            rolling_std[i] = cp.std(close_gpu[i - period + 1 : i + 1])

        # Calculate upper and lower bands
        upper_band = middle_band + (dev * rolling_std[period - 1 :])
        lower_band = middle_band - (dev * rolling_std[period - 1 :])

        # Pad to align with original length
        pad_length = len(close_gpu) - len(middle_band)
        middle_band = cp.concatenate([cp.full(pad_length, cp.nan), middle_band])
        upper_band = cp.concatenate([cp.full(pad_length, cp.nan), upper_band])
        lower_band = cp.concatenate([cp.full(pad_length, cp.nan), lower_band])

        return upper_band, middle_band, lower_band

    def calculate_vwap_gpu(self, close_gpu, volume_gpu, window=None):
        """
        Calculate VWAP using CuPy with an optional rolling window.
        """
        if window is None:
            # Cumulative VWAP
            cumulative_price_volume = cp.cumsum(close_gpu * volume_gpu)
            cumulative_volume = cp.cumsum(volume_gpu)
            with cp.errstate(divide="ignore", invalid="ignore"):
                vwap = cp.where(
                    cumulative_volume == 0, cp.nan, cumulative_price_volume / cumulative_volume
                )
            return vwap

        # Rolling VWAP
        vwap = cp.full_like(close_gpu, cp.nan)  # Initialize with NaNs
        for i in range(window - 1, len(close_gpu)):
            window_close = close_gpu[i - window + 1 : i + 1]
            window_volume = volume_gpu[i - window + 1 : i + 1]
            vwap[i] = cp.sum(window_close * window_volume) / cp.sum(window_volume)

        return vwap

    def combine_signals(self, *signals):
        """
        Combine multiple signals into a final signal.
        """
        signals_array = cp.array(signals)
        all_ones = cp.all(signals_array == 1, axis=0)
        all_neg_ones = cp.all(signals_array == -1, axis=0)
        combined_signals = cp.zeros(signals_array.shape[1], dtype=int)
        combined_signals[all_ones] = 1
        combined_signals[all_neg_ones] = -1
        return combined_signals

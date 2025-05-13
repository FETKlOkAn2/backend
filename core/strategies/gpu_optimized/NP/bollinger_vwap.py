from core.strategies.strategy import Strategy
import numpy as np
import core.utils.utils as utils


class BollingerBands_VWAP(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=True, hyper=None):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)
        self.hyper = hyper

    def custom_indicator(self, close=None, volume=None, bb_period=20, bb_dev=2):
        if not self.hyper:
            self.bb_period = bb_period
            self.bb_dev = bb_dev

        # Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(self.close_np, bb_period, bb_dev)
        bb_buy_signal = self.close_np < lower_band
        bb_sell_signal = self.close_np > upper_band

        # VWAP
        vwap = self.calculate_vwap(self.close_np, self.volume_np)
        vwap_buy_signal = self.close_np < vwap
        vwap_sell_signal = self.close_np > vwap

        # Combine signals
        bb_signals = np.zeros_like(self.close_np, dtype=int)
        bb_signals[bb_buy_signal] = 1
        bb_signals[bb_sell_signal] = -1

        vwap_signals = np.zeros_like(self.close_np, dtype=int)
        vwap_signals[vwap_buy_signal] = 1
        vwap_signals[vwap_sell_signal] = -1

        final_signals = self.combine_signals(bb_signals, vwap_signals)
        final_signals = utils.format_signals(final_signals)

        if self.with_sizing:
            final_signals = utils.calculate_with_sizing_numba(final_signals, self.close_np, self.risk_object.percent_to_size)

        self.osc1_data = ('BollingerBands', upper_band, middle_band, lower_band)
        self.osc2_data = ('VWAP', vwap)
        self.signals = final_signals

        return final_signals

    def calculate_bollinger_bands(self, close, period, dev):
        """
        Calculate Bollinger Bands manually using NumPy.
        """
        middle_band = np.convolve(close, np.ones(period) / period, mode='valid')
        rolling_std = np.array([
            np.std(close[i - period + 1:i + 1]) if i >= period - 1 else np.nan
            for i in range(len(close))
        ])
        upper_band = middle_band + (dev * rolling_std[period - 1:])
        lower_band = middle_band - (dev * rolling_std[period - 1:])

        pad_length = len(close) - len(middle_band)
        middle_band = np.concatenate([np.full(pad_length, np.nan), middle_band])
        upper_band = np.concatenate([np.full(pad_length, np.nan), upper_band])
        lower_band = np.concatenate([np.full(pad_length, np.nan), lower_band])

        return upper_band, middle_band, lower_band

    def calculate_vwap(self, close, volume):
        """
        Calculate VWAP (Volume Weighted Average Price) manually using NumPy.
        
        :param close: Array of closing prices.
        :param volume: Array of volumes.
        :return: Array of VWAP values.
        """
        # Cumulative totals of (price * volume) and volume
        cumulative_price_volume = np.cumsum(close * volume)
        cumulative_volume = np.cumsum(volume)

        # Calculate VWAP
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.where(cumulative_volume == 0, np.nan, cumulative_price_volume / cumulative_volume)

        return vwap

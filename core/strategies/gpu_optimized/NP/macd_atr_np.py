from core.strategies.strategy import Strategy
import numpy as np
import pandas as pd
import core.utils.utils as utils
import talib as ta

class MACD_ATR(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=True, hyper=None):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)
        self.hyper = hyper

    def custom_indicator(self, close=None, macd_fast=12, macd_slow=26, macd_signal=9, atr_period=14):
        if not self.hyper:
            self.macd_fast = macd_fast
            self.macd_slow = macd_slow
            self.macd_signal = macd_signal
            self.atr_period = atr_period

        # MACD
        macd, macd_signal_line, _ = self.calculate_macd(self.close_np, macd_fast, macd_slow, macd_signal)
        macd_buy_signal = macd > macd_signal_line
        macd_sell_signal = macd < macd_signal_line

        # ATR
        atr = self.calculate_atr(self.high_np, self.low_np, self.close_np, atr_period)
        atr_signal = atr > np.percentile(atr, 75)  # Use high ATR as a signal

        # Combine signals
        macd_signals = np.zeros_like(self.close_np, dtype=int)
        macd_signals[macd_buy_signal] = 1
        macd_signals[macd_sell_signal] = -1

        atr_signals = np.zeros_like(self.close_np, dtype=int)
        atr_signals[atr_signal] = 1

        final_signals = self.combine_signals(macd_signals, atr_signals)
        final_signals = utils.format_signals(final_signals)

        if self.with_sizing:
            final_signals = utils.calculate_with_sizing_numba(final_signals, self.close_np, self.risk_object.percent_to_size)

        self.osc1_data = ('MACD', macd, macd_signal_line)
        self.osc2_data = ('ATR', atr)
        self.signals = final_signals

        return final_signals

    def calculate_macd(self, close, fast, slow, signal):
        macd, macd_signal_line, macd_hist = ta.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, macd_signal_line, macd_hist

    def calculate_atr(self, high, low, close, period):
        return ta.ATR(high, low, close, timeperiod=period)

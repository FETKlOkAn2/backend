import talib as ta
from core.strategies.strategy import Strategy
import pandas as pd

class BollingerBands(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=False):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)

    def custom_indicator(self, close=None, time_period=20, nbdevup=2, nbdevdn=2):

        self.signal_period = time_period

        upperband, middleband, lowerband = self.calculate_bollinger_bands(time_period, nbdevup, nbdevdn)
        self.osc1_data = ('BollingerBands', upperband, middleband, lowerband)

        buy_signal = self.close < lowerband
        sell_signal = self.close > upperband

        self.signals = self.generate_signals(buy_signal, sell_signal)

        return self.signals

    def calculate_bollinger_bands(self, time_period, nbdevup, nbdevdn):
        upperband, middleband, lowerband = ta.BBANDS(self.close, time_period, nbdevup, nbdevdn)
        return upperband, middleband, lowerband

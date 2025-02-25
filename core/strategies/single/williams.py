import talib as ta
from core.strategies.strategy import Strategy
import pandas as pd

class WilliamsR(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=False):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)

    def custom_indicator(self, close=None, time_period=14):
        self.signal_period = time_period

        willr = self.calculate_williams_r(time_period)
        self.osc1_data = ('WilliamsR', willr)

        buy_signal = willr < -80
        sell_signal = willr > -20

        self.signals = self.generate_signals(buy_signal, sell_signal, with_formating=False)

        return self.signals

    def calculate_williams_r(self, time_period):
        willr = ta.WILLR(self.high, self.low, self.close, time_period)
        return willr

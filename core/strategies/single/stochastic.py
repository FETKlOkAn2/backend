import talib as ta
from core.strategies.strategy import Strategy
import pandas as pd

class StochasticOscillator(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=False):
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)

    def custom_indicator(self, close=None, fastk_period=14, slowk_period=3, slowd_period=3):
        self.signal_period = fastk_period

        slowk, slowd = self.calculate_stochastic(fastk_period, slowk_period, slowd_period)
        self.osc1_data = ('StochasticOscillator', slowk, slowd)

        buy_signal = slowk < 20
        sell_signal = slowk > 80

        self.signals = self.generate_signals(buy_signal, sell_signal, with_formating=False)

        return self.signals

    def calculate_stochastic(self, fastk_period, slowk_period, slowd_period):
        slowk, slowd = ta.STOCH(self.high, self.low, self.close, fastk_period, slowk_period, 0, slowd_period, 0)
        return slowk, slowd

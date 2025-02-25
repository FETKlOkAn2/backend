import talib
from core.strategies.strategy import Strategy

class ATR(Strategy):
    def __init__(self, dict_df, risk_object=None, with_sizing=False):
        super().__init__(dict_df=dict_df,risk_object=risk_object, with_sizing=with_sizing)

    def custom_indicator(self, close=None, atr_window=None):
        self.atr_window = atr_window

        atr = self.calculate_atr(atr_window)
        self.osc1_data = ('ATR', atr)
        return atr

    def calculate_atr(self, atr_window):
        atr = talib.ATR(self.high, self.low, self.close, timeperiod=atr_window)
        return atr

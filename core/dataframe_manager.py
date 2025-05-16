import pandas as pd
import datetime as dt
import core.utils.utils as utils


class DF_Manager():
    """this only needs to be instansiated during WSClient"""
    def __init__(self,scanner: object, data=None):
        self.scanner = scanner
        self.coinbase = self.scanner.coinbase
        self.client = self.scanner.client
        self.granularity = self.scanner.granularity
        self.products_to_trade = scanner.kraken_crypto
        self.products_granularity = {symbol: None for symbol in scanner.kraken_crypto}
        self.next_update_time = {symbol: None for symbol in scanner.kraken_crypto}
        if not data:
            self.dict_df = {}
            #self.data_for_live_trade()
        else:
            self.dict_df = data
            #self.update(self.df)

        self.time_map = {
            'ONE_MINUTE': pd.Timedelta(minutes=1),
            'FIVE_MINUTE': pd.Timedelta(minutes=5),
            'FIFTEEN_MINUTE': pd.Timedelta(minutes=15),
            'THIRTY_MINUTE': pd.Timedelta(minutes=30),
            'ONE_HOUR': pd.Timedelta(hours=1),
            'TWO_HOUR': pd.Timedelta(hours=2),
            'SIX_HOUR': pd.Timedelta(hours=6),
            'ONE_DAY': pd.Timedelta(days=1)
        }

    def add_to_manager(self, data):
        if not self.dict_df:
            self.dict_df = data
        else:
            for k, v in data.items():
                self.dict_df[k] = v


    def data_for_live_trade(self, symbol, update=False, resampling=None):
        """dataframe needs to be indexed by symbol"""
        
        coinbase_symbol = utils.convert_symbols(lone_symbol=symbol)
        granularity = self.products_granularity[symbol]
        print("granularity for livetrading: ", granularity)
        
        # If resampling is provided, use it
        if resampling:
            granularity = utils.convert_resampling_to_granularity(resampling)
        
        timestamps = self.coinbase._get_unix_times(
            granularity=granularity,
            days=1
        )
        
        new_dict_df = self.coinbase.get_basic_candles(
            symbols=[coinbase_symbol],
            timestamps=timestamps,
            granularity=granularity
        )
        new_dict_df[symbol] = new_dict_df.pop(coinbase_symbol)
        
        # If updating, add only the last row if symbol exists
        if update:
            self.dict_df[symbol] = pd.concat([self.dict_df[symbol], new_dict_df[symbol]]).drop_duplicates()
        else:
            self.dict_df[symbol] = new_dict_df
            
        return self.dict_df[symbol]  # Return the updated dataframe
    def set_next_update(self, symbol, initial=False):
        if initial:
            next_update_in = dt.datetime.now() - pd.Timedelta(seconds=20)
            self.next_update_time[symbol] = next_update_in
        else:
            next_update_in = dt.datetime.now() + (self.time_map[self.products_granularity[symbol]] - pd.Timedelta(seconds=20))
            self.next_update_time[symbol] = next_update_in

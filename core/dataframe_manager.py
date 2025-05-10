import pandas as pd
import datetime as dt
import backend.core.utils.utils as utils


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



    
    # def _to_df(self, dict:dict):
    #     key = next(iter(dict)) # gets the first key to the dictionary

    #     if key == 'candles': #not heartbeats only candles
    #         df = pd.DataFrame(dict[key]) # converts to df
    #         df['date'] = pd.to_datetime(df['start'].astype(float), unit='s')
    #         try:
    #             """this is for WSclient"""
    #             df = df.drop(columns=['start', 'product_id'])
    #         except:
    #             """this is for RESTClient"""
    #             df = df.drop(column=['start'])
    #         df.loc[:, df.columns != 'date'] = df.loc[:, df.columns != 'date'].apply(pd.to_numeric, errors='coerce')

    #         #df = df.ffill()
    #         df = df.infer_objects(copy=False)#forward fill NANs
    #     return df

    # def process_message(self, message, pickle = False):
    #     """"""
    #     msg_data = json.loads(message) # loads data into json format
    #     if msg_data.get('channel') == 'candles':
    #         candles_data = msg_data.get('events', [])[0].get('candles', [])

    #     #print(candles_data)
    #         if candles_data: # if not an empty list
    #             if pickle:
    #                 to_pickle(candles_data)      
    #             candles_data = {'candles': candles_data} #rename to work with _to_df
    #             self.new_df = self._to_df(candles_data)
    #             df = self.dict_df.value()
    #             df = pd.concat([df, self.new_df], ignore_index = True)
    #             self.update_df()
    #             #print(self.df.tail())

    #     if pickle:
    #         if msg_data.get('channel') == 'user':
    #             events = msg_data.get('events', [])
    #             if events: 
    #                 order_data = events[0].get('orders', [])
    #                 if order_data:  
    #                     order_info = order_data[0]
    #             list_of_keys = ['order_id',
    #                             'order_side',
    #                             'order_type']
                
    #             for key in list_of_keys:
    #                 if key in order_info:  
    #                     pickle_data = order_info[key]
    #                     to_pickle(pickle_data)

    #         #print(order_data)
        


    
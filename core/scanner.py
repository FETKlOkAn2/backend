import pandas as pd
from core.coinbase_wrapper import Coinbase_Wrapper
import core.database_interaction as database_interaction

class Scanner():
    def __init__(self, client, socketio=None):
        self.coinbase = Coinbase_Wrapper()
        self.client = client
        self.granularity = self.client.granularity
        
        self.socketio = socketio  # Store socketio
        self.coinbase = Coinbase_Wrapper(socketio=self.socketio)

        self.products = self.client.all_products
        self.df_manager = None
        self.coinbase_crypto = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD']
        self.robinhood_crypto = ['BTC', 'ETH', 'DOGE', 'SHIB', 'AVAX', 'BCH', 'LINK', 'UNI', 'LTC', 'XLM', 'ETC', 'AAVE', 'XTZ', 'COMP']
        self.kraken_crypto = ['XXBTZUSD', 'XETHZUSD', 'XDGUSD', 'SHIBUSD', 'AVAXUSD', 'BCHUSD', 'LINKUSD', 'UNIUSD', 'XLTCZUSD', 'XXLMZUSD', 'XETCZUSD', 'AAVEUSD', 'XTZUSD', 'COMPUSD']


    def assign_attribute(self, df_manager):
        self.df_manager = df_manager

    def populate_manager(self, granularity, days_ago=None):
        """currently only using robinhood symbols"""

        print('Populating DF Manager')
        for symbol in self.coinbase_crypto:
            dict_df = database_interaction.get_historical_from_db(symbol, days_ago=days_ago)
            self.df_manager.add_to_manager(dict_df)
            
            

    def filter_products(self, symbol=None):
        print('Starting filter')    
        self.products_to_trade = symbol
        self.df_manager.products_to_trade = symbol
        if symbol is None:
            self.products_to_trade = self.kraken_crypto
            self.df_manager.products_to_trade = self.kraken_crypto
        """iterate over all available symbols and only go with volume that is above the average"""
        """iterate overal all availabel symbols and filter by only market cap greater than a value"""

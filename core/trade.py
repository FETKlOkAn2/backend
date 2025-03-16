import core.pickling
import datetime
import math
import time
import core.database_interaction as database_interaction

class Trade():
    """This class will have all logic for executing trades"""

    def __init__(self, risk, strat_object, logbook, socketio=None, signals=None):
        self.risk = risk
        self.strat = strat_object
        self.logbook = logbook
        self.client = self.risk.client
        self.symbol = self.strat.symbol
        self.current_asset_price = float(self.strat.close.iloc[-1])
        self.volume_to_risk = self.get_balance_to_risk()
        self.total_volume_owned = self.client.get_extended_balance(self.symbol)
        self.socketio = socketio  # Store socketio instance

        self.signals = self.strat.signals     
        if signals:
            self.signals = signals

        if self.signals[-1] == 1:
            account_balance = self.client.get_account_balance()
            at_risk = self.volume_to_risk
            if account_balance <= at_risk:
                print('*****NO MORE MULA*********')
                print(f"Cash Balance: {account_balance}\nAt Risk: {at_risk}")
                if self.socketio:
                    self.socketio.emit('log_update', {
                        "message": f"*****NO MORE MULA********* Cash Balance: {account_balance} At Risk: {at_risk}",
                        "type": "error",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            else:
                self.buy()

        elif self.signals[-1] == -1:
            self.sell()

        else:
            self.monitor_trade()

    def buy(self):
        print('BUY')
        if self.socketio:
            self.socketio.emit('log_update', {
                "message": f"BUY {self.symbol} at price {self.current_asset_price:.2f}",
                "type": "buy",
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        try:
            # Create buy order
            buy_order = self.client.add_order(
                type_of_order='buy',
                symbol=self.symbol,
                volume=self.risk.volume_to_risk * 2,
                price=self.current_asset_price
            )
            
            if self.socketio:
                self.socketio.emit('log_update', {
                    "message": f"Created buy order for {self.symbol}",
                    "type": "buy",
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
            time.sleep(1)

            # Edit open orders until filled
            while self.client.any_open_orders():
                try:
                    if buy_order and 'result' in buy_order and 'txid' in buy_order['result']:
                        order_id = buy_order['result']['txid'][0]
                        current_price = self.client.get_recent_spreads(
                            symbol=self.symbol,
                            type_of_order='buy'
                        )
                        
                        if self.socketio:
                            self.socketio.emit('log_update', {
                                "message": f"Editing buy order {order_id} with new price {current_price}",
                                "type": "buy",
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                            
                        buy_order = self.client.edit_order(
                            order_id=order_id,
                            symbol=self.symbol,
                            volume=self.risk.volume_to_risk,
                            price=current_price
                        )
                    else:
                        print("Invalid buy_order structure or no txid:", buy_order)
                        if self.socketio:
                            self.socketio.emit('log_update', {
                                "message": f"Invalid buy_order structure or no txid: {buy_order}",
                                "type": "error",
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                        break
                except Exception as e:
                    print(f"Error while editing order: {e}")
                    if self.socketio:
                        self.socketio.emit('log_update', {
                            "message": f"Error while editing order: {e}",
                            "type": "error",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    break
                time.sleep(0.25)

            # Export trade details to the database
            if buy_order:
                try:
                    database_interaction.trade_export(
                        buy_order,
                        balance=self.client.get_account_balance()
                    )
                    
                    if self.socketio:
                        balance = self.client.get_account_balance()
                        self.socketio.emit('log_update', {
                            "message": f"Buy order completed for {self.symbol}. New balance: {balance}",
                            "type": "buy",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    print(f"Error while exporting trade: {e}")
                    if self.socketio:
                        self.socketio.emit('log_update', {
                            "message": f"Error while exporting trade: {e}",
                            "type": "error",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
            else:
                print("Buy order is None, skipping export.")
                if self.socketio:
                    self.socketio.emit('log_update', {
                        "message": "Buy order is None, skipping export.",
                        "type": "error",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error in buy method: {e}")
            if self.socketio:
                self.socketio.emit('log_update', {
                    "message": f"Error in buy method: {e}",
                    "type": "error",
                    "timestamp": datetime.datetime.now().isoformat()
                })

    def sell(self):
        print('SELL')
        if self.socketio:
            self.socketio.emit('log_update', {
                "message": f"SELL {self.symbol} at price {self.current_asset_price:.2f}",
                "type": "sell",
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        try:
            sell_order = self.client.add_order(
                type_of_order='sell',
                symbol=self.symbol,
                volume=self.total_volume_owned,
                price=self.current_asset_price,
                pickle=True
            )
            
            if self.socketio:
                self.socketio.emit('log_update', {
                    "message": f"Created sell order for {self.symbol}",
                    "type": "sell",
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
            time.sleep(1)

            while self.client.any_open_orders():
                try:
                    order_id = sell_order['result']['txid'][0]
                    current_price = self.client.get_recent_spreads(
                        symbol=self.symbol,
                        type_of_order='sell'
                    )
                    
                    if self.socketio:
                        self.socketio.emit('log_update', {
                            "message": f"Editing sell order {order_id} with new price {current_price}",
                            "type": "sell",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                    sell_order = self.client.edit_order(
                        order_id=order_id,
                        symbol=self.symbol,
                        volume=self.risk.volume_to_risk,
                        price=current_price         
                    )
                except Exception as e:
                    if self.socketio:
                        self.socketio.emit('log_update', {
                            "message": f"Error editing sell order: {e}",
                            "type": "error",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    break
                    
                time.sleep(.25)
                
            database_interaction.trade_export(sell_order, balance=self.client.get_account_balance())
            
            if self.socketio:
                balance = self.client.get_account_balance()
                self.socketio.emit('log_update', {
                    "message": f"Sell order completed for {self.symbol}. New balance: {balance}",
                    "type": "sell",
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
        except Exception as e:
            print(f"Error in sell method: {e}")
            if self.socketio:
                self.socketio.emit('log_update', {
                    "message": f"Error in sell method: {e}",
                    "type": "error",
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
    def monitor_trade(self):
        print('monitoring')
        if self.socketio:
            self.socketio.emit('log_update', {
                "message": f"Monitoring {self.symbol} at price {self.current_asset_price:.2f}",
                "type": "monitor",
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        # Your existing monitoring logic here
        pass

    def get_balance_to_risk(self):
        """Calculates balance to risk based off of backtests"""
        minimum_volume = {
            'XXBTZUSD': 0.00005,
            'XETHZUSD': 0.002,
            'XDGUSD': 30,
            'SHIBUSD': 200000,
            'AVAXUSD': .1,
            'BCHUSD': .01,
            'LINKUSD': .2,
            'UNIUSD': .5,
            'XLTCZUSD': .05,
            'XXLMZUSD': 40,
            'XETCZUSD': .3,
            'AAVEUSD': .03,
            'XTZUSD': 4,
            'COMPUSD': .1
        }
        minimum = minimum_volume[self.symbol]
        self.risk.volume_to_risk = minimum
        return minimum
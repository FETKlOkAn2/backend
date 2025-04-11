import vectorbt as vbt
import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
import sys
import talib as ta
import cupy as cp
from numba import njit
import core.utils as utils
# import logging 
# logging.basicConfig(level=logging.DEBUG)

class Strategy:
    """Class to store strategy resources"""
    def __init__(self, dict_df, risk_object=None, with_sizing=None):
        self.granularity = None
        if dict_df is not None:
            if not isinstance(dict_df, dict):
                print('You have passed a Dataframe. This Class needs to be dictionary with key as symbol and value as DataFrame')
                sys.exit(1)
            for key, value in dict_df.items():
                self.symbol = key
                self.df = value
            self.with_sizing = with_sizing

            self.open = self.df['open']
            self.high = self.df['high']
            self.low = self.df['low']
            self.close = self.df['close']
            self.volume = self.df['volume']

            self.close_np = np.array(self.close)
            self.high_np = np.array(self.high)
            self.low_np = np.array(self.low)
            self.open_np = np.array(self.open)
            self.volume_np = np.array(self.volume)            


            if self.__class__.__name__.split('_')[-1] == 'GPU':
                self.close_gpu = cp.array(self.close)
                self.high_gpu = cp.array(self.high)
                self.low_gpu = cp.array(self.low)
                self.open_gpu = cp.array(self.open)
                self.volume_gpu = cp.array(self.volume)

            self.granularity = self.set_granularity()
        self.risk_object = risk_object


        self.entries = None
        self.exits = None

        # technical indicator data assigned in custom_indicator
        #
        self.ti1_data = None
        self.ti2_data = None
        self.ti3_data = None
        self.ti4_data = None

        # oscilator data assigned in custom_indicator
        self.osc1_data = None
        self.osc2_data = None
        self.osc3_data = None
        self.osc4_data = None

        self.buy_threshold = None
        self.sell_threshold = None

        self.portfolio = None


    def custom_indicator(self, close,  fast_window=5, slow_window=30):
        self.fast_window = fast_window
        self.slow_window = slow_window

        fast_ma = vbt.MA.run(close, fast_window)
        slow_ma = vbt.MA.run(close, slow_window)
        
        self.ti1_data = ("Fast MA", fast_ma.ma)
        self.ti2_data = ("Slow MA", slow_ma.ma)

        self.entries = fast_ma.ma_crossed_above(slow_ma)
        self.exits = fast_ma.ma_crossed_below(slow_ma)

        signals =np.zeros_like(close)
        signals[self.entries] = 1
        signals[self.exits] = -1
        self.signals = signals

        return self.signals

    def update(self, dict_df):
        for key, value in dict_df.items():
            self.symbol = key
            self.df = value

        self.open = self.df['open']
        self.high = self.df['high']
        self.low = self.df['low']
        self.close = self.df['close']
        self.volume = self.df['volume']

        self.close_gpu = cp.array(self.close)
        self.high_gpu = cp.array(self.high)
        self.low_gpu = cp.array(self.low)
        self.open_gpu = cp.array(self.open)
        self.volume_gpu = cp.array(self.volume)
        
        self.granularity = self.set_granularity()

    def generate_signals(self, buy_signal, sell_signal, with_formating=True):
        """Common method to generate and format buy/sell signals"""
        signals = np.zeros_like(self.close, dtype=int)
        signals[buy_signal] = 1
        signals[sell_signal] = -1

        if with_formating:
            signals = np.array(signals)
            signals = utils.format_signals(signals)
            # if self.with_sizing:
            #     signals = self.calculate_with_sizing(signals)##########################################################################

        # For graphing
        self.entries = np.zeros_like(signals, dtype=bool)
        self.exits = np.zeros_like(signals, dtype=bool)

        self.entries[signals == 1] = True
        self.exits[signals == -1] = True

        self.entries = pd.Series(self.entries, index=self.close.index)
        self.exits = pd.Series(self.exits, index=self.close.index)
        return signals
      
    
    # def format_signals(self, signals):
    #     """formats signals so no double buy or double sells"""
    #     formatted_signals = np.zeros_like(signals)
    #     in_position = False
            
    #     for i in range(len(signals)):
    #         if signals[i] == 1 and not in_position:
    #             formatted_signals[i] = 1
    #             in_position = True
    #         elif signals[i] == -1 and in_position:
    #             formatted_signals[i] = -1
    #             in_position = False
    #     return formatted_signals    


    def combine_signals(self, *signals):
        
        signals_array = np.array(signals)  

        all_ones = np.all(signals_array == 1, axis=0)

        all_neg_ones = np.all(signals_array == -1, axis=0)
        
        combined_signals = np.zeros(signals_array.shape[1], dtype=int)
        
        combined_signals[all_ones] = 1
        
        combined_signals[all_neg_ones] = -1
        
        combined_signals = utils.format_signals(combined_signals)
        if self.with_sizing:
            percent_to_size = self.risk_object.percent_to_size
            close_array = self.close.to_numpy(dtype=np.float64)
            signal_array = np.array(combined_signals)
            combined_signals = utils.calculate_with_sizing_numba(signal_array, close_array, percent_to_size)

        self.entries = np.zeros_like(self.close, dtype=bool) 
        self.exits = np.zeros_like(self.close, dtype=bool)

        self.entries[combined_signals == 1] = True
        self.exits[combined_signals == -1] = True

        self.entries = pd.Series(self.entries, index=self.close.index)
        self.exits = pd.Series(self.exits, index=self.close.index)

        return combined_signals
          
    def ensure_series(self, data, index):
        if isinstance(data, pd.Series):
            return data
        return pd.Series(data, index=index) if len(data) == len(index) else None


    def graph(self, graph_callback=None):
        """Graphs the strategy with technical indicators and signals"""
        try:
            # Initialize the candlestick figure
            fig1 = go.Figure(data=[go.Candlestick(
                x=self.close.index,
                open=self.open,
                high=self.high,
                low=self.low,
                close=self.close
            )])
            fig2 = None
            param_number = 0

            # Add technical indicators
            while True:
                param_number += 1
                ti_data_attr = getattr(self, f"ti{param_number}_data", None)
                if ti_data_attr:
                    if isinstance(ti_data_attr, tuple) and len(ti_data_attr) >= 2:
                        ti_data_name, *ti_data_list = ti_data_attr
                        for i, ti_data in enumerate(ti_data_list):
                            ti_data_series = self.ensure_series(ti_data, self.close.index)
                            if ti_data_series is not None:
                                fig1 = ti_data_series.vbt.plot(
                                    trace_kwargs=dict(name=f"{ti_data_name}_{i+1}"), fig=fig1
                                )
                            else:
                                print(f"Skipping {ti_data_name}_{i+1}: incompatible data.")

                osc_data_attr = getattr(self, f"osc{param_number}_data", None)
                if osc_data_attr:
                    if isinstance(osc_data_attr, tuple) and len(osc_data_attr) >= 2:
                        osc_data_name, *osc_data_list = osc_data_attr
                        for i, osc_data in enumerate(osc_data_list):
                            osc_data_series = self.ensure_series(osc_data, self.close.index)
                            if osc_data_series is not None:
                                if fig2 is None:
                                    fig2 = osc_data_series.vbt.plot(trace_kwargs=dict(name=f"{osc_data_name}_{i+1}"))
                                else:
                                    fig2 = osc_data_series.vbt.plot(trace_kwargs=dict(name=f"{osc_data_name}_{i+1}"), fig=fig2)

                if not ti_data_attr and not osc_data_attr:
                    break

            # Add entry/exit markers
            if self.entries is not None:
                self.entries = self.ensure_series(self.entries, self.close.index)
                if self.entries is not None:
                    fig1 = self.entries.vbt.signals.plot_as_entry_markers(self.close, fig=fig1)
                else:
                    print("Entries are incompatible with graphing.")

            if self.exits is not None:
                self.exits = self.ensure_series(self.exits, self.close.index)
                if self.exits is not None:
                    fig1 = self.exits.vbt.signals.plot_as_exit_markers(self.close, fig=fig1)
                else:
                    print("Exits are incompatible with graphing.")

            # Combine subplots
            rows = 1 + (1 if fig2 else 0)
            fig_combined = sp.make_subplots(rows=rows, cols=1)
            
            for trace in fig1['data']:
                fig_combined.add_trace(trace, row=1, col=1)
            if fig2:
                for trace in fig2['data']:
                    fig_combined.add_trace(trace, row=2, col=1)
                if self.buy_threshold is not None:
                    fig_combined.add_hline(y=self.buy_threshold, line_color='green', row=2, col=1)
                if self.sell_threshold is not None:
                    fig_combined.add_hline(y=self.sell_threshold, line_color='red', row=2, col=1)

            fig_combined.update_layout(
                height=800,
                title_text=f"{self.__class__.__name__} strategy for {self.symbol} on {self.granularity}",
                xaxis_rangeslider_visible=False
            )

            # Final check for combined figure
            if not fig_combined or len(fig_combined.data) == 0:
                print("Error: Combined figure is empty.")
                return None

            # Mark that we're handling this graph
            setattr(fig_combined, '_sent_to_client', False)

            # Display or call callback
            if graph_callback:
                print("Graph callback is being called with a figure.")
                result = graph_callback(fig_combined)
                # Mark if the callback successfully handled the graph
                if result:
                    setattr(fig_combined, '_sent_to_client', True)
            else:
                print("Displaying the figure directly.")
                fig_combined.show()
                
            # Return the figure
            return fig_combined

        except Exception as e:
            print(f"Error while graphing: {e}")
            import traceback
            traceback.print_exc()
            return None



    def generate_backtest(self):
        # logging.debug("Generating backtest")
        """Performs backtest and returns the stats"""
        init_cash = self.risk_object.total_balance
        size = None
        size_type = None
        accumulate = False

        if self.with_sizing:
            size = pd.Series(index=self.close.index, dtype='float')
            size[self.entries] = self.risk_object.risk_percent #this sizing will be calculated through risk class
            size[self.exits] = np.inf

            size_type = 'value'
            accumulate = True

        # logging.debug(f"close: {self.close}")
        # logging.debug(f"entries: {self.entries}")
        # logging.debug(f"exits: {self.exits}")
        # logging.debug(f"size: {size}")
        # logging.debug(f"size_type: {size_type}")
        # logging.debug(f"accumulate: {accumulate}")
        # logging.debug(f"init_cash: {init_cash}")
        # logging.debug(f"strategy: {self.__class__.__name__}")
        self.portfolio = vbt.Portfolio.from_signals(
            close = self.close,
            entries = self.entries,
            exits = self.exits,
            size = size,
            size_type= size_type,
            accumulate= accumulate,
            init_cash= init_cash)

        return self.portfolio
    
    
    def set_granularity(self):
        # Retrieves multiple dates and compares then gets the most frequest
        time_differences = []
        for i in range(20):
            if i == 0:
                pass
            time_diff = self.df.index[i] - self.df.index[i-1]
            if time_diff < pd.Timedelta(minutes=4):
                time_diff = pd.Timedelta(minutes=1)
            time_differences.append(time_diff)
        time_diff = max(time_differences, key=time_differences.count)
        #print(time_diff)
        time_map = {
            pd.Timedelta(minutes=1): 'ONE_MINUTE',
            pd.Timedelta(minutes=5): 'FIVE_MINUTE',
            pd.Timedelta(minutes=15): 'FIFTEEN_MINUTE',
            pd.Timedelta(minutes=30): 'THIRTY_MINUTE',
            pd.Timedelta(hours=1): 'ONE_HOUR',
            pd.Timedelta(hours=2): 'TWO_HOUR',
            pd.Timedelta(hours=6): 'SIX_HOUR',
            pd.Timedelta(days=1): 'ONE_DAY'
        }
        # Return the corresponding string or 'Unknown' if not found
        return  time_map.get(time_diff, 'Unknown')


    def add_adx(self, adx_buy_threshold, time_period):
        adx = ta.ADX(self.high, self.low, self.close, time_period)
        
        #adds signal to oscilator data for graphing by finding the first None
        for i in range(4):
            osc_string = f'osc{i}_data'
            if getattr(self, osc_string, None) is None:
                setattr(self, osc_string, ('ADX', adx))

        buy_signal = adx > adx_buy_threshold
        sell_signal = ~buy_signal

        adx_signals = self.generate_signals(buy_signal, sell_signal, with_formating=False)

        return adx_signals


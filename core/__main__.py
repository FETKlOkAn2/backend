import time
import asyncio
import database.database_interaction as database_interaction
from core.dataframe_manager import DF_Manager
from core.strategies.strategy import Strategy
from core.strategies.single.rsi import RSI
from core.trade import Trade
from core.log import LinkedList
from core.scanner import Scanner
from core.risk import Risk_Handler
from core.kraken_wrapper import Kraken
from core.strategies.double.rsi_adx import RSI_ADX
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
import datetime as dt

granularity = 'ONE_MINUTE'
symbol = 'XBTUSD'
counter = 0


def on_message():
    print('--------------------------------------------------------------------------------------\n')
    global counter
    global kraken
    global risk
    global scanner
    global df_manager
    print(f'counter: {counter}')
    
    for k in df_manager.dict_df.keys():

        if dt.datetime.now() <= df_manager.next_update_time[k]:
            continue
        print(k)
        df_manager.data_for_live_trade(symbol=k, update=True)
        current_dict = {k: df_manager.dict_df[k]}

        strat = RSI_ADX_GPU(current_dict, risk, with_sizing=True, hyper=False)

        strat.custom_indicator(strat.close, *risk.symbol_params[k])

        trade = Trade(risk = risk,
                    strat_object=strat,
                    logbook=logbook)
        
        df_manager.set_next_update(k)
        print(df_manager.next_update_time[k])
        print('\n-------------------------------------------------------------------------------------\n')
        time.sleep(.5)

    counter += 1


async def fetch_data_periodically():
    while True:
        start_time = time.time()

        on_message()

        execution_time = time.time() - start_time
        sleep_time = max(0, kraken.time_to_wait - execution_time)

        print(f"Execution time: {execution_time:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.\n")

        await asyncio.sleep(sleep_time)


"""---------------start of program-----------------"""
kraken = Kraken()
risk = Risk_Handler(kraken)
scanner = Scanner(client=kraken)
df_manager = DF_Manager(scanner)
scanner.assign_attribute(df_manager=df_manager)

scanner.coinbase.get_candles_for_db(scanner.coinbase_crypto, kraken.granularity, days=30)


for symbol in scanner.kraken_crypto:

    strat = RSI_ADX(dict_df= None, risk_object=risk)
    strat.symbol = symbol
    params = database_interaction.get_best_params(strat, df_manager,live_trading=True, best_of_all_granularities=True, minimum_trades=3)
    risk.symbol_params[symbol] = params
    df_manager.set_next_update(symbol, initial=True)


logbook = LinkedList()


async def main():
    await fetch_data_periodically()

if __name__ == "__main__":
    asyncio.run(main())

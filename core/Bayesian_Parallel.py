import multiprocessing
import optuna
from core.backtest import Backtest
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
import database.database_interaction as database_interaction
from core.strategies.gpu_optimized.GPU.bollinger_vwap_gpu import BollingerBands_VWAP_GPU
import sqlite3 as sql
from dotenv import load_dotenv
import os
import time

backtest_instance = Backtest()
load_dotenv()

db_path = os.getenv('DATABASE_PATH')
print("DATABASE_PATH : ", db_path)
def optimize_worker(strategy_class, symbol, granularity, db_path, n_trials):
    study_name = f"{strategy_class.__name__}_{symbol}_{granularity}"
    trial_durations = []

    def objective(trial):
        start_time = time.time()
        params = {
            'bb_period': trial.suggest_int('bb_period', 10, 50),
            'bb_dev': trial.suggest_float('bb_dev', 1.0, 3.0, step=0.1),
            'vwap_window': trial.suggest_int('vwap_window', 5, 50),  
        }


        stats = backtest_instance.run_optuna_backtest(
            symbol=symbol,
            granularity=granularity,
            strategy_obj=strategy_class,
            num_days=365,
            sizing=True,
            params=params
        )

        
        end_time = time.time() - start_time
        trial_durations.append(end_time)
        return stats.get('Total Return [%]', -1)

    try:
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f"sqlite:///{db_path}/hyper_optuna.db"
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}/hyper_optuna.db"
        )

    study.optimize(objective, n_trials=n_trials)

    if trial_durations:
        avg_trial_time = sum(trial_durations) / len(trial_durations)
        total_estimated_time = avg_trial_time * n_trials
        print(f"Estimated time for '{study_name}': {total_estimated_time:.2f} seconds.")
    else:
        print(f"No trials completed for '{study_name}'.")

# Multiprocessing
if __name__ == "__main__":
    symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD'] #
    granularity = ['ONE_MINUTE','FIVE_MINUTE','FIFTEEN_MINUTE','THIRTY_MINUTE','ONE_HOUR','TWO_HOUR','SIX_HOUR','ONE_DAY']

    n_trials = 100
    num_workers = 10  

    pool = multiprocessing.Pool(processes=num_workers)
    tasks = [
        (BollingerBands_VWAP_GPU, symbol, gran, db_path, n_trials)
        for symbol in symbols
        for gran in granularity
    ]
    pool.starmap(optimize_worker, tasks)
    pool.close()
    pool.join()

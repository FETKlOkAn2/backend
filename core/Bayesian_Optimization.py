import optuna
from core.backtest import Backtest
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
import core.database_interaction as database_interaction
import sqlite3 as sql
from dotenv import load_dotenv
import os

load_dotenv()

db_path = os.getenv('DATABASE_PATH')
print("DATABASE_PATH : ", db_path)  # Ensure the correct database path is printed


class BacktestWithBayesian(Backtest):
    def run_bayesian_optimization(self, strategy_class, n_trials=100):
        for symbol in self.symbols:
            for granularity in self.granularities:
                study_name = f"{strategy_class.__name__}_{symbol}_{granularity}"
                def objective(trial):
                    # Define parameter ranges
                    params = {
                        'rsi_window': trial.suggest_int('rsi_window', 5, 50),
                        'buy_threshold': trial.suggest_int('buy_threshold', 10, 30),
                        'sell_threshold': trial.suggest_int('sell_threshold', 70, 90),
                        'adx_time_period': trial.suggest_int('adx_time_period', 10, 50),
                        'adx_buy_threshold': trial.suggest_int('adx_buy_threshold', 20, 50),
                    }

                    
                    # Run Optuna backtest for the suggested parameters
                    stats = self.run_optuna_backtest(
                        symbol=symbol,
                        granularity=granularity,
                        strategy_obj=strategy_class,
                        num_days=365,
                        sizing=True,
                        params=params
                    )

                    # Objective: Maximize total return
                    return stats.get('Total Return [%]', -1)  # Default to -1 if no stats generated

                # Run the optimization
                try:
                    study = optuna.create_study(
                        direction='maximize', 
                        study_name=study_name, 
                        storage=f"sqlite:///{db_path}/hyper_optuna.db"
                    )
                    print(f"Study '{study_name}' created successfully.")
                except optuna.exceptions.DuplicatedStudyError:
                    print(f"Study '{study_name}' already exists. Attempting to load it...")
                    # Load the existing study if it already exists
                    study = optuna.load_study(
                        study_name=study_name, 
                        storage=f"sqlite:///{db_path}/hyper_optuna.db"
                    )

                # Ensure the study variable is defined
                if not study:
                    raise RuntimeError("Failed to create or load the study.")

                study.optimize(objective, n_trials=n_trials)

                # Export the results
                database_interaction.export_optimization_results_to_db(study, strategy_class)


backtest_instance = BacktestWithBayesian()
backtest_instance.run_bayesian_optimization(RSI_ADX_GPU, n_trials=200)

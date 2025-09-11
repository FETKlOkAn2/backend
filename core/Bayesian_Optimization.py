import optuna
from optuna.samplers import TPESampler
import numpy as np
import gc
import logging
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import pandas as pd
from core.backtest import Backtest
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
import database.database_interaction as database_interaction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bayesian_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
db_path = os.getenv('DATABASE_PATH')
logger.info(f"Using database path: {db_path}")


class BacktestWithBayesian(Backtest):
    """
    Enhanced Bayesian optimization framework for trading strategy optimization
    with integrated risk management and improved resource handling.
    """
    
    def __init__(self, max_workers=3, checkpoint_interval=10):
        """
        Initialize the Bayesian optimization backtest framework
        
        Args:
            max_workers: Maximum number of parallel optimization tasks
            checkpoint_interval: Number of trials between saving intermediate results
        """
        super().__init__()
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        self.optimization_history = {}
        
    def get_parameter_ranges(self, strategy_class):
        """
        Define parameter ranges based on strategy type.
        This allows for strategy-specific customization.
        
        Args:
            strategy_class: The strategy class to get parameter ranges for
            
        Returns:
            dict: Parameter ranges for the strategy
        """
        # Default parameter ranges for RSI_ADX strategy
        if strategy_class.__name__ == 'RSI_ADX_GPU':
            return {
                'rsi_window': (5, 50),
                'buy_threshold': (10, 30),
                'sell_threshold': (70, 90),
                'adx_time_period': (10, 50),
                'adx_buy_threshold': (20, 50),
            }
        # Add more strategy-specific parameter ranges as needed
        return {}
    
    def suggest_params(self, trial, param_ranges):
        """
        Suggest parameters for a trial based on defined ranges
        
        Args:
            trial: Optuna trial object
            param_ranges: Dictionary of parameter ranges
            
        Returns:
            dict: Parameters for this trial
        """
        params = {}
        for param_name, (low, high) in param_ranges.items():
            if 'window' in param_name or 'period' in param_name or 'threshold' in param_name:
                # Integer parameters
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                # Float parameters
                params[param_name] = trial.suggest_float(param_name, low, high)
        return params
    
    def objective_function(self, trial, symbol, granularity, strategy_class, num_days, sizing, risk_params):
        """
        Objective function for Optuna optimization with enhanced risk management
        
        Args:
            trial: Optuna trial object
            symbol: Trading symbol
            granularity: Time granularity
            strategy_class: Strategy class to optimize
            num_days: Number of days for backtest
            sizing: Whether to use position sizing
            risk_params: Risk management parameters
            
        Returns:
            float: Optimization metric (could be Sharpe ratio, total return, etc.)
        """
        # Get parameter ranges for this strategy
        param_ranges = self.get_parameter_ranges(strategy_class)
        if not param_ranges:
            logger.error(f"No parameter ranges defined for {strategy_class.__name__}")
            return -1000  # Return a very bad score to indicate failure
        
        # Get parameters for this trial
        params = self.suggest_params(trial, param_ranges)
        
        # Log trial information
        logger.info(f"Trial {trial.number}: Testing parameters {params} for {symbol} ({granularity})")
        
        try:
            # Run backtest with these parameters
            stats = self.run_optuna_backtest(
                symbol=symbol,
                granularity=granularity,
                strategy_obj=strategy_class,
                num_days=num_days,
                sizing=sizing,
                params=params,
                risk_params=risk_params
            )
            
            # Handle the case when backtest returns no stats
            if not stats or 'Total Return [%]' not in stats:
                logger.warning(f"No valid stats returned for trial {trial.number}")
                return -100  # Bad but not critical score
            
            # Calculate composite score based on multiple metrics
            # This can be customized based on optimization goals
            total_return = stats.get('Total Return [%]', 0)
            sharpe_ratio = stats.get('Sharpe Ratio', 0)
            max_drawdown = abs(stats.get('Max. Drawdown [%]', 100))
            win_rate = stats.get('Win Rate [%]', 0)
            
            # Calculate score (example formula)
            # Prioritize return but penalize for drawdown
            score = total_return * (1 + (sharpe_ratio * 0.2)) * (win_rate / 100) * (1 - (max_drawdown / 200))
            
            # Log detailed results
            logger.info(f"Trial {trial.number} results: Return={total_return:.2f}%, "
                         f"Sharpe={sharpe_ratio:.2f}, Drawdown={max_drawdown:.2f}%, "
                         f"Win Rate={win_rate:.2f}%, Score={score:.2f}")
            
            # Save additional metrics as user attributes in the trial
            trial.set_user_attr('total_return', total_return)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('win_rate', win_rate)
            trial.set_user_attr('total_trades', stats.get('Total Trades', 0))
            
            # Memory management
            gc.collect()
            
            return score
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}", exc_info=True)
            return -1000  # Return a very bad score on error
    
    def run_optuna_backtest(self, symbol, granularity, strategy_obj, num_days, sizing, params, risk_params=None):
        """
        Run backtest with given parameters and integrated risk management
        
        Args:
            symbol: Trading symbol
            granularity: Time granularity
            strategy_obj: Strategy class
            num_days: Number of days for backtest
            sizing: Whether to use position sizing
            params: Strategy parameters
            risk_params: Risk management parameters
            
        Returns:
            dict: Backtest statistics
        """
        try:
            dict_df = database_interaction.get_historical_from_db(
                granularity=granularity,
                symbols=symbol,
                num_days=num_days
            )
            
            if not dict_df:
                logger.warning(f"No historical data found for {symbol} at {granularity}")
                return {}
            
            for key, value in dict_df.items():
                # Create Risk_Handler with custom parameters
                from core.risk import Risk_Handler
                risk = Risk_Handler()
                
                # Apply custom risk parameters if provided
                if risk_params:
                    risk.risk_percent = risk_params.get('risk_percent', risk.risk_percent)
                    risk.max_drawdown = risk_params.get('max_drawdown', risk.max_drawdown)
                    risk.max_open_trades = risk_params.get('max_open_trades', risk.max_open_trades)
                
                # Initialize strategy with the data and risk handler
                strat = strategy_obj(
                    dict_df={key: value},
                    risk_object=risk,
                    with_sizing=sizing,
                )
                
                # Apply the parameters to the strategy
                strat.custom_indicator(**params)
                
                # Run backtest
                strat.generate_backtest()
                stats = strat.portfolio.stats().to_dict()
                
                # Memory management
                del strat
                gc.collect()
                
                return stats
                
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}", exc_info=True)
            return {}
    
    def optimize_symbol(self, symbol, granularity, strategy_class, num_days, n_trials, risk_params=None):
        """
        Run optimization for a single symbol
        
        Args:
            symbol: Trading symbol
            granularity: Time granularity 
            strategy_class: Strategy class
            num_days: Number of days for backtest
            n_trials: Number of optimization trials
            risk_params: Risk management parameters
            
        Returns:
            optuna.Study: Completed study
        """
        study_name = f"{strategy_class.__name__}_{symbol}_{granularity}_{datetime.now().strftime('%Y%m%d')}"
        storage_path = f"sqlite:///{db_path}/hyper_optuna.db"
        
        # Custom TPE sampler with multivariate optimization
        sampler = TPESampler(multivariate=True, seed=42)
        
        # Define pruner to stop unpromising trials early
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=1
        )
        
        try:
            # Try to create a new study or load if exists
            try:
                study = optuna.create_study(
                    direction='maximize',
                    study_name=study_name,
                    storage=storage_path,
                    sampler=sampler,
                    pruner=pruner,  # Attach pruner here, not to the sampler
                    load_if_exists=True
                )
                logger.info(f"Created study '{study_name}'")
            except Exception as e:
                # Handle the case where the study exists but tables might be duplicated
                if "already exists" in str(e):
                    # Try to load the existing study instead
                    study = optuna.load_study(
                        study_name=study_name,
                        storage=storage_path,
                        sampler=sampler,  # Need to specify sampler when loading too
                        pruner=pruner
                    )
                    logger.info(f"Loaded existing study '{study_name}' with {len(study.trials)} previous trials")
                else:
                    # Re-raise if it's a different error
                    raise
            
            start_time = time.time()
            current_trials = len(study.trials)
            
            # No need to set the sampler again with pruner
            # Just keep the sampler and pruner separate
            
            # Define callback for progress tracking and checkpointing
            def progress_callback(study, trial):
                nonlocal current_trials
                elapsed = time.time() - start_time
                current_trials += 1
                
                # Print progress
                if trial.number % 5 == 0:
                    best_trial = study.best_trial
                    logger.info(f"Trial {trial.number}/{n_trials} complete. "
                                f"Elapsed: {elapsed:.1f}s. "
                                f"Best value: {best_trial.value:.4f} "
                                f"(params: {best_trial.params})")
                
                # Export intermediate results
                if trial.number % self.checkpoint_interval == 0:
                    database_interaction.export_optimization_results_to_db(study, strategy_class)
                    logger.info(f"Checkpoint saved at trial {trial.number}")
            
            # Set up the objective function with fixed parameters
            objective = lambda trial: self.objective_function(
                trial, symbol, granularity, strategy_class, num_days, True, risk_params
            )
            
            # Run the optimization
            remaining_trials = max(0, n_trials - current_trials)
            if remaining_trials > 0:
                study.optimize(
                    objective, 
                    n_trials=remaining_trials, 
                    callbacks=[progress_callback],
                    gc_after_trial=True,
                    show_progress_bar=True
                )
            
            # Log completion
            total_time = time.time() - start_time
            logger.info(f"Optimization for {symbol} ({granularity}) completed in {total_time:.1f}s")
            logger.info(f"Best value: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")
            
            # Save final results
            database_interaction.export_optimization_results_to_db(study, strategy_class)
            
            return study
            
        except Exception as e:
            logger.error(f"Error optimizing {symbol} ({granularity}): {str(e)}", exc_info=True)
            return None
        
    def run_bayesian_optimization(self, strategy_class, n_trials=100, optimize_risk=True):
        """
        Run Bayesian optimization for multiple symbols and granularities
        
        Args:
            strategy_class: Strategy class to optimize
            n_trials: Number of trials per symbol/granularity
            optimize_risk: Whether to optimize risk parameters as well
        """
        # Record start time for overall progress tracking
        overall_start = time.time()
        total_combinations = len(self.symbols) * len(self.granularities)
        completed = 0
        
        # Default risk parameters
        risk_params = {
            'risk_percent': 0.02,
            'max_drawdown': 0.15,
            'max_open_trades': 3
        }
        
        # Optimize risk parameters if requested
        if optimize_risk:
            risk_params = self.optimize_risk_parameters(strategy_class)
            logger.info(f"Optimized risk parameters: {risk_params}")
        
        # Process each symbol/granularity combination
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for symbol in self.symbols:
                for granularity in self.granularities:
                    # Determine appropriate backtest duration based on granularity
                    num_days = self.get_optimal_backtest_days(granularity)
                    
                    # Submit task to executor
                    future = executor.submit(
                        self.optimize_symbol, 
                        symbol, 
                        granularity, 
                        strategy_class, 
                        num_days, 
                        n_trials,
                        risk_params
                    )
                    futures[(symbol, granularity)] = future
            
            # Process results as they complete
            for (symbol, granularity), future in futures.items():
                try:
                    study = future.result()
                    if study:
                        best_params = study.best_params
                        best_value = study.best_value
                        
                        # Add to results
                        results.append({
                            'symbol': symbol,
                            'granularity': granularity,
                            'best_value': best_value,
                            'best_params': best_params,
                            'n_trials': len(study.trials)
                        })
                        
                    completed += 1
                    elapsed = time.time() - overall_start
                    eta = (elapsed / completed) * (total_combinations - completed) if completed > 0 else 0
                    
                    logger.info(f"Progress: {completed}/{total_combinations} combinations complete. "
                                f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                    
                except Exception as e:
                    logger.error(f"Error processing results for {symbol} ({granularity}): {str(e)}")
        
        # Create summary report
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            logger.info("\n===== OPTIMIZATION SUMMARY =====")
            logger.info(f"Total combinations processed: {completed}/{total_combinations}")
            logger.info(f"Total time: {time.time() - overall_start:.1f}s")
            
            # Find best performing symbol/granularity
            if 'best_value' in results_df.columns:
                best_idx = results_df['best_value'].idxmax()
                best_row = results_df.iloc[best_idx]
                logger.info(f"Best overall: {best_row['symbol']} ({best_row['granularity']}) "
                            f"with score {best_row['best_value']:.4f}")
                logger.info(f"Parameters: {best_row['best_params']}")
            
            # Export final results to database
            database_interaction.export_optimization_summary_to_db(
                results_df,
                strategy_class.__name__
            )
    
    def get_optimal_backtest_days(self, granularity):
        """
        Determine optimal backtest duration based on granularity
        
        Args:
            granularity: Time granularity
            
        Returns:
            int: Number of days for backtest
        """
        # Map granularities to appropriate backtest durations
        granularity_days = {
            'ONE_MINUTE': 30,
            'FIVE_MINUTE': 90,
            'FIFTEEN_MINUTE': 180,
            'THIRTY_MINUTE': 365,
            'ONE_HOUR': 365,
            'TWO_HOUR': 730,
            'SIX_HOUR': 730,
            'ONE_DAY': 1095  # ~3 years
        }
        return granularity_days.get(granularity, 365)
    
    def optimize_risk_parameters(self, strategy_class, n_trials=50):
        """
        Optimize risk management parameters
        
        Args:
            strategy_class: Strategy class
            n_trials: Number of trials
            
        Returns:
            dict: Optimized risk parameters
        """
        logger.info("Optimizing risk parameters...")
        
        # Select a representative symbol and granularity for risk optimization
        symbol = self.symbols[0] if self.symbols else 'BTC-USD'
        granularity = 'ONE_HOUR'
        num_days = 365
        
        # Define the study
        study_name = f"Risk_Optimization_{strategy_class.__name__}_{datetime.now().strftime('%Y%m%d')}"
        storage_path = f"sqlite:///{db_path}/risk_optuna.db"
        
        try:
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_path
            )
        except optuna.exceptions.DuplicatedStudyError:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_path
            )
        
        # Define risk optimization objective function
        def risk_objective(trial):
            # Suggest risk parameters
            risk_params = {
                'risk_percent': trial.suggest_float('risk_percent', 0.005, 0.05),
                'max_drawdown': trial.suggest_float('max_drawdown', 0.05, 0.25),
                'max_open_trades': trial.suggest_int('max_open_trades', 1, 5)
            }
            
            # Get predefined strategy parameters (could be best from previous runs)
            strategy_params = self.get_best_strategy_params(strategy_class, symbol, granularity)
            
            # Run backtest with these parameters
            stats = self.run_optuna_backtest(
                symbol=symbol,
                granularity=granularity,
                strategy_obj=strategy_class,
                num_days=num_days,
                sizing=True,
                params=strategy_params,
                risk_params=risk_params
            )
            
            # Calculate score based on risk-adjusted returns
            if not stats:
                return -100
                
            total_return = stats.get('Total Return [%]', 0)
            max_drawdown = abs(stats.get('Max. Drawdown [%]', 100))
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # We want to maximize risk-adjusted returns
            return calmar_ratio
        
        # Run the optimization
        study.optimize(risk_objective, n_trials=n_trials)
        
        # Return optimized risk parameters
        return study.best_params
    
    def get_best_strategy_params(self, strategy_class, symbol, granularity):
        """
        Get best strategy parameters from database
        
        Args:
            strategy_class: Strategy class
            symbol: Trading symbol
            granularity: Time granularity
            
        Returns:
            dict: Best strategy parameters
        """
        try:
            # Try to load from database first
            params = get_best_strategy_params(self, strategy_class, symbol, granularity)
            
            if params:
                return params
                
        except Exception as e:
            logger.warning(f"Could not get best parameters from database: {str(e)}")
        
        # Return default parameters if database lookup fails
        return self.get_default_strategy_params(strategy_class)
    
    def get_default_strategy_params(self, strategy_class):
        """
        Get default strategy parameters
        
        Args:
            strategy_class: Strategy class
            
        Returns:
            dict: Default strategy parameters
        """
        if strategy_class.__name__ == 'RSI_ADX_GPU':
            return {
                'rsi_window': 14,
                'buy_threshold': 30,
                'sell_threshold': 70,
                'adx_time_period': 14,
                'adx_buy_threshold': 25,
            }
        # Add more strategy defaults as needed
        return {}
# Modified get_best_strategy_params method for the BacktestWithBayesian class
def get_best_strategy_params(self, strategy_class, symbol, granularity):
    """
    Get best strategy parameters from database
    
    Args:
        strategy_class: Strategy class
        symbol: Trading symbol
        granularity: Time granularity
        
    Returns:
        dict: Best strategy parameters
    """
    try:
        # Try to load from database using the modified function that doesn't need df
        params = database_interaction.get_best_params_without_df(
            strategy_class.__name__,
            symbol,
            granularity,
            minimum_trades=4
        )
        
        if params:
            # Convert params list to dictionary
            param_names = self.get_parameter_names(strategy_class)
            param_dict = dict(zip(param_names, params))
            logger.info(f"Found best parameters in database: {param_dict}")
            return param_dict
                
    except Exception as e:
        logger.warning(f"Could not get best parameters from database: {str(e)}")
    
    # Return default parameters if database lookup fails
    return self.get_default_strategy_params(strategy_class)

# Add this helper method to get parameter names
def get_parameter_names(self, strategy_class):
    """
    Get parameter names for a strategy class
    
    Args:
        strategy_class: Strategy class
        
    Returns:
        list: Parameter names
    """
    # First check if it's a known strategy and return hardcoded parameters
    if strategy_class.__name__ == 'RSI_ADX_GPU' or strategy_class.__name__ == 'RSI_ADX_NP':
        return ['rsi_window', 'buy_threshold', 'sell_threshold', 'adx_time_period', 'adx_buy_threshold']
    
    # For other strategies, try to inspect the function signature
    import inspect
    try:
        params = inspect.signature(strategy_class.custom_indicator)
        param_keys = list(dict(params.parameters).keys())[1:]  # Exclude 'self'
        return param_keys
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not determine parameters for {strategy_class.__name__}: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    backtest_instance = BacktestWithBayesian(max_workers=6, checkpoint_interval=10)
    
    # Optional: Use a subset of symbols for faster testing
    backtest_instance.symbols = ['BTC-USD', 'ETH-USD']
    backtest_instance.granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY']
    
    # Run the optimization
    backtest_instance.run_bayesian_optimization(
        strategy_class=RSI_ADX_GPU,
        n_trials=200,
        optimize_risk=True
    )
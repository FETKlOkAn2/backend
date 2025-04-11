import os
import sys
import logging
import time
import gc

# Add your project's core directory to sys.path
sys.path.append('/opt/python')

# Import your existing modules
import core.database_interaction as database_interaction
import core.utils as utils
from core.risk import Risk_Handler
from core.hyper import Hyper

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class HyperOptimizer:
    def __init__(self, symbols, granularities):
        self.symbols = symbols
        self.granularities = granularities

    def run_hyper(self, strategy_obj, param_ranges):
        """Runs hyperparameter optimization for a given strategy and parameter ranges."""
        
        risk = Risk_Handler()
        results = {}

        for granularity in self.granularities:
            logger.info(f"Running hyper optimization for granularity: {granularity}")
            
            days = 25 if granularity == 'ONE_MINUTE' else \
                100 if granularity == 'FIVE_MINUTE' else \
                250 if granularity == 'FIFTEEN_MINUTE' else 365

            dict_df = database_interaction.get_historical_from_db(
                granularity=granularity,
                symbols=self.symbols,
                num_days=days
            )

            logger.info(f'...Running hyper on {len(self.symbols)} symbols')

            start_time = time.time()
            for i, (key, value) in enumerate(dict_df.items()):
                current_dict = {key: value}

                # Initialize the provided strategy dynamically
                strat = strategy_obj(current_dict, risk_object=risk, with_sizing=True)

                # Run hyperparameter optimization with dynamic ranges
                hyper = Hyper(
                    strategy_object=strat,
                    close=strat.close,
                    **param_ranges
                )

                # Export hyperparameter results to the database
                database_interaction.export_hyper_to_db(
                    strategy=strat,
                    hyper=hyper
                )

                exec_time = time.time() - start_time
                logger.info(f"Symbol {key} completed in {exec_time:.2f} seconds")
                
                # Track progress
                progress = (i + 1) / len(dict_df.keys()) * 100
                logger.info(f"Progress: {progress:.1f}% - {i+1}/{len(dict_df.keys())}")

                # Store results
                results[key] = {
                    'best_params': hyper.best_params,
                    'best_score': hyper.best_score
                }

                # Cleanup to free memory
                del hyper
                gc.collect()
                
        return results

def lambda_handler(event, context):
    """AWS Lambda handler for hyperparameter optimization"""
    try:
        # Get configuration from event or use defaults
        strategy_name = event.get('strategy', 'SuperTrend')
        symbols = event.get('symbols', ['BTC-USD', 'ETH-USD'])
        granularities = event.get('granularities', ['ONE_MINUTE', 'FIVE_MINUTE'])
        
        # Get parameter ranges from event or use defaults
        param_ranges = event.get('param_ranges', {
            'period_range': range(5, 30, 5),
            'multiplier_range': range(2, 10)
        })
        
        # Dynamically import the strategy class
        strategy_module = __import__(f'core.strategies.{strategy_name.lower()}', fromlist=[strategy_name])
        strategy_class = getattr(strategy_module, strategy_name)
        
        logger.info(f"Starting hyperparameter optimization for strategy: {strategy_name}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Granularities: {granularities}")
        logger.info(f"Parameter ranges: {param_ranges}")
        
        # Initialize optimizer and run optimization
        optimizer = HyperOptimizer(symbols=symbols, granularities=granularities)
        results = optimizer.run_hyper(strategy_obj=strategy_class, param_ranges=param_ranges)
        
        return {
            'statusCode': 200,
            'body': {
                'status': 'success',
                'message': f"Hyperparameter optimization completed for {len(symbols)} symbols",
                'results': results
            }
        }
        
    except Exception as e:
        logger.error(f"Error running hyperparameter optimization: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'status': 'error',
                'message': f"Error running hyperparameter optimization: {str(e)}"
            }
        }
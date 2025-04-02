import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import talib as ta
import gc
import matplotlib.pyplot as plt
import optuna
from functools import partial
import joblib
from datetime import datetime

# Assuming these imports work with your existing setup
from core.database_interaction import get_historical_from_db, save_backtest_results, get_db_connection, initialize_backtest_tables
from core.risk import Risk_Handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ai_backtest_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AIBacktestOptimizer:
    def __init__(self, symbol='BTC-USD', granularity='ONE_MINUTE', num_days=100, 
                 n_trials=50, study_name=None):
        """
        Initialize the AI Backtest Optimizer
        
        Args:
            symbol (str): Trading symbol (default: 'BTC-USD')
            granularity (str): Time granularity (default: 'ONE_MINUTE')
            num_days (int): Number of days of historical data to use (default: 100)
            n_trials (int): Number of optimization trials to run
            study_name (str): Name for the optuna study (for persistence)
        """
        self.symbol = symbol
        self.granularity = granularity
        self.num_days = num_days
        self.n_trials = n_trials
        
        # Create a unique study name if not provided
        if study_name is None:
            self.study_name = f"{symbol}_{granularity}_{datetime.now().strftime('%Y%m%d')}"
        else:
            self.study_name = study_name
            
        # Load historical data
        logger.info(f"Loading historical data for {symbol} ({granularity}) - {num_days} days")
        data_dict = get_historical_from_db(granularity, symbol, num_days)
        
        if not data_dict or symbol not in data_dict:
            raise ValueError(f"Failed to load historical data for {symbol}")
            
        self.data = data_dict[symbol]
        logger.info(f"Loaded {len(self.data)} data points")
        
        # Initialize risk handler
        self.risk_handler = Risk_Handler()
        
        # Initialize storage for feature data
        self.feature_data = None
        self.best_params = None
        self.best_model = None
        self.best_backtest_results = None
        
    def prepare_features(self):
        """
        Calculate technical indicators and prepare features for machine learning
        """
        try:
            logger.info("Preparing technical indicators and features...")
            df = self.data.copy()
            
            # Calculate technical indicators
            # RSI - Relative Strength Index
            for period in [7, 14, 21]:
                df[f'rsi_{period}'] = ta.RSI(df['close'], timeperiod=period)
            
            # ADX - Average Directional Index
            for period in [14, 21]:
                df[f'adx_{period}'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=period)
            
            # MACD - Moving Average Convergence Divergence
            for (fast, slow, signal) in [(12, 26, 9), (8, 21, 5)]:
                macd, macd_signal, macd_hist = ta.MACD(
                    df['close'], 
                    fastperiod=fast, 
                    slowperiod=slow, 
                    signalperiod=signal
                )
                df[f'macd_{fast}_{slow}'] = macd
                df[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
                df[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist
            
            # Bollinger Bands
            for (period, devup, devdn) in [(20, 2, 2), (10, 1.5, 1.5)]:
                upper, middle, lower = ta.BBANDS(
                    df['close'],
                    timeperiod=period,
                    nbdevup=devup,
                    nbdevdn=devdn
                )
                df[f'bb_upper_{period}'] = upper
                df[f'bb_middle_{period}'] = middle
                df[f'bb_lower_{period}'] = lower
                df[f'bb_width_{period}'] = (upper - lower) / middle
            
            # ATR - Average True Range
            for period in [14, 21]:
                df[f'atr_{period}'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                if len(df) > period:  # Ensure enough data points
                    df[f'sma_{period}'] = ta.SMA(df['close'], timeperiod=period)
                    df[f'ema_{period}'] = ta.EMA(df['close'], timeperiod=period)
            
            # Add moving average crossovers
            if 'sma_10' in df.columns and 'sma_20' in df.columns:
                df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
            
            if 'ema_10' in df.columns and 'ema_20' in df.columns:
                df['ema_10_20_cross'] = np.where(df['ema_10'] > df['ema_20'], 1, -1)
            
            # Price change features
            df['pct_change'] = df['close'].pct_change()
            for period in [1, 5, 10, 20]:
                df[f'return_{period}d'] = df['close'].pct_change(periods=period)
            
            # Volume features
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma_10'] = ta.SMA(df['volume'], timeperiod=10)
            df['volume_ma_ratio'] = df['volume'] / df['volume_ma_10']
            
            # Volatility
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            
            # Generate target labels with variable look-ahead
            # The optimization will determine the best prediction horizon
            for horizon in [1, 3, 5]:
                # 1 = price goes up, 0 = price goes down
                df[f'target_{horizon}'] = np.where(df['close'].shift(-horizon) > df['close'], 1, 0)
            
            # Drop NaN values
            df = df.dropna()
            
            # Check for NaNs
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                logger.warning(f"NaN values found in columns: {nan_cols}")
                df = df.dropna()  # Ensure all NaNs are dropped

            # Check for infinities
            inf_cols = df.columns[(df == np.inf).any() | (df == -np.inf).any()].tolist()
            if inf_cols:
                logger.warning(f"Infinite values found in columns: {inf_cols}")
                # Replace infinities with large but finite values
                df = df.replace([np.inf, -np.inf], [1e9, -1e9])

            # Log feature statistics to help debugging
            logger.info(f"Feature data shape after preprocessing: {df.shape}")
            if df.shape[0] < 100:
                logger.warning(f"Very small dataset after preprocessing: only {df.shape[0]} rows")
            
            self.feature_data = df
            logger.info(f"Feature preparation complete. Dataset shape: {df.shape}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def objective(self, trial, validation_size=0.2):
        """
        Objective function for Optuna to optimize
        
        Args:
            trial: Optuna trial object
            validation_size: Size of validation dataset
            
        Returns:
            float: Metric to optimize (e.g., profit factor or Sharpe ratio)
        """
        try:
            if self.feature_data is None:
                self.feature_data = self.prepare_features()
                
            df = self.feature_data.copy()
            
            # Hyperparameters for the ML model
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 25)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            
            # Hyperparameters for feature selection
            rsi_period = trial.suggest_categorical('rsi_period', [7, 14, 21])
            sma_fast = trial.suggest_categorical('sma_fast', [5, 10, 20])
            sma_slow = trial.suggest_categorical('sma_slow', [20, 50, 100])
            macd_fast = trial.suggest_categorical('macd_fast', [8, 12])
            macd_slow = trial.suggest_categorical('macd_slow', [21, 26])
            
            # Hyperparameters for prediction
            prediction_horizon = trial.suggest_categorical('prediction_horizon', [1, 3, 5])
            # Lowered the minimum threshold to allow more trades
            probability_threshold = trial.suggest_float('probability_threshold', 0.3, 0.8)
            
            # Hyperparameters for trading strategy - adjusted ranges
            position_size_pct = trial.suggest_float('position_size_pct', 0.05, 0.5)
            stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.005, 0.05)
            take_profit_pct = trial.suggest_float('take_profit_pct', 0.01, 0.15)
            
            # Select target based on prediction horizon
            target_col = f'target_{prediction_horizon}'
            
            # Improved feature selection logic
            feature_cols = []
            for col in df.columns:
                if col.startswith(f'rsi_{rsi_period}'):
                    feature_cols.append(col)
                elif col.startswith(f'sma_{sma_fast}') or col.startswith(f'sma_{sma_slow}'):
                    feature_cols.append(col)
                elif col.startswith(f'ema_{sma_fast}') or col.startswith(f'ema_{sma_slow}'):
                    feature_cols.append(col)
                elif col.startswith(f'macd_{macd_fast}_{macd_slow}'):
                    feature_cols.append(col)
                elif 'adx_' in col or 'atr_' in col or 'bb_' in col:
                    feature_cols.append(col)
                elif 'return_' in col or 'volume_' in col or col == 'high_low_range':
                    feature_cols.append(col)
            
            # Ensure we have sufficient features with improved selection logic
            if len(feature_cols) < 5:
                # More structured approach to feature selection
                core_features = []
                
                # Include at least one RSI indicator
                rsi_cols = [col for col in df.columns if 'rsi_' in col]
                if rsi_cols:
                    core_features.extend(rsi_cols[:2])  # Take at most 2 RSI features
                
                # Include at least one moving average indicator
                ma_cols = [col for col in df.columns if 'sma_' in col or 'ema_' in col]
                if ma_cols:
                    core_features.extend(ma_cols[:4])  # Take at most 4 MA features
                
                # Include MACD if available
                macd_cols = [col for col in df.columns if 'macd_' in col]
                if macd_cols:
                    core_features.extend(macd_cols[:3])  # Take at most 3 MACD features
                
                # Add more core indicators
                other_cols = [col for col in df.columns if 'atr_' in col or 'bb_' in col or 'adx_' in col]
                if other_cols:
                    core_features.extend(other_cols[:3])  # Take at most 3 other features
                
                # Add some price change features
                price_cols = [col for col in df.columns if 'return_' in col or 'pct_change' in col]
                if price_cols:
                    core_features.extend(price_cols[:2])  # Take at most 2 price features
                
                # If we still don't have enough features, add more
                if len(core_features) < 5:
                    remaining_cols = [col for col in df.columns if col not in 
                                    ['open', 'high', 'low', 'close', 'volume'] + 
                                    [f'target_{h}' for h in [1, 3, 5]] and
                                    col not in core_features]
                    core_features.extend(remaining_cols[:10-len(core_features)])
                
                feature_cols = core_features
            
            logger.info(f"Selected {len(feature_cols)} features: {feature_cols[:5]}...")
            
            # Prepare data for training
            X = df[feature_cols]
            y = df[target_col]
            
            # Train/validation split - use temporal split (no shuffle)
            split_idx = int(len(df) * (1 - validation_size))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train the model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions on validation set
            val_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1 (price up)
            
            # Check model performance
            val_predictions = (val_proba > 0.5).astype(int)
            val_accuracy = accuracy_score(y_val, val_predictions)
            logger.info(f"Model validation accuracy: {val_accuracy:.4f}")
            
            # If model performance is extremely poor, penalize but don't reject completely
            if val_accuracy < 0.51:  # Barely better than random
                logger.warning(f"Model has very poor predictive power: accuracy={val_accuracy:.4f}")
                return -5000  # Better than -99999 but still bad
            
            # Create a copy of validation data for backtest simulation
            backtest_df = df.iloc[split_idx:].copy()
            backtest_df['prediction_proba'] = val_proba
            backtest_df['prediction'] = (val_proba > probability_threshold).astype(int)
            
            # Log distribution of predictions and probabilities
            logger.info(f"Prediction stats: Count={len(backtest_df)}, "
                      f"Predicted Ups={backtest_df['prediction'].sum()}, "
                      f"Proba min={val_proba.min():.4f}, max={val_proba.max():.4f}, mean={val_proba.mean():.4f}")
            
            # Simulate trading on validation set
            initial_capital = 10000
            backtest_df['capital'] = float(initial_capital)
            backtest_df['position'] = 0.0
            backtest_df['entry_price'] = 0.0
            backtest_df['exit_price'] = 0.0
            backtest_df['holdings'] = 0.0
            backtest_df['total_value'] = float(initial_capital)
            backtest_df['stop_level'] = 0.0
            backtest_df['profit_target'] = 0.0
            
            # Trading simulation
            trades = []
            currently_in_trade = False
            trade_entry_price = 0
            trade_entry_idx = None
            
            # Loop through the validation data (starting from second row)
            for i in range(1, len(backtest_df)):
                prev_idx = backtest_df.index[i-1]
                curr_idx = backtest_df.index[i]
                
                prev_capital = backtest_df.loc[prev_idx, 'capital']
                prev_position = backtest_df.loc[prev_idx, 'position']
                prev_total = backtest_df.loc[prev_idx, 'total_value']
                
                curr_price = backtest_df.loc[curr_idx, 'close']
                prev_prediction = backtest_df.loc[prev_idx, 'prediction']
                prev_proba = backtest_df.loc[prev_idx, 'prediction_proba']
                
                # Modified confidence check - less strict to allow more trades
                high_confidence = prev_proba > (probability_threshold - 0.1)
                
                # Copy forward capital and positions by default (will override if there's a trade)
                backtest_df.loc[curr_idx, 'capital'] = prev_capital
                backtest_df.loc[curr_idx, 'position'] = prev_position
                
                # Check for trade exit based on stop-loss or take-profit
                if currently_in_trade:
                    curr_stop = backtest_df.loc[prev_idx, 'stop_level']
                    curr_target = backtest_df.loc[prev_idx, 'profit_target']
                    
                    # Check if stop-loss was hit
                    if backtest_df.loc[curr_idx, 'low'] <= curr_stop:
                        # Exit at stop price
                        exit_price = curr_stop
                        currently_in_trade = False
                        
                        # Calculate position value and update capital
                        position_value = prev_position * exit_price
                        backtest_df.loc[curr_idx, 'capital'] = prev_capital + position_value
                        backtest_df.loc[curr_idx, 'position'] = 0
                        backtest_df.loc[curr_idx, 'exit_price'] = exit_price
                        
                        # Record trade
                        trade_return = (exit_price / trade_entry_price) - 1
                        trades.append({
                            'entry_date': backtest_df.index[trade_entry_idx],
                            'exit_date': curr_idx,
                            'entry_price': trade_entry_price,
                            'exit_price': exit_price,
                            'return': trade_return,
                            'exit_type': 'stop_loss'
                        })
                    
                    # Check if take-profit was hit
                    elif backtest_df.loc[curr_idx, 'high'] >= curr_target:
                        # Exit at target price
                        exit_price = curr_target
                        currently_in_trade = False
                        
                        # Calculate position value and update capital
                        position_value = prev_position * exit_price
                        backtest_df.loc[curr_idx, 'capital'] = prev_capital + position_value
                        backtest_df.loc[curr_idx, 'position'] = 0
                        backtest_df.loc[curr_idx, 'exit_price'] = exit_price
                        
                        # Record trade
                        trade_return = (exit_price / trade_entry_price) - 1
                        trades.append({
                            'entry_date': backtest_df.index[trade_entry_idx],
                            'exit_date': curr_idx,
                            'entry_price': trade_entry_price,
                            'exit_price': exit_price,
                            'return': trade_return,
                            'exit_type': 'take_profit'
                        })
                    
                    # Exit based on prediction change - relaxed to only check prediction
                    elif prev_prediction == 0:
                        # Exit at current price
                        exit_price = curr_price
                        currently_in_trade = False
                        
                        # Calculate position value and update capital
                        position_value = prev_position * exit_price
                        backtest_df.loc[curr_idx, 'capital'] = prev_capital + position_value
                        backtest_df.loc[curr_idx, 'position'] = 0
                        backtest_df.loc[curr_idx, 'exit_price'] = exit_price
                        
                        # Record trade
                        trade_return = (exit_price / trade_entry_price) - 1
                        trades.append({
                            'entry_date': backtest_df.index[trade_entry_idx],
                            'exit_date': curr_idx,
                            'entry_price': trade_entry_price,
                            'exit_price': exit_price,
                            'return': trade_return,
                            'exit_type': 'signal'
                        })
                
                # Check for new trade entry - relaxed to only check prediction, not high confidence
                elif prev_prediction == 1 and not currently_in_trade:
                    # Enter new long position
                    position_size = prev_capital * position_size_pct
                    shares_bought = position_size / curr_price
                    trade_entry_price = curr_price
                    trade_entry_idx = i
                    
                    # Set stop-loss and take-profit levels
                    stop_level = curr_price * (1 - stop_loss_pct)
                    profit_target = curr_price * (1 + take_profit_pct)
                    
                    # Update position and capital
                    backtest_df.loc[curr_idx, 'capital'] = prev_capital - position_size
                    backtest_df.loc[curr_idx, 'position'] = shares_bought
                    backtest_df.loc[curr_idx, 'entry_price'] = trade_entry_price
                    backtest_df.loc[curr_idx, 'stop_level'] = stop_level
                    backtest_df.loc[curr_idx, 'profit_target'] = profit_target
                    
                    currently_in_trade = True
                
                # Update holdings and total value
                backtest_df.loc[curr_idx, 'holdings'] = backtest_df.loc[curr_idx, 'position'] * curr_price
                backtest_df.loc[curr_idx, 'total_value'] = backtest_df.loc[curr_idx, 'capital'] + backtest_df.loc[curr_idx, 'holdings']
            
            # Calculate performance metrics
            if not trades:
                # Log details about why no trades were generated
                logger.warning(f"No trades executed. Trial params: {trial.params}")
                logger.warning(f"Predictions summary: Mean={backtest_df['prediction'].mean()}, Sum={backtest_df['prediction'].sum()}")
                logger.warning(f"Probability summary: Min={backtest_df['prediction_proba'].min()}, "
                             f"Max={backtest_df['prediction_proba'].max()}, Mean={backtest_df['prediction_proba'].mean()}")
                
                # Instead of completely rejecting, give a very low but not worst possible score
                # This helps optuna explore more of the parameter space
                return -1000
            
            # Calculate returns and metrics
            backtest_df['daily_return'] = backtest_df['total_value'].pct_change()
            backtest_df['cumulative_return'] = (backtest_df['total_value'] / initial_capital) - 1
            
            # Calculate drawdowns
            backtest_df['peak_value'] = backtest_df['total_value'].cummax()
            backtest_df['drawdown'] = (backtest_df['total_value'] - backtest_df['peak_value']) / backtest_df['peak_value']
            
            # Performance metrics
            final_value = backtest_df['total_value'].iloc[-1]
            total_return = (final_value / initial_capital - 1)
            max_drawdown = backtest_df['drawdown'].min()
            
            # Calculate win rate
            trades_df = pd.DataFrame(trades)
            if len(trades_df) > 0:
                winning_trades = (trades_df['return'] > 0).sum()
                win_rate = winning_trades / len(trades_df)
                
                # Calculate profit factor (sum of profits / sum of losses)
                profits = trades_df.loc[trades_df['return'] > 0, 'return'].sum()
                losses = abs(trades_df.loc[trades_df['return'] < 0, 'return'].sum()) if len(trades_df.loc[trades_df['return'] < 0]) > 0 else 0.001
                
                profit_factor = profits / losses if losses > 0 else profits
                
                # Calculate average win/loss ratio
                avg_win = trades_df.loc[trades_df['return'] > 0, 'return'].mean() if winning_trades > 0 else 0
                avg_loss = abs(trades_df.loc[trades_df['return'] < 0, 'return'].mean()) if len(trades_df) - winning_trades > 0 else 0.001
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else avg_win
                
                # Calculate Sharpe ratio (annualized)
                annual_factor = 252  # trading days per year
                if backtest_df['daily_return'].std() > 0:
                    sharpe_ratio = backtest_df['daily_return'].mean() / backtest_df['daily_return'].std() * (annual_factor ** 0.5)
                else:
                    sharpe_ratio = 0
            else:
                win_rate = 0
                profit_factor = 0
                win_loss_ratio = 0
                sharpe_ratio = 0
            
            # Combine metrics into a single optimization score
            # You can adjust these weights based on what's most important for your strategy
            if profit_factor == 0:
                optimization_score = -500  # Penalize but don't completely reject
            else:
                optimization_score = (
                    2.0 * profit_factor +          # Weight profit factor heavily
                    1.0 * win_rate +              # Include win rate
                    0.5 * win_loss_ratio +        # Include win/loss ratio
                    1.0 * total_return -          # Include total return
                    2.0 * abs(max_drawdown) +     # Penalize drawdowns
                    1.0 * sharpe_ratio +          # Include risk-adjusted return
                    0.2 * (len(trades_df) / 20)   # Small bonus for more trades (up to a point)
                )
            
            # Store metrics for logging
            trial.set_user_attr('win_rate', win_rate)
            trial.set_user_attr('profit_factor', profit_factor)
            trial.set_user_attr('total_return', total_return)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('num_trades', len(trades_df))
            
            logger.info(f"Trial {trial.number}: Score={optimization_score:.4f}, "
                      f"Return={total_return:.4f}, Win Rate={win_rate:.4f}, "
                      f"PF={profit_factor:.4f}, Trades={len(trades_df)}")
            
            return optimization_score
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            import traceback
            traceback.print_exc()
            return -800  # Return a low but not worst possible score on error
    
    def optimize(self, storage=None):
        """
        Run hyperparameter optimization
        
        Args:
            storage: Optuna storage URL (optional, for distributed optimization)
            
        Returns:
            dict: Best parameters
        """
        try:
            logger.info(f"Starting hyperparameter optimization for {self.symbol} ({self.granularity})")
            
            # Create Optuna study
            if storage:
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=storage,
                    load_if_exists=True,
                    direction="maximize"
                )
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    direction="maximize"
                )
            
            # Run optimization
            study.optimize(self.objective, n_trials=self.n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Optimization complete. Best score: {best_value:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            # Store best parameters
            self.best_params = best_params
            
            # Save the study for future reference
            joblib.dump(study, f"optimization_{self.symbol}_{self.granularity}.pkl")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            raise  
# Continue from where the code left off - completing the run_backtest_with_best_params method

    def run_backtest_with_best_params(self, test_data=None):
        """
        Run a backtest using the best parameters from optimization
        
        Args:
            test_data: Optional DataFrame for out-of-sample testing
            
        Returns:
            tuple: (results, backtest_df)
        """
        try:
            if self.best_params is None:
                logger.error("No optimized parameters available. Run optimize() first.")
                return None
                
            logger.info(f"Running backtest with best parameters for {self.symbol}")
            
            if self.feature_data is None:
                self.feature_data = self.prepare_features()
                
            # Use either provided test data or last 20% of available data
            if test_data is not None:
                df = test_data
            else:
                df = self.feature_data
                split_idx = int(len(df) * 0.8)
                df = df.iloc[split_idx:].copy()
            
            # Extract parameters
            n_estimators = self.best_params['n_estimators']
            max_depth = self.best_params['max_depth']
            min_samples_split = self.best_params['min_samples_split']
            min_samples_leaf = self.best_params['min_samples_leaf']
            
            rsi_period = self.best_params['rsi_period']
            sma_fast = self.best_params['sma_fast']
            sma_slow = self.best_params['sma_slow']
            macd_fast = self.best_params['macd_fast']
            macd_slow = self.best_params['macd_slow']
            
            prediction_horizon = self.best_params['prediction_horizon']
            probability_threshold = self.best_params['probability_threshold']
            
            position_size_pct = self.best_params['position_size_pct']
            stop_loss_pct = self.best_params['stop_loss_pct']
            take_profit_pct = self.best_params['take_profit_pct']
            
            # Select target based on prediction horizon
            target_col = f'target_{prediction_horizon}'
            
            # Select features based on hyperparameters
            feature_cols = []
            for col in df.columns:
                if col.startswith(f'rsi_{rsi_period}'):
                    feature_cols.append(col)
                elif col.startswith(f'sma_{sma_fast}') or col.startswith(f'sma_{sma_slow}'):
                    feature_cols.append(col)
                elif col.startswith(f'ema_{sma_fast}') or col.startswith(f'ema_{sma_slow}'):
                    feature_cols.append(col)
                elif col.startswith(f'macd_{macd_fast}_{macd_slow}'):
                    feature_cols.append(col)
                elif 'adx_' in col or 'atr_' in col or 'bb_' in col:
                    feature_cols.append(col)
                elif 'return_' in col or 'volume_' in col or col == 'high_low_range':
                    feature_cols.append(col)
            
            # Ensure we have at least some features
            if len(feature_cols) < 5:
                feature_cols = [col for col in df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume'] + 
                             [f'target_{h}' for h in [1, 3, 5]]]
            
            # Use full dataset for training (to get the most data)
            full_df = self.feature_data.copy()
            X_full = full_df[feature_cols]
            y_full = full_df[target_col]
            
            # Train the model on full dataset
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_full, y_full)
            self.best_model = model
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Make predictions on test data
            X_test = df[feature_cols]
            test_proba = model.predict_proba(X_test)[:, 1]
            
            # Create a copy of test data for backtest simulation
            backtest_df = df.copy()
            backtest_df['prediction_proba'] = test_proba
            backtest_df['prediction'] = (test_proba > probability_threshold).astype(int)
            
            # Simulate trading
            initial_capital = 10000
            backtest_df['capital'] = float(initial_capital)
            backtest_df['position'] = 0.0
            backtest_df['entry_price'] = 0.0
            backtest_df['exit_price'] = 0.0
            backtest_df['holdings'] = 0.0
            backtest_df['total_value'] = float(initial_capital)
            backtest_df['stop_level'] = 0.0
            backtest_df['profit_target'] = 0.0
            
            # Trading simulation
            trades = []
            currently_in_trade = False
            trade_entry_price = 0
            trade_entry_idx = None
            
            # Loop through the test data (starting from second row)
            for i in range(1, len(backtest_df)):
                prev_idx = backtest_df.index[i-1]
                curr_idx = backtest_df.index[i]
                
                prev_capital = backtest_df.loc[prev_idx, 'capital']
                prev_position = backtest_df.loc[prev_idx, 'position']
                prev_total = backtest_df.loc[prev_idx, 'total_value']
                
                curr_price = backtest_df.loc[curr_idx, 'close']
                prev_prediction = backtest_df.loc[prev_idx, 'prediction']
                prev_proba = backtest_df.loc[prev_idx, 'prediction_proba']
                high_confidence = prev_proba > probability_threshold or prev_proba < (1 - probability_threshold)
                
                # Copy forward capital and positions by default (will override if there's a trade)
                backtest_df.loc[curr_idx, 'capital'] = prev_capital
                backtest_df.loc[curr_idx, 'position'] = prev_position
                
                # Check for trade exit based on stop-loss or take-profit
                if currently_in_trade:
                    curr_stop = backtest_df.loc[prev_idx, 'stop_level']
                    curr_target = backtest_df.loc[prev_idx, 'profit_target']
                    
                    # Check if stop-loss was hit
                    if backtest_df.loc[curr_idx, 'low'] <= curr_stop:
                        # Exit at stop price
                        exit_price = curr_stop
                        currently_in_trade = False
                        
                        # Calculate position value and update capital
                        position_value = prev_position * exit_price
                        backtest_df.loc[curr_idx, 'capital'] = prev_capital + position_value
                        backtest_df.loc[curr_idx, 'position'] = 0
                        backtest_df.loc[curr_idx, 'exit_price'] = exit_price
                        
                        # Record trade
                        trade_return = (exit_price / trade_entry_price) - 1
                        trades.append({
                            'entry_date': backtest_df.index[trade_entry_idx],
                            'exit_date': curr_idx,
                            'entry_price': trade_entry_price,
                            'exit_price': exit_price,
                            'return': trade_return,
                            'exit_type': 'stop_loss'
                        })
                    
                    # Check if take-profit was hit
                    elif backtest_df.loc[curr_idx, 'high'] >= curr_target:
                        # Exit at target price
                        exit_price = curr_target
                        currently_in_trade = False
                        
                        # Calculate position value and update capital
                        position_value = prev_position * exit_price
                        backtest_df.loc[curr_idx, 'capital'] = prev_capital + position_value
                        backtest_df.loc[curr_idx, 'position'] = 0
                        backtest_df.loc[curr_idx, 'exit_price'] = exit_price
                        
                        # Record trade
                        trade_return = (exit_price / trade_entry_price) - 1
                        trades.append({
                            'entry_date': backtest_df.index[trade_entry_idx],
                            'exit_date': curr_idx,
                            'entry_price': trade_entry_price,
                            'exit_price': exit_price,
                            'return': trade_return,
                            'exit_type': 'take_profit'
                        })
                    
                    # Exit based on prediction change
                    elif prev_prediction == 0 and high_confidence:
                        # Exit at current price
                        exit_price = curr_price
                        currently_in_trade = False
                        
                        # Calculate position value and update capital
                        position_value = prev_position * exit_price
                        backtest_df.loc[curr_idx, 'capital'] = prev_capital + position_value
                        backtest_df.loc[curr_idx, 'position'] = 0
                        backtest_df.loc[curr_idx, 'exit_price'] = exit_price
                        
                        # Record trade
                        trade_return = (exit_price / trade_entry_price) - 1
                        trades.append({
                            'entry_date': backtest_df.index[trade_entry_idx],
                            'exit_date': curr_idx,
                            'entry_price': trade_entry_price,
                            'exit_price': exit_price,
                            'return': trade_return,
                            'exit_type': 'signal'
                        })
                
                # Check for new trade entry
                elif prev_prediction == 1 and high_confidence and not currently_in_trade:
                    # Enter new long position
                    position_size = prev_capital * position_size_pct
                    shares_bought = position_size / curr_price
                    trade_entry_price = curr_price
                    trade_entry_idx = i
                    
                    # Set stop-loss and take-profit levels
                    stop_level = curr_price * (1 - stop_loss_pct)
                    profit_target = curr_price * (1 + take_profit_pct)
                    
                    # Update position and capital
                    backtest_df.loc[curr_idx, 'capital'] = prev_capital - position_size
                    backtest_df.loc[curr_idx, 'position'] = shares_bought
                    backtest_df.loc[curr_idx, 'entry_price'] = trade_entry_price
                    backtest_df.loc[curr_idx, 'stop_level'] = stop_level
                    backtest_df.loc[curr_idx, 'profit_target'] = profit_target
                    
                    currently_in_trade = True
                
                # Update holdings and total value
                backtest_df.loc[curr_idx, 'holdings'] = backtest_df.loc[curr_idx, 'position'] * curr_price
                backtest_df.loc[curr_idx, 'total_value'] = backtest_df.loc[curr_idx, 'capital'] + backtest_df.loc[curr_idx, 'holdings']
            
            # Calculate performance metrics
            if not trades:
                logger.warning("No trades were executed in backtest")
                backtest_results = {
                    'total_return': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'num_trades': 0
                }
                return backtest_results, backtest_df
            
            # Calculate returns and metrics
            backtest_df['daily_return'] = backtest_df['total_value'].pct_change()
            backtest_df['cumulative_return'] = (backtest_df['total_value'] / initial_capital) - 1
            
            # Calculate drawdowns
            backtest_df['peak_value'] = backtest_df['total_value'].cummax()
            backtest_df['drawdown'] = (backtest_df['total_value'] - backtest_df['peak_value']) / backtest_df['peak_value']
            
            # Performance metrics
            final_value = backtest_df['total_value'].iloc[-1]
            total_return = (final_value / initial_capital - 1)
            max_drawdown = backtest_df['drawdown'].min()
            
            # Calculate win rate
            trades_df = pd.DataFrame(trades)
            winning_trades = (trades_df['return'] > 0).sum()
            win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
            
            # Calculate profit factor (sum of profits / sum of losses)
            profits = trades_df.loc[trades_df['return'] > 0, 'return'].sum() if winning_trades > 0 else 0
            losses = abs(trades_df.loc[trades_df['return'] < 0, 'return'].sum()) if len(trades_df) - winning_trades > 0 else 1
            profit_factor = profits / losses if losses > 0 else profits
            
            # Calculate Sharpe ratio (annualized)
            annual_factor = 252  # trading days per year
            if len(backtest_df['daily_return'].dropna()) > 0 and backtest_df['daily_return'].std() > 0:
                sharpe_ratio = backtest_df['daily_return'].mean() / backtest_df['daily_return'].std() * (annual_factor ** 0.5)
            else:
                sharpe_ratio = 0
            
            # Summarize results
            backtest_results = {
                'total_return': total_return,
                'annualized_return': total_return / (len(backtest_df) / 252),  # Annualized
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(trades_df),
                'feature_importance': feature_importance,
                'trades': trades_df
            }
            
            # Save summary to log
            logger.info(f"Backtest results for {self.symbol} ({self.granularity}):")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Profit Factor: {profit_factor:.2f}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"Number of Trades: {len(trades_df)}")
            
            # Save top feature importance
            logger.info("Top 10 important features:")
            for _, row in feature_importance.head(10).iterrows():
                logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
            
            # Store results for later use
            self.best_backtest_results = backtest_results
            
            return backtest_results, backtest_df
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_model(self, filename=None):
        """
        Save the trained model to a file
        
        Args:
            filename: Optional filename, will use default if not provided
        """
        if self.best_model is None:
            logger.error("No model available to save. Run backtest with best params first.")
            return
            
        if filename is None:
            filename = f"model_{self.symbol}_{self.granularity}_{datetime.now().strftime('%Y%m%d')}.pkl"
            
        logger.info(f"Saving model to {filename}")
        joblib.dump(self.best_model, filename)
        
        # Also save parameters
        param_filename = filename.replace('.pkl', '_params.json')
        import json
        with open(param_filename, 'w') as f:
            json.dump(self.best_params, f, indent=4)
            
        logger.info(f"Model and parameters saved successfully")
    
    def plot_backtest_results(self, backtest_df, save_path=None):
        """
        Plot the backtest results
        
        Args:
            backtest_df: DataFrame with backtest results
            save_path: Optional path to save the figure
        """
        try:
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Equity Curve
            plt.subplot(3, 1, 1)
            plt.plot(backtest_df['total_value'])
            plt.title(f'Equity Curve - {self.symbol} ({self.granularity})')
            plt.grid(True)
            plt.ylabel('Portfolio Value ($)')
            
            # Plot 2: Drawdown
            plt.subplot(3, 1, 2)
            plt.fill_between(backtest_df.index, backtest_df['drawdown'] * 100, 0, color='red', alpha=0.3)
            plt.title('Drawdown (%)')
            plt.grid(True)
            plt.ylabel('Drawdown %')
            
            # Plot 3: Trade Entry/Exit points on Price Chart
            plt.subplot(3, 1, 3)
            plt.plot(backtest_df['close'], label='Close Price', color='blue')
            
            # Mark entries
            entries = backtest_df[backtest_df['entry_price'] > 0]
            plt.scatter(entries.index, entries['entry_price'], 
                      marker='^', color='green', s=100, label='Entry')
            
            # Mark exits
            exits = backtest_df[backtest_df['exit_price'] > 0]
            plt.scatter(exits.index, exits['exit_price'],
                      marker='v', color='red', s=100, label='Exit')
            
            plt.title('Price Chart with Trades')
            plt.grid(True)
            plt.ylabel('Price')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Backtest chart saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")
            
    def save_backtest_to_db(self, backtest_results, backtest_df):
        """
        Save backtest results to database
        
        Args:
            backtest_results: Dictionary with backtest results
            backtest_df: DataFrame with detailed backtest data
        """
        try:
            # Initialize database tables if needed
            initialize_backtest_tables()
            
            # Prepare backtest summary
            backtest_summary = {
                'symbol': self.symbol,
                'granularity': self.granularity,
                'strategy_name': f"ML_RF_{self.study_name}",
                'start_date': backtest_df.index[0].strftime('%Y-%m-%d'),
                'end_date': backtest_df.index[-1].strftime('%Y-%m-%d'),
                'initial_capital': 10000,  # Same as used in backtest
                'final_capital': backtest_df['total_value'].iloc[-1],
                'total_return': backtest_results['total_return'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'max_drawdown': backtest_results['max_drawdown'],
                'num_trades': backtest_results['num_trades'],
                'win_rate': backtest_results['win_rate'],
                'profit_factor': backtest_results['profit_factor'],
                'params': str(self.best_params),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to database
            save_backtest_results(backtest_summary, backtest_df, backtest_results['trades'])
            logger.info(f"Backtest results saved to database successfully")
            
        except Exception as e:
            logger.error(f"Error saving backtest to database: {e}")
            import traceback
            traceback.print_exc()

# Add a main function to run the AI Backtest Optimizer
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Backtest Optimizer')
    parser.add_argument('--symbol', type=str, default='BTC-USD', help='Trading symbol')
    parser.add_argument('--granularity', type=str, default='ONE_MINUTE', help='Time granularity')
    parser.add_argument('--days', type=int, default=100, help='Number of days of historical data')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--study', type=str, default=None, help='Study name for persistence')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL')
    parser.add_argument('--plot', action='store_true', help='Plot backtest results')
    parser.add_argument('--save', action='store_true', help='Save model and results')
    parser.add_argument('--db', action='store_true', help='Save results to database')
    
    args = parser.parse_args()
    
    try:
        # Initialize the optimizer
        optimizer = AIBacktestOptimizer(
            symbol=args.symbol,
            granularity=args.granularity,
            num_days=args.days,
            n_trials=args.trials,
            study_name=args.study
        )
        
        # Prepare features (do this separately to catch errors early)
        optimizer.prepare_features()
        
        # Run optimization
        best_params = optimizer.optimize(storage=args.storage)
        
        # Run backtest with best parameters
        results, backtest_df = optimizer.run_backtest_with_best_params()
        
        # Plot results if requested
        if args.plot:
            plot_path = f"backtest_{args.symbol}_{args.granularity}.png" if args.save else None
            optimizer.plot_backtest_results(backtest_df, save_path=plot_path)
        
        # Save model if requested
        if args.save:
            optimizer.save_model()
        
        # Save to database if requested
        if args.db:
            optimizer.save_backtest_to_db(results, backtest_df)
        
        logger.info("AI Backtest Optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
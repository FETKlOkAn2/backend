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
from database.database_interaction import get_historical_from_db, save_backtest_results, get_db_connection, initialize_backtest_tables
from core.risk import Risk_Handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ai_backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AIBacktest:
    def __init__(self, symbol='BTC-USD', granularity='ONE_MINUTE', num_days=200):
        """
        Initialize the AI Backtest framework
        
        Args:
            symbol (str): Trading symbol (default: 'BTC-USD')
            granularity (str): Time granularity (default: 'ONE_MINUTE')
            num_days (int): Number of days of historical data to use (default: 100)
        """
        self.symbol = symbol
        self.granularity = granularity
        self.num_days = num_days
        
        # Load historical data
        logger.info(f"Loading historical data for {symbol} ({granularity}) - {num_days} days")
        data_dict = get_historical_from_db(granularity, symbol, num_days)
        
        if not data_dict or symbol not in data_dict:
            raise ValueError(f"Failed to load historical data for {symbol}")
            
        self.data = data_dict[symbol]
        logger.info(f"Loaded {len(self.data)} data points")
        
        # Initialize ML components
        self.ml_model = None
        self.feature_data = None
        self.backtest_results = None
        self.risk_handler = Risk_Handler()

    def prepare_features(self):
        """
        Calculate technical indicators and prepare features for machine learning
        """
        try:
            logger.info("Preparing technical indicators and features...")
            df = self.data.copy()
            
            # Calculate technical indicators
            # RSI - Relative Strength Index
            df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
            
            # ADX - Average Directional Index
            df['adx_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # MACD - Moving Average Convergence Divergence
            macd, macd_signal, macd_hist = ta.MACD(
                df['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(
                df['close'],
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            
            # ATR - Average True Range
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Moving Averages
            df['sma_10'] = ta.SMA(df['close'], timeperiod=10)
            df['sma_30'] = ta.SMA(df['close'], timeperiod=30)
            df['ema_10'] = ta.EMA(df['close'], timeperiod=10)
            df['ema_30'] = ta.EMA(df['close'], timeperiod=30)
            
            # Price change features
            df['pct_change'] = df['close'].pct_change()
            df['return_1d'] = df['close'].pct_change(periods=1)
            df['return_5d'] = df['close'].pct_change(periods=5)
            
            # Generate target labels (price movement direction)
            # 1 = price goes up, 0 = price goes down
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            
            # Drop NaN values
            df = df.dropna()
            
            self.feature_data = df
            logger.info(f"Feature preparation complete. Dataset shape: {df.shape}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def train_model(self, test_size=0.2, random_state=42):
        """
        Train machine learning model for price prediction
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        try:
            if self.feature_data is None:
                self.feature_data = self.prepare_features()
            
            df = self.feature_data
            
            # Select features (remove date, target, and original OHLCV)
            feature_columns = [col for col in df.columns if col not in 
                              ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            # Prepare features and target
            X = df[feature_columns]
            y = df['target']
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False, random_state=random_state
            )
            
            logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
            
            # Train model
            logger.info("Training RandomForest model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:\n" + 
                        classification_report(y_test, y_pred))
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            logger.info("Top 10 important features:")
            logger.info(feature_importance.head(10))
            
            self.ml_model = model
            self.feature_importance = feature_importance
            
            return model, accuracy
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
            
    def run_backtest(self, initial_capital=10000, position_size_pct=0.2):
        """
        Run backtest simulation using trained model predictions
        
        Args:
            initial_capital (float): Initial trading capital
            position_size_pct (float): Percentage of capital to risk per trade
        """
        try:
            if self.ml_model is None:
                logger.info("Model not trained. Training now...")
                self.train_model()
                
            if self.feature_data is None:
                logger.info("Features not prepared. Preparing now...")
                self.prepare_features()
                
            df = self.feature_data.copy()
            
            # Get feature columns
            feature_columns = [col for col in df.columns if col not in 
                              ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            # Make predictions
            X = df[feature_columns]
            df['prediction'] = self.ml_model.predict(X)
            
            # Add prediction probability
            prediction_proba = self.ml_model.predict_proba(X)
            df['prediction_proba'] = prediction_proba[:, 1]  # Probability of class 1 (price up)
            
            # Initialize portfolio tracking
            df['capital'] = float(initial_capital)
            df['position'] = 0.0
            df['holdings'] = 0.0
            df['total_value'] = float(initial_capital)
            
            # Trading logic
            logger.info("Running backtest simulation...")
            for i in range(1, len(df)):
                prev_idx = df.index[i-1]
                curr_idx = df.index[i]
                
                prev_capital = df.loc[prev_idx, 'capital']
                prev_position = df.loc[prev_idx, 'position']
                prev_total = df.loc[prev_idx, 'total_value']
                
                # Current price and prediction
                curr_price = df.loc[curr_idx, 'close']
                prev_prediction = df.loc[prev_idx, 'prediction']
                prev_confidence = df.loc[prev_idx, 'prediction_proba']
                
                # High confidence threshold (only take trades with high confidence)
                high_confidence = prev_confidence > 0.6 or prev_confidence < 0.4
                
                # Trading logic with position sizing
                if prev_prediction == 1 and prev_position == 0 and high_confidence:
                    # Buy signal
                    position_size = prev_capital * position_size_pct
                    shares_bought = position_size / curr_price
                    
                    df.loc[curr_idx, 'capital'] = prev_capital - position_size
                    df.loc[curr_idx, 'position'] = shares_bought
                    df.loc[curr_idx, 'holdings'] = shares_bought * curr_price
                    
                elif prev_prediction == 0 and prev_position > 0:
                    # Sell signal
                    position_value = prev_position * curr_price
                    
                    df.loc[curr_idx, 'capital'] = prev_capital + position_value
                    df.loc[curr_idx, 'position'] = 0
                    df.loc[curr_idx, 'holdings'] = 0
                    
                else:
                    # Hold
                    df.loc[curr_idx, 'capital'] = prev_capital
                    df.loc[curr_idx, 'position'] = prev_position
                    df.loc[curr_idx, 'holdings'] = prev_position * curr_price
                
                # Update total value
                df.loc[curr_idx, 'total_value'] = df.loc[curr_idx, 'capital'] + df.loc[curr_idx, 'holdings']
            
            # Calculate returns and metrics
            df['daily_return'] = df['total_value'].pct_change()
            df['cumulative_return'] = (df['total_value'] / initial_capital) - 1
            
            # Calculate drawdowns
            df['peak_value'] = df['total_value'].cummax()
            df['drawdown'] = (df['total_value'] - df['peak_value']) / df['peak_value']
            
            # Performance metrics
            final_value = df['total_value'].iloc[-1]
            total_return = (final_value / initial_capital - 1) * 100
            max_drawdown = df['drawdown'].min() * 100
            
            # Count trades
            buy_signals = np.where(np.diff(df['position']) > 0)[0]
            sell_signals = np.where(np.diff(df['position']) < 0)[0]
            total_trades = len(buy_signals)
            
            # Calculate win rate
            trade_returns = []
            for i in range(len(sell_signals)):
                if i < len(buy_signals):
                    buy_price = df['close'].iloc[buy_signals[i]]
                    sell_price = df['close'].iloc[sell_signals[i]]
                    trade_return = (sell_price / buy_price) - 1
                    trade_returns.append(trade_return)
            
            winning_trades = sum(1 for x in trade_returns if x > 0)
            win_rate = (winning_trades / len(trade_returns)) * 100 if trade_returns else 0
            
            # Annualized return and Sharpe ratio
            annual_factor = 252  # trading days per year
            mean_daily_return = df['daily_return'].mean()
            std_daily_return = df['daily_return'].std()
            annualized_return = ((1 + mean_daily_return) ** annual_factor) - 1
            sharpe_ratio = (mean_daily_return / std_daily_return) * (annual_factor ** 0.5) if std_daily_return > 0 else 0
            
            # Compile results
            results = {
                'Initial Capital': initial_capital,
                'Final Value': final_value,
                'Total Return (%)': total_return,
                'Annualized Return (%)': annualized_return * 100,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': max_drawdown,
                'Total Trades': total_trades,
                'Win Rate (%)': win_rate,
                'Trading Period': f"{df.index[0]} to {df.index[-1]}"
            }
            
            logger.info("Backtest results:")
            for key, value in results.items():
                logger.info(f"{key}: {value}")
            
            self.backtest_results = results
            self.backtest_data = df

            backtest_id = save_backtest_results(self)
            if backtest_id:
                logger.info(f"Backtest results saved to database with ID: {backtest_id}")

            return results, df
        
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            raise

    def plot_results(self):
        """
        Plot backtest results including equity curve, drawdown, and trade signals
        """
        if self.backtest_data is None:
            logger.error("No backtest data available to plot")
            return
            
        try:
            df = self.backtest_data
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot equity curve
            axes[0].plot(df.index, df['total_value'], label='Portfolio Value')
            axes[0].set_title('Portfolio Equity Curve')
            axes[0].set_ylabel('Value ($)')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot drawdown
            axes[1].fill_between(df.index, df['drawdown'] * 100, 0, color='red', alpha=0.3)
            axes[1].set_title('Drawdown (%)')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].grid(True)
            
            # Plot price with buy/sell signals
            axes[2].plot(df.index, df['close'], label='Price')
            
            # Add buy signals
            buy_signals = df[df['position'] > df['position'].shift(1)]
            axes[2].scatter(buy_signals.index, buy_signals['close'], 
                           color='green', label='Buy', marker='^', s=100)
            
            # Add sell signals
            sell_signals = df[df['position'] < df['position'].shift(1)]
            axes[2].scatter(sell_signals.index, sell_signals['close'], 
                           color='red', label='Sell', marker='v', s=100)
            
            axes[2].set_title('Price Chart with Trading Signals')
            axes[2].set_ylabel('Price ($)')
            axes[2].set_xlabel('Date')
            axes[2].legend()
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'ai_backtest/ai_backtest_results/{self.symbol}_{self.granularity}.png')
            logger.info(f"Backtest results plot saved as '{self.symbol}_{self.granularity}.png'")
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = self.feature_importance.head(15)
            plt.barh(top_features['Feature'], top_features['Importance'])
            plt.title('Top 15 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'ai_backtest/feature_importance/{self.symbol}_{self.granularity}.png')
            logger.info(f"Feature importance plot saved as '{self.symbol}_{self.granularity}.png'")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            import traceback
            traceback.print_exc()

    def run_walk_forward_optimization(self, window_size=30, step_size=10, test_size=10):
        """
        Perform walk-forward optimization to test model stability
        
        Args:
            window_size (int): Size of training window in days
            step_size (int): How many days to advance for the next window
            test_size (int): How many days to use for out-of-sample testing
        """
        try:
            logger.info("Starting walk-forward optimization...")
            
            if self.feature_data is None:
                self.prepare_features()
                
            df = self.feature_data.copy()
            
            # Feature columns
            feature_columns = [col for col in df.columns if col not in 
                              ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            # Initialize results storage
            wfo_results = []
            
            # Convert index to days from start for easier windowing
            df = df.reset_index(names=['date'])
            
            # Calculate number of windows
            total_days = len(df)
            num_windows = (total_days - window_size - test_size) // step_size + 1
            
            logger.info(f"Running {num_windows} windows for walk-forward optimization")
            
            for i in range(num_windows):
                try:
                    # Define window boundaries
                    train_start = i * step_size
                    train_end = train_start + window_size
                    test_start = train_end
                    test_end = test_start + test_size
                    
                    if test_end > total_days:
                        break
                        
                    # Extract train/test data
                    train_data = df.iloc[train_start:train_end]
                    test_data = df.iloc[test_start:test_end]
                    
                    X_train = train_data[feature_columns]
                    y_train = train_data['target']
                    
                    X_test = test_data[feature_columns]
                    y_test = test_data['target']
                    
                    # Train model
                    model = RandomForestClassifier(
                        n_estimators=100, 
                        max_depth=10,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    
                    # Evaluate on test data
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Simulate trading
                    test_data = test_data.copy()
                    test_data['prediction'] = model.predict(X_test)
                    
                    # Simple buy/hold strategy based on predictions
                    initial_price = test_data['close'].iloc[0]
                    final_price = test_data['close'].iloc[-1]
                    
                    buy_hold_return = (final_price / initial_price) - 1
                    
                    # Strategy returns
                    returns = []
                    position = 0
                    
                    for j in range(1, len(test_data)):
                        curr_price = test_data['close'].iloc[j]
                        prev_price = test_data['close'].iloc[j-1]
                        prediction = test_data['prediction'].iloc[j-1]
                        
                        if prediction == 1 and position == 0:
                            # Buy
                            position = 1
                            entry_price = curr_price
                        elif prediction == 0 and position == 1:
                            # Sell and calculate return
                            position = 0
                            trade_return = (curr_price / entry_price) - 1
                            returns.append(trade_return)
                    
                    # Close any open position at the end
                    if position == 1:
                        trade_return = (test_data['close'].iloc[-1] / entry_price) - 1
                        returns.append(trade_return)
                    
                    # Calculate strategy metrics
                    if returns:
                        strategy_return = np.sum(returns)
                        win_rate = np.sum([r > 0 for r in returns]) / len(returns) * 100
                    else:
                        strategy_return = 0
                        win_rate = 0
                    
                    # Store results
                    window_result = {
                        'Window': i+1,
                        'Train_Start': df['date'].iloc[train_start],
                        'Train_End': df['date'].iloc[train_end-1],
                        'Test_Start': df['date'].iloc[test_start],
                        'Test_End': df['date'].iloc[test_end-1],
                        'Accuracy': accuracy,
                        'Strategy_Return': strategy_return,
                        'Buy_Hold_Return': buy_hold_return,
                        'Win_Rate': win_rate,
                        'Trades': len(returns)
                    }
                    
                    wfo_results.append(window_result)
                    
                    # Free memory
                    del train_data, test_data, model
                    gc.collect()
                    
                    logger.info(f"Window {i+1}/{num_windows}: Accuracy={accuracy:.4f}, Return={strategy_return:.4%}")
                    
                except Exception as e:
                    logger.error(f"Error in window {i+1}: {e}")
                    continue
            
            # Compile results
            wfo_df = pd.DataFrame(wfo_results)
            
            # Calculate overall metrics
            overall_accuracy = wfo_df['Accuracy'].mean()
            overall_return = wfo_df['Strategy_Return'].mean()
            overall_win_rate = wfo_df['Win_Rate'].mean()
            
            logger.info("\nWalk-Forward Optimization Results:")
            logger.info(f"Average Accuracy: {overall_accuracy:.4f}")
            logger.info(f"Average Return: {overall_return:.4%}")
            logger.info(f"Average Win Rate: {overall_win_rate:.2f}%")
            logger.info(f"Strategy vs Buy-Hold: {(wfo_df['Strategy_Return'] > wfo_df['Buy_Hold_Return']).mean():.2%} of windows outperformed")
            
            self.wfo_results = wfo_df
            return wfo_df
            
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """
    Main function to run the AI backtest
    """
    symbol_list = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD']
    granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY']
    try:
        with get_db_connection('ai_backtest') as conn:
            initialize_backtest_tables(conn)
        
        for symbol in symbol_list:
            for granularity in granularities:
                logger.info(f"Running AI backtest for {symbol} ({granularity})")
                
                # Initialize the AI backtest with symbol and granularity
                ai_backtest = AIBacktest(symbol=symbol, granularity=granularity, num_days=100)
                
                # Prepare features
                ai_backtest.prepare_features()
                
                # Train the model
                ai_backtest.train_model()
                
                # Run the backtest
                ai_backtest.run_backtest(initial_capital=10000, position_size_pct=0.2)
                
                # Plot the results
                ai_backtest.plot_results()
                
                logger.info(f"AI backtest for simulation completed successfully!")
                # Optional: Run walk-forward optimization
                ai_backtest.run_walk_forward_optimization()
         
    except Exception as e:
        logger.error(f"AI backtest simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from core.database_interaction import get_historical_from_db
from core.strategies import ADX, ATR, BollingerBands, EFratio, Kama, MACD, RSI, strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('backtest_simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AIBacktestSimulation:
    def __init__(self, symbol='BTC-USD', timeframe='15min'):
        """
        Initialize the AI Backtest Simulation
        
        Args:
            symbol (str): Trading symbol to backtest
            timeframe (str): Timeframe for historical data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.historical_data = get_historical_from_db('ONE_MINUTE', 'BTC-USD', 100)
        self.ml_model = None
        self.backtest_results = None

    # def fetch_historical_data(self, start_date='2024-01-01', end_date='2024-03-27'):
    #     """
    #     Fetch historical price data 
        
    #     Args:
    #         start_date (str): Start date for historical data
    #         end_date (str): End date for historical data
    #     """
    #     try:
    #         # In a real scenario, replace this with actual data fetching method
    #         # For simulation, we'll generate synthetic data
    #         dates = pd.date_range(start=start_date, end=end_date, freq=self.timeframe)
    #         np.random.seed(42)
            
    #         self.historical_data = pd.DataFrame({
    #             'date': dates,
    #             'open': np.cumsum(np.random.normal(0, 100, len(dates))) + 50000,
    #             'high': np.cumsum(np.random.normal(0, 100, len(dates))) + 50100,
    #             'low': np.cumsum(np.random.normal(0, 100, len(dates))) + 49900,
    #             'close': np.cumsum(np.random.normal(0, 100, len(dates))) + 50050,
    #             'volume': np.random.uniform(10, 100, len(dates))
    #         })
    #         self.historical_data.set_index('date', inplace=True)
            
    #         logger.info(f"Generated synthetic historical data for {self.symbol}")
    #         logger.info(f"Data shape: {self.historical_data.shape}")
            
    #     except Exception as e:
    #         logger.error(f"Error fetching historical data: {e}")
    #         raise

    def prepare_ml_features(self):
        """
        Prepare features for machine learning
        """
        try:
            # Calculate technical indicators
            self.historical_data['rsi'] = self._calculate_rsi(self.historical_data['close'])
            self.historical_data['adx'] = self._calculate_adx(
                self.historical_data['high'], 
                self.historical_data['low'], 
                self.historical_data['close']
            )
            
            # Generate labels (binary classification: up/down)
            self.historical_data['target'] = np.where(
                self.historical_data['close'].pct_change().shift(-1) > 0, 
                1, 0
            )
            
            # Drop NaN values
            ml_data = self.historical_data.dropna()
            
            logger.info("Technical indicators and labels prepared")
            return ml_data
        
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            raise

    def train_ml_model(self, test_size=0.2):
        """
        Train machine learning model
        
        Args:
            test_size (float): Proportion of data to use for testing
        """
        try:
            ml_data = self.prepare_ml_features()
            
            # Prepare features and target
            X = ml_data[['rsi', 'adx']]
            y = ml_data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Train RandomForest Classifier
            self.ml_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
            self.ml_model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = self.ml_model.predict(X_test)
            
            logger.info("Machine Learning Model Training Results:")
            logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
            logger.info("\nClassification Report:\n" + 
                        classification_report(y_test, y_pred))
            logger.info("\nConfusion Matrix:\n" + 
                        str(confusion_matrix(y_test, y_pred)))
            
            return self.ml_model
        
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            raise

    def _calculate_rsi(self, close, window=14):
        """Calculate RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_adx(self, high, low, close, window=14):
        """Calculate ADX"""
        tr = np.maximum(high - low, 
                        np.abs(high - close.shift(1)), 
                        np.abs(low - close.shift(1)))
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr_smooth = tr.rolling(window=window).sum()
        plus_dm_smooth = plus_dm.rolling(window=window).sum()
        minus_dm_smooth = minus_dm.rolling(window=window).sum()
        
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(window=window).mean()
        
        return adx

    def simulate_trading_strategy(self):
        """
        Simulate trading strategy using ML predictions
        """
        try:
            if self.ml_model is None:
                self.train_ml_model()
            
            # Predict on full dataset
            X_full = self.historical_data[['rsi', 'adx']].dropna()
            predictions = self.ml_model.predict(X_full)
            
            # Add predictions to dataframe
            self.historical_data.loc[X_full.index, 'ml_prediction'] = predictions
            
            # Basic trading simulation
            initial_capital = 10000
            position = 0
            portfolio_value = [initial_capital]
            
            for i in range(1, len(self.historical_data)):
                if self.historical_data['ml_prediction'].iloc[i-1] == 1 and position == 0:
                    # Buy signal
                    position = initial_capital / self.historical_data['close'].iloc[i-1]
                    initial_capital = 0
                elif self.historical_data['ml_prediction'].iloc[i-1] == 0 and position > 0:
                    # Sell signal
                    initial_capital = position * self.historical_data['close'].iloc[i-1]
                    position = 0
                
                # Track portfolio value
                current_value = (position * self.historical_data['close'].iloc[i]) if position > 0 else initial_capital
                portfolio_value.append(current_value)
            
            # Performance metrics
            total_return = (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] * 100
            max_drawdown = max((1 - current/peak) * 100 
                               for peak, current in zip(np.maximum.accumulate(portfolio_value), portfolio_value))
            
            logger.info("\nTrading Simulation Results:")
            logger.info(f"Total Return: {total_return:.2f}%")
            logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
            
            return {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'portfolio_values': portfolio_value
            }
        
        except Exception as e:
            logger.error(f"Error in trading simulation: {e}")
            raise

def main():
    # Run the simulation
    try:
        simulation = AIBacktestSimulation(symbol='BTC-USD')
        simulation.fetch_historical_data()
        simulation.train_ml_model()
        backtest_results = simulation.simulate_trading_strategy()
        logger.info("Simulation completed successfully!")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")

if __name__ == "__main__":
    main()
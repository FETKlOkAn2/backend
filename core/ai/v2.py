import sys
import os
import pandas as pd
import numpy as np
import gc
import logging
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import core.utils.utils as utils
import database.database_interaction as database_interaction
from core.risk import Risk_Handler
from core.hyper import Hyper
from core.log import LinkedList
from core.strategies.strategy import Strategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


class FeatureGenerator:
    """Generate technical indicators and features for ML models"""
    
    def __init__(self):
        """Initialize the feature generator"""
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        
    def validate_dataframe(self, df):
        """Validate the dataframe has required columns"""
        for col in self.required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe")
        return True
        
    def generate_features(self, df, window_sizes=[14, 20, 50], include_price_ratios=True, include_returns=True):
        """
        Generate technical indicators and features for ML model
        
        Args:
            df: DataFrame with OHLCV data
            window_sizes: List of window sizes for indicators
            include_price_ratios: Include price ratio features
            include_returns: Include return features
            
        Returns:
            DataFrame with added technical indicators
        """
        self.validate_dataframe(df)
        
        # Create a copy to avoid modifying the original
        features_df = df.copy()
        
        # Price features
        if include_price_ratios:
            features_df['high_low_ratio'] = features_df['high'] / features_df['low']
            features_df['close_open_ratio'] = features_df['close'] / features_df['open']
        
        # Return features  
        if include_returns:
            features_df['returns_1'] = features_df['close'].pct_change(1)
            features_df['returns_5'] = features_df['close'].pct_change(5)
            features_df['returns_10'] = features_df['close'].pct_change(10)
        
        # Moving averages
        for window in window_sizes:
            features_df[f'sma_{window}'] = features_df['close'].rolling(window=window).mean()
            features_df[f'ema_{window}'] = features_df['close'].ewm(span=window, adjust=False).mean()
        
        # Volatility indicators
        for window in window_sizes:
            features_df[f'std_{window}'] = features_df['close'].rolling(window=window).std()
            features_df[f'atr_{window}'] = (
                features_df['high'].rolling(window).max() - 
                features_df['low'].rolling(window).min()
            ) / features_df['close']
        
        # Volume indicators
        features_df['volume_ma_10'] = features_df['volume'].rolling(window=10).mean()
        features_df['volume_ma_30'] = features_df['volume'].rolling(window=30).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_10']
        
        # RSI
        delta = features_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        features_df['ema_12'] = features_df['close'].ewm(span=12, adjust=False).mean()
        features_df['ema_26'] = features_df['close'].ewm(span=26, adjust=False).mean()
        features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_hist'] = features_df['macd'] - features_df['macd_signal']
        
        # Bollinger Bands
        for window in window_sizes:
            features_df[f'bb_middle_{window}'] = features_df['close'].rolling(window=window).mean()
            features_df[f'bb_std_{window}'] = features_df['close'].rolling(window=window).std()
            features_df[f'bb_upper_{window}'] = features_df[f'bb_middle_{window}'] + (features_df[f'bb_std_{window}'] * 2)
            features_df[f'bb_lower_{window}'] = features_df[f'bb_middle_{window}'] - (features_df[f'bb_std_{window}'] * 2)
            features_df[f'bb_width_{window}'] = (features_df[f'bb_upper_{window}'] - features_df[f'bb_lower_{window}']) / features_df[f'bb_middle_{window}']
            
        # Add target features for ML prediction
        features_df['future_return_1'] = features_df['close'].pct_change(1).shift(-1)
        features_df['future_return_5'] = features_df['close'].pct_change(5).shift(-5)
        features_df['future_return_10'] = features_df['close'].pct_change(10).shift(-10)
        
        # Signal columns (will be filled by prediction or strategy)
        features_df['ml_signal'] = 0
        
        # Drop NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        
        return features_df


class AIModel:
    """Machine learning model for price prediction and signals"""
    
    def __init__(self, model_type='lstm', sequence_length=20, prediction_horizon=1):
        """
        Initialize AI model
        
        Args:
            model_type: Type of model ('lstm', 'cnn', 'mlp')
            sequence_length: Number of time steps for sequence models
            prediction_horizon: Steps ahead to predict
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = []
        self.target_column = f'future_return_{prediction_horizon}'
        
    def prepare_sequences(self, data, features, targets):
        """
        Prepare data as sequences for LSTM
        
        Args:
            data: DataFrame with features
            features: List of feature column names
            targets: List of target column names
            
        Returns:
            X_seq: Sequence data for features
            y_seq: Sequence data for targets
        """
        X = data[features].values
        y = data[targets].values
        
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            y_seq.append(y_scaled[i + self.sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape, output_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_shape: Number of output values
            
        Returns:
            Compiled model
        """
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(output_shape))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, data, features, target_col=None, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the model
        
        Args:
            data: DataFrame with features
            features: List of feature columns
            target_col: Target column name (default: None uses self.target_column)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if target_col is None:
            target_col = self.target_column
            
        self.feature_columns = features
        
        # Prepare data
        X_seq, y_seq = self.prepare_sequences(data, features, [target_col])
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Build and train model
        if self.model is None:
            input_shape = (self.sequence_length, len(features))
            output_shape = 1
            self.model = self.build_lstm_model(input_shape, output_shape)
            
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, data, features=None):
        """
        Generate predictions from the model
        
        Args:
            data: DataFrame with features
            features: List of feature columns (default: None uses self.feature_columns)
            
        Returns:
            Predictions array
        """
        if features is None:
            features = self.feature_columns
            
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Prepare sequences
        X = data[features].values
        X_scaled = self.feature_scaler.transform(X)
        
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            
        X_seq = np.array(X_seq)
        
        # Generate predictions
        y_pred_scaled = self.model.predict(X_seq)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame(index=data.index[self.sequence_length:])
        pred_df['predicted_return'] = y_pred.flatten()
        
        return pred_df
    
    def generate_signals(self, predictions, threshold=0.0):
        """
        Generate trading signals from predictions
        
        Args:
            predictions: DataFrame with predictions
            threshold: Return threshold for signal generation
            
        Returns:
            DataFrame with signals (1: buy, -1: sell, 0: hold)
        """
        signals = pd.DataFrame(index=predictions.index)
        signals['signal'] = 0
        
        # Generate signals based on predicted returns
        signals.loc[predictions['predicted_return'] > threshold, 'signal'] = 1
        signals.loc[predictions['predicted_return'] < -threshold, 'signal'] = -1
        
        return signals
    
    def save(self, filepath):
        """Save model to file"""
        if self.model is not None:
            save_model(self.model, filepath)
            # Save scalers
            scaler_data = {
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }
            pd.to_pickle(scaler_data, filepath + '_scalers.pkl')
            
    def load(self, filepath):
        """Load model from file"""
        self.model = load_model(filepath)
        # Load scalers
        scaler_data = pd.read_pickle(filepath + '_scalers.pkl')
        self.feature_scaler = scaler_data['feature_scaler']
        self.target_scaler = scaler_data['target_scaler']
        self.feature_columns = scaler_data['feature_columns']
        self.sequence_length = scaler_data['sequence_length']
        self.prediction_horizon = scaler_data['prediction_horizon']


class AI_Strategy(Strategy):
    """Strategy that incorporates AI predictions with technical indicators"""
    
    def __init__(self, dict_df, risk_object=None, with_sizing=True, ai_model=None, use_ai_signals=True, 
                 sequence_length=20, prediction_horizon=5, signal_threshold=0.01):
        """
        Initialize AI Strategy
        
        Args:
            dict_df: Dictionary with symbol: DataFrame pairs
            risk_object: Risk handler object
            with_sizing: Whether to use position sizing
            ai_model: Pre-trained AI model or None to create new
            use_ai_signals: Whether to use AI signals directly
            sequence_length: Length of sequences for AI model
            prediction_horizon: Prediction steps ahead
            signal_threshold: Threshold for signal generation
        """
        super().__init__(dict_df=dict_df, risk_object=risk_object, with_sizing=with_sizing)
        
        self.ai_model = ai_model or AIModel(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        self.feature_generator = FeatureGenerator()
        self.use_ai_signals = use_ai_signals
        self.signal_threshold = signal_threshold
        self.features_df = None
        self.predictions = None
        self.signals = None
        
    def prepare_features(self):
        """Prepare features for the model"""
        self.features_df = self.feature_generator.generate_features(self.df)
        return self.features_df
        
    def train_model(self, features=None, target_col=None, validation_split=0.2, epochs=50, batch_size=32):
        """Train the AI model"""
        if self.features_df is None:
            self.prepare_features()
            
        if features is None:
            # Select all numeric columns except the target and signal columns
            features = self.features_df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if not f.startswith('future_') and f != 'ml_signal']
            
        history = self.ai_model.fit(
            data=self.features_df,
            features=features,
            target_col=target_col,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def generate_predictions(self, features=None):
        """Generate predictions from the model"""
        if self.features_df is None:
            self.prepare_features()
            
        if features is None:
            features = self.ai_model.feature_columns
            
        self.predictions = self.ai_model.predict(self.features_df, features)
        return self.predictions
        
    def generate_ai_signals(self):
        """Generate trading signals from AI predictions"""
        if self.predictions is None:
            self.generate_predictions()
            
        self.signals = self.ai_model.generate_signals(self.predictions, threshold=self.signal_threshold)
        
        # Map signals to the original dataframe
        self.df['ml_signal'] = 0
        self.df.loc[self.signals.index, 'ml_signal'] = self.signals['signal']
        
        return self.signals
    
    def custom_indicator(self, data=None, ai_weight=0.7, rsi_weight=0.15, macd_weight=0.15, 
                        rsi_buy=30, rsi_sell=70, macd_fast=12, macd_slow=26, macd_signal=9):
        """
        Custom indicator combining AI signals with technical indicators
        
        Args:
            data: Optional data parameter
            ai_weight: Weight given to AI signals (0-1)
            rsi_weight: Weight given to RSI signals (0-1)
            macd_weight: Weight given to MACD signals (0-1)
            rsi_buy: RSI buy threshold (below this value)
            rsi_sell: RSI sell threshold (above this value)
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            
        Returns:
            Series with combined signals
        """
        if self.features_df is None:
            self.prepare_features()
            
        if self.use_ai_signals and self.signals is None:
            self.generate_ai_signals()
            
        # Calculate RSI signals
        rsi = self.features_df['rsi_14']
        rsi_signal = pd.Series(0, index=self.features_df.index)
        rsi_signal.loc[rsi < rsi_buy] = 1  # Buy signal
        rsi_signal.loc[rsi > rsi_sell] = -1  # Sell signal
        
        # Calculate MACD signals
        ema_fast = self.features_df['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = self.features_df['close'].ewm(span=macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal_line = macd.ewm(span=macd_signal, adjust=False).mean()
        macd_hist = macd - macd_signal_line
        
        macd_signal = pd.Series(0, index=self.features_df.index)
        macd_signal.loc[macd_hist > 0] = 1  # Buy signal
        macd_signal.loc[macd_hist < 0] = -1  # Sell signal
        
        # Combine signals with weighted approach
        combined_signal = pd.Series(0, index=self.features_df.index)
        
        if self.use_ai_signals:
            ai_signal = self.df['ml_signal'].loc[self.features_df.index]
            weighted_signal = (
                ai_weight * ai_signal +
                rsi_weight * rsi_signal +
                macd_weight * macd_signal
            )
        else:
            # Without AI, rebalance the weights
            adjusted_rsi_weight = rsi_weight / (rsi_weight + macd_weight)
            adjusted_macd_weight = macd_weight / (rsi_weight + macd_weight)
            
            weighted_signal = (
                adjusted_rsi_weight * rsi_signal +
                adjusted_macd_weight * macd_signal
            )
        
        # Convert weighted signals to discrete signals
        combined_signal.loc[weighted_signal > 0.3] = 1  # Buy threshold
        combined_signal.loc[weighted_signal < -0.3] = -1  # Sell threshold
        
        return combined_signal
    
    def save_model(self, filepath):
        """Save the AI model"""
        if self.ai_model is not None:
            self.ai_model.save(filepath)
            
    def load_model(self, filepath):
        """Load the AI model"""
        self.ai_model = AIModel()
        self.ai_model.load(filepath)


class Backtest_AI:
    """Enhanced backtesting class with AI and hyperparameter optimization"""
    
    def __init__(self):
        """Initialize the backtester"""
        self.symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD', 'LINK-USD', 'XLM-USD']
        self.granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY']
        self.product = ['XTZ-USD']
        self.granularity = 'ONE_MINUTE'
        
    def chunk_dataframe(self, df, chunk_size):
        """Split DataFrame into smaller chunks"""
        for start_idx in range(0, len(df), chunk_size):
            yield df.iloc[start_idx:start_idx + chunk_size]
            
    def get_training_data(self, symbol, granularity, num_days):
        """Get historical data for training"""
        dict_df = database_interaction.get_historical_from_db(
            granularity=granularity,
            symbols=symbol,
            num_days=num_days
        )
        
        if not dict_df or symbol not in dict_df:
            logging.warning(f"No data available for {symbol} at {granularity}")
            return None
            
        return dict_df
        
    def train_ai_model(self, symbol, granularity, num_days=60, sequence_length=20, 
                      prediction_horizon=5, epochs=50, batch_size=32):
        """
        Train an AI model on historical data
        
        Args:
            symbol: Trading pair symbol
            granularity: Time granularity
            num_days: Number of days of historical data
            sequence_length: Sequence length for LSTM
            prediction_horizon: Steps ahead to predict
            epochs: Training epochs
            batch_size: Training batch size
            
        Returns:
            Trained AI model
        """
        logging.info(f"Training AI model for {symbol} at {granularity}")
        
        # Get historical data
        dict_df = self.get_training_data(symbol, granularity, num_days)
        if not dict_df:
            return None
            
        # Initialize strategy with AI components
        ai_strat = AI_Strategy(
            dict_df=dict_df,
            use_ai_signals=True,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        # Prepare features
        ai_strat.prepare_features()
        
        # Train model
        history = ai_strat.train_model(
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save model
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{symbol}_{granularity}_model.h5")
        ai_strat.save_model(model_path)
        
        logging.info(f"Model saved to {model_path}")
        return ai_strat.ai_model
        
    def run_ai_backtest(self, symbol, granularity, num_days=30, sequence_length=20, prediction_horizon=5, 
                       with_sizing=True, use_trained_model=True, ai_weight=0.7):
        """
        Run backtest with AI strategy
        
        Args:
            symbol: Trading pair symbol
            granularity: Time granularity
            num_days: Number of days of historical data
            sequence_length: Sequence length for LSTM
            prediction_horizon: Steps ahead to predict
            with_sizing: Whether to use position sizing
            use_trained_model: Whether to use a pre-trained model
            ai_weight: Weight given to AI signals
            
        Returns:
            Dictionary with backtest results
        """
        logging.info(f"Running AI backtest for {symbol} at {granularity}")
        
        # Get historical data
        dict_df = self.get_training_data(symbol, granularity, num_days)
        if not dict_df:
            return {"error": "No data available"}
            
        # Load trained model if requested
        ai_model = None
        if use_trained_model:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 
                                     f"{symbol}_{granularity}_model.h5")
            
            if os.path.exists(model_path):
                logging.info(f"Loading model from {model_path}")
                ai_model = AIModel()
                ai_model.load(model_path)
            else:
                logging.warning(f"No pre-trained model found at {model_path}")
        
        # Initialize risk handler
        risk = Risk_Handler()
        
        # Initialize AI strategy
        ai_strat = AI_Strategy(
            dict_df=dict_df,
            risk_object=risk,
            with_sizing=with_sizing,
            ai_model=ai_model,
            use_ai_signals=True,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        # Prepare features
        ai_strat.prepare_features()
        
        # Train model if needed
        if ai_model is None:
            logging.info("Training new model for backtest")
            ai_strat.train_model(epochs=30, batch_size=32)
        
        # Generate signals
        ai_strat.generate_ai_signals()
        
        # Run backtest with custom indicator
        ai_strat.custom_indicator(ai_weight=ai_weight)
        ai_strat.generate_backtest()
        
        # Get portfolio statistics
        stats = ai_strat.portfolio.stats(silence_warnings=True).to_dict()
        
        logging.info(f"AI backtest completed for {symbol}")
        return stats
        
    def run_hybrid_optimization(self, symbol, granularity, num_days=30, sequence_length=20, 
                              prediction_horizon=5, with_sizing=True):
        """
        Run hyperparameter optimization for hybrid AI-technical strategy
        
        Args:
            symbol: Trading pair symbol
            granularity: Time granularity
            num_days: Number of days of historical data
            sequence_length: Sequence length for LSTM
            prediction_horizon: Steps ahead to predict
            with_sizing: Whether to use position sizing
            
        Returns:
            Dictionary with optimal parameters
        """
        logging.info(f"Running hybrid optimization for {symbol} at {granularity}")
        
        # Define parameter ranges
        param_ranges = {
            'ai_weight': np.linspace(0.3, 0.9, 4),
            'rsi_weight': np.linspace(0.05, 0.35, 4),
            'macd_weight': np.linspace(0.05, 0.35, 4),
            'rsi_buy': np.linspace(20, 40, 5),
            'rsi_sell': np.linspace(60, 80, 5),
            'macd_fast': np.array([8, 12, 16]),
            'macd_slow': np.array([21, 26, 30]),
            'macd_signal': np.array([7, 9, 11])
        }
        
        # Get historical data
        dict_df = self.get_training_data(symbol, granularity, num_days)
        if not dict_df:
            return {"error": "No data available"}
        
        # Initialize risk handler
        risk = Risk_Handler()
        
        # Check if we have a pretrained model
        model_path = os.path.join(os.path.dirname(__file__), 'models', 
                                 f"{symbol}_{granularity}_model.h5")
        
        ai_model = None
        if os.path.exists(model_path):
            logging.info(f"Loading model from {model_path}")
            ai_model = AIModel()
            ai_model.load(model_path)
        
        # Initialize AI strategy
        ai_strat = AI_Strategy(
            dict_df=dict_df,
            risk_object=risk,
            with_sizing=with_sizing,
            ai_model=ai_model,
            use_ai_signals=True,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        # Prepare features
        ai_strat.prepare_features()
        
        # Train model if needed
        if ai_model is None:
            logging.info("Training new model for optimization")
            ai_strat.train_model(epochs=30, batch_size=32)
        
        # Generate signals
        ai_strat.generate_ai_signals()
        
        # Run hyperparameter optimization
        hyper = Hyper(strategy_object=ai_strat, close=ai_strat.close, **param_ranges)
        
        # Export results to database
        database_interaction.export_hyper_to_db(strategy=ai_strat, hyper=hyper)
        
        # Get best parameters
        best_params = database_interaction.get_best_params(ai_strat)
        
        logging.info(f"Best hybrid parameters for {symbol}: {best_params}")
        return best_params
        
    def run_walk_forward_analysis(self, symbol, granularity, total_days=90, window_size=30, 
                                step_size=10, with_sizing=True):
        """
        Run walk-forward analysis with hybrid optimization
        
        Args:
            symbol: Trading pair symbol
            granularity: Time granularity
            total_days: Total days of historical data
            window_size: Size of each window in days
            step_size: Days to step forward
            with_sizing: Whether to use position sizing
            
        Returns:
            DataFrame with walk-forward results
        """
        logging.info(f"Running walk-forward analysis for {symbol} at {granularity}")
        
        # Get historical data
        dict_df = self.get_training_data(symbol, granularity, total_days)
        if not dict_df:
            return pd.DataFrame()
            
        df = dict_df[symbol]
        
        # Calculate window sizes in rows
        # Calculate window sizes in rows
        rows_per_day = 24 * 60 // utils.granularity_to_minutes(granularity)
        window_rows = window_size * rows_per_day
        step_rows = step_size * rows_per_day
        
        # Initialize results
        results = []
        
        # Walk forward
        for start_idx in range(0, len(df) - window_rows, step_rows):
            end_idx = start_idx + window_rows
            if end_idx > len(df):
                break
                
            window_df = df.iloc[start_idx:end_idx].copy()
            window_dict = {symbol: window_df}
            
            # Initialize risk handler
            risk = Risk_Handler()
            
            # Initialize AI strategy
            ai_strat = AI_Strategy(
                dict_df=window_dict,
                risk_object=risk,
                with_sizing=with_sizing,
                use_ai_signals=True
            )
            
            # Prepare features and train model
            ai_strat.prepare_features()
            ai_strat.train_model(epochs=30, batch_size=32)
            
            # Generate signals
            ai_strat.generate_ai_signals()
            
            # Run optimization
            param_ranges = {
                'ai_weight': np.linspace(0.3, 0.9, 3),
                'rsi_weight': np.linspace(0.05, 0.35, 3),
                'macd_weight': np.linspace(0.05, 0.35, 3),
                'rsi_buy': np.linspace(20, 40, 3),
                'rsi_sell': np.linspace(60, 80, 3)
            }
            
            hyper = Hyper(strategy_object=ai_strat, close=ai_strat.close, **param_ranges)
            best_params = hyper.best_params
            
            # Get out-of-sample window (validation)
            val_start_idx = end_idx
            val_end_idx = min(val_start_idx + step_rows, len(df))
            val_df = df.iloc[val_start_idx:val_end_idx].copy()
            val_dict = {symbol: val_df}
            
            # Run with optimized params on validation set
            val_ai_strat = AI_Strategy(
                dict_df=val_dict,
                risk_object=risk,
                with_sizing=with_sizing,
                ai_model=ai_strat.ai_model,  # Use trained model
                use_ai_signals=True
            )
            
            # Prepare features
            val_ai_strat.prepare_features()
            val_ai_strat.generate_ai_signals()
            
            # Run with best params
            val_ai_strat.custom_indicator(**best_params)
            val_ai_strat.generate_backtest()
            
            # Get stats
            stats = val_ai_strat.portfolio.stats(silence_warnings=True).to_dict()
            
            # Store results
            window_results = {
                'start_date': window_df.index[0],
                'end_date': window_df.index[-1],
                'val_start_date': val_df.index[0],
                'val_end_date': val_df.index[-1],
                'return': stats.get('Return [%]', 0),
                'sharpe': stats.get('Sharpe Ratio', 0),
                'sortino': stats.get('Sortino Ratio', 0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0),
                'win_rate': stats.get('Win Rate [%]', 0),
                'params': best_params
            }
            
            results.append(window_results)
            
        # Compile results
        results_df = pd.DataFrame(results)
        
        logging.info(f"Walk-forward analysis completed for {symbol}")
        return results_df
    
    def train_multiple_symbols(self, symbols=None, granularities=None, num_days=60):
        """Train AI models for multiple symbols and granularities"""
        if symbols is None:
            symbols = self.symbols[:5]  # Use first 5 symbols by default
            
        if granularities is None:
            granularities = ['ONE_HOUR', 'FOUR_HOUR', 'ONE_DAY']
            
        results = {}
        
        for symbol in symbols:
            symbol_results = {}
            for granularity in granularities:
                try:
                    model = self.train_ai_model(symbol, granularity, num_days=num_days)
                    if model is not None:
                        symbol_results[granularity] = "Success"
                    else:
                        symbol_results[granularity] = "Failed"
                except Exception as e:
                    logging.error(f"Error training model for {symbol} {granularity}: {str(e)}")
                    symbol_results[granularity] = f"Error: {str(e)}"
                    
            results[symbol] = symbol_results
            
        return results
        

class Performance_Monitor:
    """Monitor and analyze AI strategy performance"""
    
    def __init__(self):
        """Initialize the performance monitor"""
        self.backtest_results = {}
        self.live_performance = {}
        self.model_accuracy = {}
        
    def compare_strategies(self, symbol, granularity, num_days=30):
        """Compare AI strategy with traditional strategies"""
        # Get historical data
        dict_df = database_interaction.get_historical_from_db(
            granularity=granularity,
            symbols=[symbol],
            num_days=num_days
        )
        
        if not dict_df or symbol not in dict_df:
            logging.warning(f"No data available for {symbol} at {granularity}")
            return None
            
        # Initialize risk handler
        risk = Risk_Handler()
        
        # Initialize AI strategy
        ai_strat = AI_Strategy(
            dict_df=dict_df,
            risk_object=risk,
            with_sizing=True,
            use_ai_signals=True
        )
        
        # Prepare features and train model
        ai_strat.prepare_features()
        ai_strat.train_model(epochs=30, batch_size=32)
        ai_strat.generate_ai_signals()
        ai_strat.generate_backtest()
        
        # Get AI strategy stats
        ai_stats = ai_strat.portfolio.stats(silence_warnings=True).to_dict()
        
        # Initialize technical strategy (without AI)
        tech_strat = AI_Strategy(
            dict_df=dict_df,
            risk_object=risk,
            with_sizing=True,
            use_ai_signals=False
        )
        
        # Run technical strategy
        tech_strat.prepare_features()
        tech_strat.custom_indicator(ai_weight=0)  # No AI weight
        tech_strat.generate_backtest()
        
        # Get technical strategy stats
        tech_stats = tech_strat.portfolio.stats(silence_warnings=True).to_dict()
        
        # Compare results
        comparison = {
            'symbol': symbol,
            'granularity': granularity,
            'ai_return': ai_stats.get('Return [%]', 0),
            'tech_return': tech_stats.get('Return [%]', 0),
            'ai_sharpe': ai_stats.get('Sharpe Ratio', 0),
            'tech_sharpe': tech_stats.get('Sharpe Ratio', 0),
            'ai_sortino': ai_stats.get('Sortino Ratio', 0),
            'tech_sortino': tech_stats.get('Sortino Ratio', 0),
            'ai_max_drawdown': ai_stats.get('Max. Drawdown [%]', 0),
            'tech_max_drawdown': tech_stats.get('Max. Drawdown [%]', 0),
            'ai_win_rate': ai_stats.get('Win Rate [%]', 0),
            'tech_win_rate': tech_stats.get('Win Rate [%]', 0)
        }
        
        return comparison
        
    def evaluate_model_accuracy(self, symbol, granularity, num_days=30):
        """Evaluate predictive accuracy of the AI model"""
        # Get historical data
        dict_df = database_interaction.get_historical_from_db(
            granularity=granularity,
            symbols=[symbol],
            num_days=num_days
        )
        
        if not dict_df or symbol not in dict_df:
            logging.warning(f"No data available for {symbol} at {granularity}")
            return None
            
        # Initialize AI strategy
        ai_strat = AI_Strategy(
            dict_df=dict_df,
            with_sizing=False,
            use_ai_signals=True
        )
        
        # Prepare features and train model
        ai_strat.prepare_features()
        ai_strat.train_model(epochs=30, batch_size=32)
        
        # Generate predictions
        predictions = ai_strat.generate_predictions()
        
        # Compare predictions with actual returns
        features_df = ai_strat.features_df
        
        # Align predictions with actual future returns
        merged = pd.merge(
            predictions,
            features_df[['future_return_1', 'future_return_5', 'future_return_10']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Calculate accuracy metrics
        horizon = ai_strat.ai_model.prediction_horizon
        actual_column = f'future_return_{horizon}'
        
        # Direction accuracy (binary classification)
        merged['pred_direction'] = merged['predicted_return'] > 0
        merged['actual_direction'] = merged[actual_column] > 0
        direction_accuracy = (merged['pred_direction'] == merged['actual_direction']).mean()
        
        # Mean Absolute Error
        mae = np.abs(merged['predicted_return'] - merged[actual_column]).mean()
        
        # RMSE
        rmse = np.sqrt(((merged['predicted_return'] - merged[actual_column]) ** 2).mean())
        
        # Correlation
        correlation = merged['predicted_return'].corr(merged[actual_column])
        
        accuracy_metrics = {
            'symbol': symbol,
            'granularity': granularity,
            'direction_accuracy': direction_accuracy,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation
        }
        
        # Store results
        self.model_accuracy[(symbol, granularity)] = accuracy_metrics
        
        return accuracy_metrics
        
    def generate_performance_report(self, symbols=None, granularities=None):
        """Generate comprehensive performance report"""
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD']
            
        if granularities is None:
            granularities = ['ONE_HOUR', 'ONE_DAY']
            
        report_data = []
        
        for symbol in symbols:
            for granularity in granularities:
                # Get strategy comparison
                comparison = self.compare_strategies(symbol, granularity)
                
                # Get model accuracy
                accuracy = self.evaluate_model_accuracy(symbol, granularity)
                
                if comparison is not None and accuracy is not None:
                    report_entry = {
                        'symbol': symbol,
                        'granularity': granularity,
                        'ai_return': comparison['ai_return'],
                        'tech_return': comparison['tech_return'],
                        'return_diff': comparison['ai_return'] - comparison['tech_return'],
                        'ai_sharpe': comparison['ai_sharpe'],
                        'tech_sharpe': comparison['tech_sharpe'],
                        'direction_accuracy': accuracy['direction_accuracy'],
                        'correlation': accuracy['correlation']
                    }
                    
                    report_data.append(report_entry)
                    
        # Create report DataFrame
        report_df = pd.DataFrame(report_data)
        
        return report_df


if __name__ == "__main__":
    # Track memory usage
    tracemalloc.start()
    
    # Create backtest instance
    backtest = Backtest_AI()
    
    # Define symbols and granularities to test
    test_symbols = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SHIB-USD', 'AVAX-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD', 'AAVE-USD', 'XTZ-USD', 'COMP-USD']
    test_granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY']
    
    # Train models
    print("Training AI models...")
    training_results = backtest.train_multiple_symbols(
        symbols=test_symbols,
        granularities=test_granularities,
        num_days=60
    )
    print(f"Training results: {training_results}")
    
    # Run backtests
    print("Running backtests...")
    backtest_results = {}
    for symbol in test_symbols:
        symbol_results = {}
        for granularity in test_granularities:
            results = backtest.run_ai_backtest(
                symbol=symbol,
                granularity=granularity,
                num_days=30,
                use_trained_model=True
            )
            symbol_results[granularity] = results
        backtest_results[symbol] = symbol_results
    
    print("Backtest results:")
    for symbol, gran_results in backtest_results.items():
        print(f"\n{symbol}:")
        for gran, results in gran_results.items():
            if 'Return [%]' in results:
                print(f"  {gran}: Return = {results['Return [%]']:.2f}%, Sharpe = {results.get('Sharpe Ratio', 0):.2f}")
            else:
                print(f"  {gran}: {results}")
    
    # Run optimization for BTC-USD
    print("\nRunning hybrid optimization for BTC-USD...")
    btc_params = backtest.run_hybrid_optimization(
        symbol='BTC-USD',
        granularity='ONE_HOUR',
        num_days=30
    )
    print(f"Best parameters: {btc_params}")
    
    # Create performance monitor
    print("\nGenerating performance report...")
    monitor = Performance_Monitor()
    report = monitor.generate_performance_report(
        symbols=test_symbols,
        granularities=['ONE_HOUR']
    )
    
    print("\nPerformance Report:")
    print(report)
    
    # Memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import talib as ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json

# Import your existing modules
from core.strategies.strategy import Strategy
from core.risk import Risk_Handler
import database.database_interaction as database_interaction
from core.backtest import Backtest

class IntelligentBuyHold(Strategy):
    """
    Intelligent Buy and Hold Strategy with ML-enhanced timing and portfolio management
    
    Features:
    - Dollar Cost Averaging with intelligent timing
    - Multi-asset portfolio management
    - ML-driven market condition assessment
    - Dynamic position sizing based on volatility
    - Automated rebalancing
    """
    
    def __init__(self, dict_df, risk_object=None, with_sizing=True, initial_capital=3000):
        super().__init__(dict_df, risk_object, with_sizing)
        
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.positions = {}  # Track current positions
        self.target_allocations = {}  # Target allocation percentages
        self.ml_models = {}  # Store ML models for each asset
        self.market_conditions = {}  # Store market condition assessments
        
        # Strategy parameters
        self.dca_frequency_days = 7  # DCA every 7 days by default
        self.rebalance_threshold = 0.15  # Rebalance if allocation differs by 15%
        self.volatility_lookback = 30  # Days to look back for volatility calculation
        self.momentum_lookback = 14  # Days for momentum calculation
        
        # ML feature parameters
        self.feature_lookback = 60  # Days of features for ML
        self.min_data_points = 100  # Minimum data points before ML predictions
        
        # Initialize components
        self._initialize_portfolio()
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the strategy"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"BuyHold_{self.symbol}")
        
    def _initialize_portfolio(self):
        """Initialize portfolio with target allocations"""
        # Default allocation strategy (can be customized)
        if self.symbol in ['BTC-USD']:
            self.target_allocations[self.symbol] = 0.4  # 40% BTC
        elif self.symbol in ['ETH-USD']:
            self.target_allocations[self.symbol] = 0.3  # 30% ETH
        elif self.symbol in ['LINK-USD', 'AVAX-USD', 'UNI-USD']:
            self.target_allocations[self.symbol] = 0.1  # 10% each for quality alts
        else:
            self.target_allocations[self.symbol] = 0.05  # 5% for others
            
        self.positions[self.symbol] = {
            'quantity': 0.0,
            'avg_cost': 0.0,
            'total_invested': 0.0,
            'last_purchase': None
        }

    def calculate_technical_features(self) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for ML model"""
        features = pd.DataFrame(index=self.close.index)
        
        # Price-based features
        features['close'] = self.close
        features['returns'] = self.close.pct_change()
        features['log_returns'] = np.log(self.close / self.close.shift(1))
        
        # Moving averages and trends
        features['sma_10'] = ta.SMA(self.close, 10)
        features['sma_20'] = ta.SMA(self.close, 20)
        features['sma_50'] = ta.SMA(self.close, 50)
        features['ema_12'] = ta.EMA(self.close, 12)
        features['ema_26'] = ta.EMA(self.close, 26)
        
        # Trend strength
        features['price_sma20_ratio'] = self.close / features['sma_20']
        features['sma10_sma20_ratio'] = features['sma_10'] / features['sma_20']
        features['ema12_ema26_ratio'] = features['ema_12'] / features['ema_26']
        
        # Volatility indicators
        features['volatility_20'] = self.close.rolling(20).std()
        features['atr'] = ta.ATR(self.high, self.low, self.close, 14)
        
        # Momentum indicators
        features['rsi'] = ta.RSI(self.close, 14)
        features['macd'], features['macd_signal'], features['macd_hist'] = ta.MACD(self.close)
        features['adx'] = ta.ADX(self.high, self.low, self.close, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(self.close, 20, 2, 2)
        features['bb_position'] = (self.close - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Volume indicators (if available)
        if hasattr(self, 'volume') and self.volume is not None:
            features['volume'] = self.volume
            features['volume_sma'] = ta.SMA(self.volume, 20)
            features['volume_ratio'] = self.volume / features['volume_sma']
            
        # Market structure
        features['higher_high'] = (self.high > self.high.shift(1)).astype(int)
        features['higher_low'] = (self.low > self.low.shift(1)).astype(int)
        features['bull_structure'] = (features['higher_high'] & features['higher_low']).astype(int)
        
        return features.dropna()

    def assess_market_condition(self) -> Dict:
        """Assess current market conditions using multiple indicators"""
        condition = {
            'trend': 'neutral',
            'strength': 0.5,
            'volatility': 'medium',
            'momentum': 'neutral',
            'buy_opportunity': False,
            'confidence': 0.5
        }
        
        if len(self.close) < 50:
            return condition
            
        # Trend assessment
        sma_20 = ta.SMA(self.close, 20)
        sma_50 = ta.SMA(self.close, 50)
        current_price = self.close.iloc[-1]
        
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            condition['trend'] = 'bullish'
            condition['strength'] = min(0.9, 0.5 + (current_price / sma_20.iloc[-1] - 1) * 2)
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            condition['trend'] = 'bearish'
            condition['strength'] = max(0.1, 0.5 - (sma_20.iloc[-1] / current_price - 1) * 2)
            
        # Volatility assessment
        volatility = self.close.pct_change().rolling(20).std().iloc[-1]
        vol_percentile = (self.close.pct_change().rolling(20).std().rank(pct=True)).iloc[-1]
        
        if vol_percentile > 0.8:
            condition['volatility'] = 'high'
        elif vol_percentile < 0.2:
            condition['volatility'] = 'low'
            
        # Momentum assessment
        rsi = ta.RSI(self.close, 14).iloc[-1]
        if rsi < 30:
            condition['momentum'] = 'oversold'
            condition['buy_opportunity'] = True
        elif rsi > 70:
            condition['momentum'] = 'overbought'
        elif 40 <= rsi <= 60:
            condition['momentum'] = 'neutral'
            
        # Overall confidence
        factors = []
        if condition['trend'] == 'bullish':
            factors.append(0.3)
        elif condition['trend'] == 'bearish':
            factors.append(-0.2)
            
        if condition['momentum'] == 'oversold':
            factors.append(0.4)
        elif condition['momentum'] == 'overbought':
            factors.append(-0.3)
            
        if condition['volatility'] == 'high':
            factors.append(0.2)  # High volatility can be opportunity
            
        condition['confidence'] = max(0.1, min(0.9, 0.5 + sum(factors)))
        
        return condition

    def train_ml_model(self) -> bool:
        """Train ML model for price prediction and timing optimization"""
        try:
            features_df = self.calculate_technical_features()
            
            if len(features_df) < self.min_data_points:
                self.logger.warning(f"Insufficient data for ML training: {len(features_df)} points")
                return False
                
            # Prepare features
            feature_cols = [col for col in features_df.columns if col not in ['close', 'returns', 'log_returns']]
            X = features_df[feature_cols].dropna()
            
            # Create targets for different time horizons
            y_1d = (features_df['close'].shift(-1) > features_df['close']).astype(int)  # Next day direction
            y_7d = (features_df['close'].shift(-7) > features_df['close']).astype(int)  # Weekly direction
            y_30d = (features_df['close'].shift(-30) > features_df['close']).astype(int)  # Monthly direction
            
            # Align data
            valid_idx = X.index.intersection(y_1d.dropna().index)
            valid_idx = valid_idx.intersection(y_7d.dropna().index)
            valid_idx = valid_idx.intersection(y_30d.dropna().index)
            
            if len(valid_idx) < 50:
                self.logger.warning("Insufficient aligned data for ML training")
                return False
                
            X_clean = X.loc[valid_idx]
            y_1d_clean = y_1d.loc[valid_idx]
            y_7d_clean = y_7d.loc[valid_idx]
            y_30d_clean = y_30d.loc[valid_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Train models for different time horizons
            models = {}
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            for horizon, y in [('1d', y_1d_clean), ('7d', y_7d_clean), ('30d', y_30d_clean)]:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # Simple train on all available data (you could add cross-validation here)
                model.fit(X_scaled, y)
                models[horizon] = {'model': model, 'scaler': scaler, 'features': feature_cols}
                
            self.ml_models[self.symbol] = models
            self.logger.info(f"ML models trained successfully for {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML model: {str(e)}")
            return False

    def get_ml_prediction(self) -> Dict:
        """Get ML-based market predictions"""
        if self.symbol not in self.ml_models:
            return {'1d': 0.5, '7d': 0.5, '30d': 0.5, 'confidence': 0.0}
            
        try:
            features_df = self.calculate_technical_features()
            if len(features_df) == 0:
                return {'1d': 0.5, '7d': 0.5, '30d': 0.5, 'confidence': 0.0}
                
            predictions = {}
            models = self.ml_models[self.symbol]
            
            for horizon in ['1d', '7d', '30d']:
                if horizon in models:
                    model_data = models[horizon]
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_cols = model_data['features']
                    
                    # Get latest features
                    latest_features = features_df[feature_cols].iloc[-1:].dropna()
                    if len(latest_features) == 0:
                        predictions[horizon] = 0.5
                        continue
                        
                    # Scale and predict
                    X_scaled = scaler.transform(latest_features)
                    prob = model.predict_proba(X_scaled)[0]
                    predictions[horizon] = prob[1] if len(prob) > 1 else 0.5
                else:
                    predictions[horizon] = 0.5
                    
            # Calculate overall confidence based on prediction consistency
            pred_values = list(predictions.values())
            confidence = 1.0 - np.std(pred_values) if len(pred_values) > 1 else 0.5
            predictions['confidence'] = confidence
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting ML prediction: {str(e)}")
            return {'1d': 0.5, '7d': 0.5, '30d': 0.5, 'confidence': 0.0}

    def calculate_position_size(self, available_capital: float, market_condition: Dict, ml_prediction: Dict) -> float:
        """Calculate optimal position size based on market conditions and ML predictions"""
        
        # Base position size (regular DCA amount)
        base_size = available_capital * 0.1  # 10% of available capital
        
        # Adjust based on market conditions
        condition_multiplier = 1.0
        
        # More aggressive during oversold conditions
        if market_condition['momentum'] == 'oversold':
            condition_multiplier *= 1.5
        elif market_condition['momentum'] == 'overbought':
            condition_multiplier *= 0.5
            
        # Adjust based on trend
        if market_condition['trend'] == 'bullish':
            condition_multiplier *= 1.2
        elif market_condition['trend'] == 'bearish':
            condition_multiplier *= 0.8
            
        # Adjust based on ML predictions
        ml_multiplier = 1.0
        if ml_prediction['confidence'] > 0.7:
            # High confidence predictions
            avg_prediction = (ml_prediction['1d'] + ml_prediction['7d'] + ml_prediction['30d']) / 3
            if avg_prediction > 0.6:
                ml_multiplier *= 1.3
            elif avg_prediction < 0.4:
                ml_multiplier *= 0.7
                
        # Volatility adjustment
        if market_condition['volatility'] == 'high':
            vol_multiplier = 0.8  # Reduce size during high volatility
        else:
            vol_multiplier = 1.0
            
        # Calculate final position size
        position_size = base_size * condition_multiplier * ml_multiplier * vol_multiplier
        
        # Ensure we don't exceed available capital
        position_size = min(position_size, available_capital * 0.3)  # Max 30% in single purchase
        
        return max(10, position_size)  # Minimum $10 purchase

    def should_buy_now(self) -> Tuple[bool, str]:
        """Determine if we should buy now based on multiple factors"""
        
        # Check if we have available capital
        if self.available_capital < 50:  # Minimum $50 for purchases
            return False, "Insufficient capital"
            
        # Get market assessment
        market_condition = self.assess_market_condition()
        ml_prediction = self.get_ml_prediction()
        
        # Check last purchase date (enforce minimum time between purchases)
        last_purchase = self.positions[self.symbol]['last_purchase']
        if last_purchase:
            days_since_last = (datetime.now() - last_purchase).days
            if days_since_last < self.dca_frequency_days:
                return False, f"Last purchase was {days_since_last} days ago"
                
        # Decision logic
        buy_signals = []
        
        # Market condition signals
        if market_condition['buy_opportunity']:
            buy_signals.append("Oversold condition")
            
        if market_condition['trend'] == 'bullish' and market_condition['strength'] > 0.7:
            buy_signals.append("Strong bullish trend")
            
        # ML signals
        if ml_prediction['confidence'] > 0.6:
            avg_pred = (ml_prediction['1d'] + ml_prediction['7d'] + ml_prediction['30d']) / 3
            if avg_pred > 0.65:
                buy_signals.append("ML prediction bullish")
                
        # Volatility opportunity
        if market_condition['volatility'] == 'high' and market_condition['momentum'] == 'oversold':
            buy_signals.append("High volatility buying opportunity")
            
        # Force DCA if it's been too long
        if last_purchase is None or (datetime.now() - last_purchase).days >= self.dca_frequency_days * 2:
            buy_signals.append("Regular DCA interval")
            
        # Decision
        should_buy = len(buy_signals) >= 2 or "Regular DCA interval" in buy_signals
        reason = "; ".join(buy_signals) if buy_signals else "No buy signals"
        
        return should_buy, reason

    def execute_buy(self, amount: float) -> bool:
        """Execute buy order (simulation for backtesting, real execution for live trading)"""
        try:
            current_price = self.close.iloc[-1]
            quantity = amount / current_price
            
            # Update position
            pos = self.positions[self.symbol]
            new_total_invested = pos['total_invested'] + amount
            new_quantity = pos['quantity'] + quantity
            new_avg_cost = new_total_invested / new_quantity if new_quantity > 0 else current_price
            
            self.positions[self.symbol] = {
                'quantity': new_quantity,
                'avg_cost': new_avg_cost,
                'total_invested': new_total_invested,
                'last_purchase': datetime.now()
            }
            
            # Update available capital
            self.available_capital -= amount
            
            self.logger.info(f"BUY: ${amount:.2f} of {self.symbol} at ${current_price:.2f}")
            self.logger.info(f"Position: {new_quantity:.6f} units, Avg cost: ${new_avg_cost:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing buy: {str(e)}")
            return False

    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        current_price = self.close.iloc[-1]
        position_value = self.positions[self.symbol]['quantity'] * current_price
        return self.available_capital + position_value

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        current_value = self.get_portfolio_value()
        total_return = (current_value / self.initial_capital - 1) * 100
        
        pos = self.positions[self.symbol]
        unrealized_pnl = 0
        if pos['quantity'] > 0:
            current_price = self.close.iloc[-1]
            unrealized_pnl = (current_price - pos['avg_cost']) * pos['quantity']
            
        return {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'available_capital': self.available_capital,
            'total_invested': pos['total_invested'],
            'unrealized_pnl': unrealized_pnl,
            'total_return_pct': total_return,
            'position_quantity': pos['quantity'],
            'avg_cost': pos['avg_cost'],
            'current_price': self.close.iloc[-1]
        }

    def custom_indicator(self, close=None, *args, **kwargs):
        """Main strategy logic - called by the backtesting framework"""
        try:
            # Train ML model if we have enough data
            if len(self.close) >= self.min_data_points:
                self.train_ml_model()
                
            # Initialize signals arrays
            signals = np.zeros(len(self.close))
            
            # Simulate the strategy day by day
            for i in range(max(60, self.min_data_points), len(self.close)):
                # Use data up to current point
                current_data = {
                    key: getattr(self, key).iloc[:i+1] for key in ['open', 'high', 'low', 'close', 'volume']
                }
                
                # Create temporary strategy instance for current point
                temp_strategy = type(self)(
                    dict_df={self.symbol: pd.DataFrame(current_data, index=self.close.index[:i+1])},
                    risk_object=self.risk_object,
                    with_sizing=self.with_sizing,
                    initial_capital=self.initial_capital
                )
                
                # Check if we should buy
                should_buy, reason = temp_strategy.should_buy_now()
                
                if should_buy:
                    # Calculate position size
                    market_condition = temp_strategy.assess_market_condition()
                    ml_prediction = temp_strategy.get_ml_prediction()
                    
                    if temp_strategy.available_capital >= 50:  # Minimum buy amount
                        amount = temp_strategy.calculate_position_size(
                            temp_strategy.available_capital,
                            market_condition,
                            ml_prediction
                        )
                        
                        if temp_strategy.execute_buy(amount):
                            signals[i] = 1  # Buy signal
                            
            # Set strategy signals for backtesting framework
            self.signals = signals
            self.entries = signals == 1
            self.exits = np.zeros_like(signals, dtype=bool)  # Buy and hold - no sells
            
            # Store additional data for analysis
            self.market_conditions[self.symbol] = self.assess_market_condition()
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in custom_indicator: {str(e)}")
            # Return default signals if error occurs
            signals = np.zeros(len(self.close))
            self.signals = signals
            self.entries = signals == 1
            self.exits = np.zeros_like(signals, dtype=bool)
            return signals


class BuyHoldBacktester(Backtest):
    """Enhanced backtester specifically for buy-and-hold strategies"""
    
    def __init__(self):
        super().__init__()
        self.symbols = ['BTC-USD', 'ETH-USD', 'LINK-USD', 'AVAX-USD', 'UNI-USD', 'DOGE-USD']
        self.granularities = ['ONE_DAY', 'ONE_HOUR', 'SIX_HOUR']
        
    def run_buy_hold_backtest(self, symbols: List[str], granularity: str, num_days: int, 
                             initial_capital: float = 3000, graph_callback=None):
        """Run comprehensive buy-and-hold backtest across multiple assets"""
        
        results = {}
        
        for symbol in symbols:
            try:
                print(f"\nRunning Buy-Hold backtest for {symbol}...")
                
                # Get historical data
                dict_df = database_interaction.get_historical_from_db(
                    granularity=granularity,
                    symbols=symbol,
                    num_days=num_days
                )
                
                if not dict_df or symbol not in dict_df:
                    print(f"No data available for {symbol}")
                    continue
                    
                # Initialize strategy
                current_dict = {symbol: dict_df[symbol]}
                risk = Risk_Handler()
                
                strategy = IntelligentBuyHold(
                    dict_df=current_dict,
                    risk_object=risk,
                    with_sizing=True,
                    initial_capital=initial_capital
                )
                
                # Run strategy
                strategy.custom_indicator()
                
                # Generate backtest
                strategy.generate_backtest()
                
                # Get results
                stats = strategy.portfolio.stats(silence_warnings=True).to_dict()
                performance = strategy.get_performance_metrics()
                
                results[symbol] = {
                    'stats': stats,
                    'performance': performance,
                    'strategy': strategy
                }
                
                # Print key metrics
                print(f"Results for {symbol}:")
                print(f"  Total Return: {stats.get('Total Return [%]', 0):.2f}%")
                print(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.2f}")
                print(f"  Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%")
                print(f"  Total Trades: {stats.get('Total Trades', 0)}")
                
                # Generate graph if callback provided
                if graph_callback:
                    try:
                        fig = strategy.graph(graph_callback)
                        if fig:
                            print(f"Graph generated for {symbol}")
                    except Exception as e:
                        print(f"Error generating graph for {symbol}: {e}")
                        
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
                
        return results
    
    def compare_strategies(self, results: Dict):
        """Compare performance across different assets"""
        comparison = []
        
        for symbol, data in results.items():
            stats = data['stats']
            comparison.append({
                'Symbol': symbol,
                'Total Return (%)': stats.get('Total Return [%]', 0),
                'Sharpe Ratio': stats.get('Sharpe Ratio', 0),
                'Max Drawdown (%)': stats.get('Max Drawdown [%]', 0),
                'Win Rate (%)': stats.get('Win Rate [%]', 0),
                'Total Trades': stats.get('Total Trades', 0)
            })
            
        df = pd.DataFrame(comparison)
        df = df.sort_values('Total Return (%)', ascending=False)
        
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        
        return df


# Usage example and testing functions
def test_buy_hold_strategy():
    """Test the buy-and-hold strategy"""
    
    # Initialize backtester
    backtester = BuyHoldBacktester()
    
    # Test parameters
    symbols = ['BTC-USD', 'ETH-USD', 'LINK-USD']
    granularity = 'ONE_DAY'
    num_days = 365  # 1 year backtest
    initial_capital = 3000
    
    print("Starting Intelligent Buy-Hold Strategy Backtest...")
    print(f"Initial Capital: ${initial_capital}")
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {num_days} days on {granularity}")
    print("="*60)
    
    # Run backtest
    results = backtester.run_buy_hold_backtest(
        symbols=symbols,
        granularity=granularity,
        num_days=num_days,
        initial_capital=initial_capital
    )
    
    # Compare results
    if results:
        comparison = backtester.compare_strategies(results)
        
        # Calculate portfolio metrics
        total_return = sum(data['stats'].get('Total Return [%]', 0) for data in results.values()) / len(results)
        best_performer = max(results.keys(), key=lambda x: results[x]['stats'].get('Total Return [%]', 0))
        
        print(f"\nBest Performing Asset: {best_performer}")
        print(f"Average Return Across Assets: {total_return:.2f}%")
        
        return results, comparison
    else:
        print("No results generated")
        return None, None


if __name__ == "__main__":
    test_buy_hold_strategy()
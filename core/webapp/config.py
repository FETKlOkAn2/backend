import os
import datetime

class Config:
    # App configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key')  # In production, set this from environment
    
    # JWT configuration
    JWT_EXPIRATION = datetime.timedelta(hours=1)
    
    # Trading configuration
    TRADING_DEFAULT_SIZING = 1000  # Default position sizing
    
    # Backtest configuration
    BACKTEST_DEFAULT_DAYS = 30  # Default number of days for backtesting
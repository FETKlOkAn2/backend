# Coinbase Trader

A cryptocurrency trading platform with technical analysis, backtesting, and AI-powered strategy optimization.

## Setup

1. Copy `.env.example` to `.env` and fill in your Coinbase API credentials
2. Make sure the DATABASE_PATH environment variable is set properly
3. Install dependencies: `pip install -r requirements.txt`

## Database Initialization

Before running backtests, you need to fetch historical data:

```bash
python -m core.coinbase_wrapper
```

This will create database files in the `database` directory.

## Running AI Backtest Optimizer

```bash
python -m core.ai.ai_backtest_optimizer --symbol ETH-USD --granularity ONE_HOUR --days 60 --trials 10 --save --plot
```

Parameters:
- `--symbol`: The trading pair to analyze (default: BTC-USD)
- `--granularity`: Timeframe (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, etc.)
- `--days`: Number of days of historical data to use
- `--trials`: Number of optimization trials to run
- `--save`: Save the model and parameters
- `--plot`: Generate and save performance charts

## Troubleshooting

If you get "No data found" errors, make sure:
1. You've run the coinbase_wrapper module to fetch historical data
2. The DATABASE_PATH in .env points to the correct directory
3. There's enough historical data for the symbol and granularity you requested
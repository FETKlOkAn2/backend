from core.backtest import Backtest
from core.strategies.single.rsi import RSI
from core.strategies.gpu_optimized.NP.rsi_adx_np import RSI_ADX_NP
from core.strategies.gpu_optimized.GPU.rsi_adx_gpu import RSI_ADX_GPU
from core.strategies.gpu_optimized.NP.rsi_bollinger_np import BollingerBands_RSI
from core.strategies.gpu_optimized.NP.bollinger_vwap import BollingerBands_VWAP
from core.strategies.gpu_optimized.NP.macd_atr_np import MACD_ATR
from core.strategies.single.bollinger import BollingerBands
from core.strategies.single.macd import MACD
from core.strategies.single.adx import ADX
from core.strategies.single.vwap import Vwap
from core.strategies.single.atr import ATR
from core.strategies.single.stochastic import StochasticOscillator
from core.strategies.single.efratio import EFratio
from core.strategies.single.williams import WilliamsR
from core.strategies.single.kama import Kama

# Strategy mapping dictionary
STRATEGY_MAP = { 
    "RSI": RSI,
    "RSI_ADX_NP": RSI_ADX_NP,
    "RSI_ADX_GPU": RSI_ADX_GPU,
    "BollingerBands_RSI": BollingerBands_RSI,
    "BollingerBands_VWAP": BollingerBands_VWAP,
    "MACD_ATR": MACD_ATR,
    "BollingerBands": BollingerBands,
    "MACD": MACD,
    "ADX": ADX,
    "Vwap": Vwap,
    "ATR": ATR,
    "StochasticOscillator": StochasticOscillator,
    "EFratio": EFratio,
    "WilliamsR": WilliamsR,
    "Kama": Kama
}


def get_strategy_class(strategy_name):
    """Get the strategy class from the strategy name."""
    return STRATEGY_MAP.get(strategy_name)

def run_backtest(symbol, granularity, strategy_obj, num_days, sizing, best_params, graph_callback):
    """Run a backtest with the provided parameters."""
    if not isinstance(symbol, list):
        symbol = [symbol]
        
    backtest_instance = Backtest()
    stats, graph_base64 = backtest_instance.run_basic_backtest(
        symbol=symbol,
        granularity=granularity,
        strategy_obj=strategy_obj,
        num_days=num_days,
        sizing=sizing,
        best_params=best_params,
        graph_callback=graph_callback
    )
    
    return stats, graph_base64
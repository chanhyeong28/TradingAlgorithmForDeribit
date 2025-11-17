"""
Deribit Trading Toolkit

A comprehensive Python package for implementing trading algorithms on Deribit exchange.
Provides tools for volatility analysis, options pricing, and systematic trading strategies.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.client import DeribitClient, DeribitAuth, DeribitError, DeribitConnectionError, DeribitAuthError
from .core.session_manager import SessionManager, MultiSessionTradingApp

# Analytics imports
from .analytics.volatility import VolatilityAnalyzer
from .analytics.pnl_decomposition import PnLDecomposer, PnLDecompositionResult
from .analytics.ssr import SSRCalculator, ExpiryState, KalmanState
from .analytics.min_var_delta import (
    MinimumVarianceDeltaCalculator,
    MinimumVarianceDeltaResult,
    ExpiryIntradayData,
    IntradayDataPoint
)
from .analytics.backtesting import HistoricalDataCollector, HistoricalOHLCV

# Model imports
from .models.market_data import (
    MarketData, OptionContract, OrderBook, VolatilityCurve, VolatilityPoint,
    RiskReversalSpread, TradingSignal, Position, Portfolio, Trade,
    OptionType, OrderType, OrderSide, InstrumentKind
)

# Strategy imports
from .strategies.base import BaseStrategy, RiskReversalStrategy, RiskReversalConfig, StrategyConfig

# Risk management imports
from .risk.manager import RiskManager, RiskLimits, RiskLevel, RiskMetrics

# Utility imports
from .utils.config import AppConfig, ConfigManager, DatabaseConfig, DeribitConfig, TelegramConfig, TradingConfig, RiskConfig, SessionConfig, LoggingConfig

# Main application
from .main import TradingApp

# Real-time applications
from .apps import RealTimeIVApp

# Backtesting
from .backtesting import (
    BacktestingEnvironment, BacktestingResult,
    BacktestResult, BacktestPosition, BacktestTrade, HedgePosition,
    SimpleOptionBacktester, OptionSpec, BacktestPlotter
)

__all__ = [
    # Core
    'DeribitClient',
    'DeribitAuth', 
    'DeribitError',
    'DeribitConnectionError',
    'DeribitAuthError',
    'SessionManager',
    'MultiSessionTradingApp',
    
    # Analytics
    'VolatilityAnalyzer',
    'PnLDecomposer',
    'PnLDecompositionResult',
    'SSRCalculator',
    'ExpiryState',
    'KalmanState',
    'MinimumVarianceDeltaCalculator',
    'MinimumVarianceDeltaResult',
    'ExpiryIntradayData',
    'IntradayDataPoint',
    'HistoricalDataCollector',
    'HistoricalOHLCV',
    
    # Models
    'MarketData',
    'OptionContract',
    'OrderBook',
    'VolatilityCurve',
    'VolatilityPoint',
    'RiskReversalSpread',
    'TradingSignal',
    'Position',
    'Portfolio',
    'Trade',
    'OptionType',
    'OrderType',
    'OrderSide',
    'InstrumentKind',
    
    # Strategies
    'BaseStrategy',
    'RiskReversalStrategy',
    'RiskReversalConfig',
    'StrategyConfig',
    
    # Risk
    'RiskManager',
    'RiskLimits',
    'RiskLevel',
    'RiskMetrics',
    
    # Utils
    'AppConfig',
    'ConfigManager',
    'DatabaseConfig',
    'DeribitConfig',
    'TelegramConfig',
    'TradingConfig',
    'RiskConfig',
    'SessionConfig',
    'LoggingConfig',
    
    # Main application
    'TradingApp',
    
    # Real-time applications
    'RealTimeIVApp',
    
    # Backtesting
    'BacktestingEnvironment',
    'BacktestingResult',
    'BacktestResult',
    'BacktestPosition',
    'BacktestTrade',
    'HedgePosition',
    'SimpleOptionBacktester',
    'OptionSpec',
    'BacktestPlotter',
]

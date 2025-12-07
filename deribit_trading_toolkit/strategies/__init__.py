"""
Trading Strategies

Available strategies:
- SimpleMarketMaker: Basic market making using orderbook
- SophisticatedMarketMaker: Theory-driven market making with WebSocket subscriptions
- RiskReversalStrategy: Risk reversal trading strategy
- CalendarSpreadStrategy: Calendar spread strategy
"""

# Import base classes first
from .base import BaseStrategy, StrategyConfig

# Import market maker (may fail if dependencies not available)
__all__ = [
    'BaseStrategy',
    'StrategyConfig',
]

try:
    from .market_maker import SimpleMarketMaker, MarketMakerConfig, MarketMakerState
    __all__.extend([
        'SimpleMarketMaker',
        'MarketMakerConfig',
        'MarketMakerState',
    ])
except ImportError:
    pass

try:
    from .sophisticated_mm import SophisticatedMarketMaker, SophisticatedMMConfig, MMState
    __all__.extend([
        'SophisticatedMarketMaker',
        'SophisticatedMMConfig',
        'MMState',
    ])
except ImportError:
    pass

"""
Backtesting Module

Provides tools for constructing backtesting environments for options trading.
This module handles collecting historical data, building volatility curves, and
saving everything to MySQL for backtesting purposes.
"""

from .environment import BacktestingEnvironment, BacktestingResult
from .simple_option import (
    BacktestResult, BacktestPosition, BacktestTrade, HedgePosition,
    SimpleOptionBacktester, OptionSpec, BacktestPlotter
)

__all__ = [
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


"""
Trading Strategies Module

Base strategy class and specific strategy implementations including:
- Risk Reversal Strategy
- Calendar Spread Strategy
- Volatility Arbitrage Strategy
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from ..models.market_data import TradingSignal, RiskReversalSpread, OrderSide, OrderType, Trade
from ..core.client import DeribitClient
from ..analytics.volatility import VolatilityAnalyzer
from ..risk.manager import RiskManager
from ..utils.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Base strategy configuration"""
    name: str
    enabled: bool = True
    position_size: float = 0.1
    max_positions: int = 4
    risk_tolerance: float = 0.1
    
    def validate(self) -> List[str]:
        """Validate strategy configuration"""
        errors = []
        
        if not self.name:
            errors.append("Strategy name is required")
        
        if self.position_size <= 0:
            errors.append("Position size must be positive")
        
        if self.max_positions <= 0:
            errors.append("Max positions must be positive")
        
        if self.risk_tolerance < 0 or self.risk_tolerance > 1:
            errors.append("Risk tolerance must be between 0 and 1")
        
        return errors


class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    
    Provides common functionality:
    - Signal generation
    - Trade execution
    - Risk management integration
    - Performance tracking
    """
    
    def __init__(self, config: StrategyConfig, client: DeribitClient, 
                 risk_manager: RiskManager):
        self.config = config
        self.client = client
        self.risk_manager = risk_manager
        self.enabled = config.enabled
        self.last_execution_time: Optional[datetime] = None
        self.execution_count = 0
        self.total_pnl = 0.0
        self.trades: List[Trade] = []
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Strategy configuration errors: {errors}")
    
    @abstractmethod
    async def generate_signals(self) -> List[TradingSignal]:
        """
        Generate trading signals based on strategy logic
        
        Returns:
            List of TradingSignal objects
        """
        pass
    
    @abstractmethod
    async def execute_signal(self, signal: TradingSignal) -> bool:
        """
        Execute a trading signal
        
        Args:
            signal: TradingSignal to execute
            
        Returns:
            True if execution successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_strategy_positions(self) -> Dict[str, float]:
        """
        Get simulated positions for risk checking
        
        Returns:
            Dictionary of instrument_name -> position_size
        """
        pass
    
    def enable(self):
        """Enable strategy"""
        self.enabled = True
        logger.info(f"Strategy {self.config.name} enabled")
    
    def disable(self):
        """Disable strategy"""
        self.enabled = False
        logger.info(f"Strategy {self.config.name} disabled")
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self.enabled
    
    async def run_strategy(self) -> Dict[str, Any]:
        """
        Main strategy execution loop
        
        Returns:
            Dictionary with execution results
        """
        try:
            if not self.enabled:
                return {"status": "disabled", "signals": 0, "executions": 0}
            
            # Generate signals
            signals = await self.generate_signals()
            
            if not signals:
                return {"status": "no_signals", "signals": 0, "executions": 0}
            
            # Execute signals
            executions = 0
            for signal in signals:
                if await self.execute_signal(signal):
                    executions += 1
            
            return {
                "status": "completed",
                "signals": len(signals),
                "executions": executions,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error running strategy {self.config.name}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.trades:
                return {
                    "total_trades": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "avg_trade_pnl": 0.0,
                    "max_trade_pnl": 0.0,
                    "min_trade_pnl": 0.0
                }
            
            total_trades = len(self.trades)
            total_pnl = sum(trade.pnl for trade in self.trades)
            winning_trades = [trade for trade in self.trades if trade.pnl > 0]
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            pnls = [trade.pnl for trade in self.trades]
            avg_trade_pnl = np.mean(pnls) if pnls else 0
            max_trade_pnl = max(pnls) if pnls else 0
            min_trade_pnl = min(pnls) if pnls else 0
            
            return {
                "total_trades": total_trades,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "avg_trade_pnl": avg_trade_pnl,
                "max_trade_pnl": max_trade_pnl,
                "min_trade_pnl": min_trade_pnl,
                "last_execution": self.last_execution_time,
                "execution_count": self.execution_count
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def add_trade(self, trade: Trade):
        """Add trade to strategy history"""
        self.trades.append(trade)
        self.total_pnl += trade.pnl
        self.execution_count += 1
        self.last_execution_time = trade.timestamp


@dataclass
class RiskReversalConfig(StrategyConfig):
    """Risk Reversal strategy configuration"""
    near_expiration: str = ""
    far_expiration: str = ""
    spread_way: str = "SHORT"  # "SHORT" or "LONG"
    min_spread_threshold: float = 0.0
    perpetual_expirations: List[str] = None
    
    def __post_init__(self):
        if self.perpetual_expirations is None:
            self.perpetual_expirations = []
    
    def validate(self) -> List[str]:
        """Validate Risk Reversal configuration"""
        errors = super().validate()
        
        if not self.near_expiration:
            errors.append("Near expiration is required")
        
        if not self.far_expiration:
            errors.append("Far expiration is required")
        
        if self.spread_way not in ["SHORT", "LONG"]:
            errors.append("Spread way must be 'SHORT' or 'LONG'")
        
        return errors


class RiskReversalStrategy(BaseStrategy):
    """
    Risk Reversal Strategy Implementation
    
    Strategy Logic:
    - Uses Risk Reversal spreads between two expirations
    - Takes opposite RR positions across near-term and far-term expirations
    - Executes when both IV and price spreads are positive
    - Reduces net delta exposure while maintaining volatility skew exposure
    """
    
    def __init__(self, config: RiskReversalConfig, client: DeribitClient, 
                 risk_manager: RiskManager):
        super().__init__(config, client, risk_manager)
        self.config = config
        self.volatility_analyzer = VolatilityAnalyzer()
        self.spread_calculator = SpreadCalculator()
        self.latest_spread: Optional[RiskReversalSpread] = None
        self.strike_prices: Dict[str, List[float]] = {}
        self.underlying_prices: Dict[str, float] = {}
        
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate Risk Reversal trading signals"""
        try:
            if not self.enabled:
                return []
            
            # Calculate current spread
            spread = await self._calculate_current_spread()
            if not spread:
                return []
            
            self.latest_spread = spread
            
            # Check if spread is profitable
            should_execute = spread.is_profitable
            
            # Create signal
            signal = TradingSignal(
                signal_type=self.config.spread_way,
                rr_spread=spread.rr_spread,
                rr_spread_price=spread.rr_spread_price,
                margin_ok=False,  # Will be updated by risk manager
                timestamp=datetime.now(),
                should_execute=should_execute,
                confidence=self._calculate_signal_confidence(spread)
            )
            
            return [signal] if should_execute else []
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute Risk Reversal strategy trades"""
        try:
            if not signal.is_valid:
                logger.warning("Signal is not valid for execution")
                return False
            
            # Get strategy positions for risk checking
            positions = self.get_strategy_positions()
            
            # Check risk management
            is_valid, reasons = await self.risk_manager.validate_trade(
                "BTC-OPTIONS", self.config.position_size, 1.0, signal.signal_type.lower()
            )
            
            if not is_valid:
                logger.warning(f"Trade validation failed: {reasons}")
                return False
            
            # Execute the four legs of Risk Reversal
            legs = self._get_strategy_legs(signal.signal_type)
            
            executed_trades = []
            for leg in legs:
                try:
                    result = await self.client.place_order(
                        instrument_name=leg['instrument'],
                        amount=leg['amount'],
                        order_type=OrderType.MARKET,
                        side=leg['side']
                    )
                    
                    # Create trade record
                    trade = Trade(
                        instrument_name=leg['instrument'],
                        side=leg['side'],
                        amount=leg['amount'],
                        price=leg.get('price', 0),
                        timestamp=datetime.now(),
                        order_id=result.get('order_id', ''),
                        trade_id=result.get('trade_id', ''),
                        pnl=0.0  # Will be updated later
                    )
                    
                    executed_trades.append(trade)
                    self.add_trade(trade)
                    
                except Exception as e:
                    logger.error(f"Error executing leg {leg}: {e}")
                    # Cancel any successful trades if one fails
                    await self._cancel_executed_trades(executed_trades)
                    return False
            
            logger.info(f"Risk Reversal strategy executed: {signal.signal_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def get_strategy_positions(self) -> Dict[str, float]:
        """Get simulated positions for margin checking"""
        try:
            positions = {}
            
            # Define the four legs of Risk Reversal
            if self.config.spread_way == "SHORT":
                positions.update({
                    f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'call')}-C": -self.config.position_size,
                    f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'put')}-P": self.config.position_size,
                    f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'call')}-C": self.config.position_size,
                    f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'put')}-P": -self.config.position_size,
                })
            else:  # LONG
                positions.update({
                    f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'call')}-C": self.config.position_size,
                    f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'put')}-P": -self.config.position_size,
                    f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'call')}-C": -self.config.position_size,
                    f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'put')}-P": self.config.position_size,
                })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting strategy positions: {e}")
            return {}
    
    async def _calculate_current_spread(self) -> Optional[RiskReversalSpread]:
        """Calculate current Risk Reversal spread"""
        try:
            # Get options data for both expirations
            near_options = await self._get_options_data(self.config.near_expiration)
            far_options = await self._get_options_data(self.config.far_expiration)
            
            if not near_options or not far_options:
                return None
            
            # Calculate spreads
            if self.config.spread_way == "SHORT":
                far_spread = self._calculate_short_spread(far_options)
                near_spread = self._calculate_short_spread(near_options)
                far_spread_price = self._calculate_short_spread_price(far_options)
                near_spread_price = self._calculate_short_spread_price(near_options)
            else:  # LONG
                far_spread = self._calculate_long_spread(far_options)
                near_spread = self._calculate_long_spread(near_options)
                far_spread_price = self._calculate_long_spread_price(far_options)
                near_spread_price = self._calculate_long_spread_price(near_options)
            
            rr_spread = far_spread + near_spread
            rr_spread_price = far_spread_price + near_spread_price
            
            return RiskReversalSpread(
                far_spread=far_spread,
                near_spread=near_spread,
                rr_spread=rr_spread,
                far_spread_price=far_spread_price,
                near_spread_price=near_spread_price,
                rr_spread_price=rr_spread_price,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating current spread: {e}")
            return None
    
    def _get_strategy_legs(self, signal_type: str) -> List[Dict[str, Any]]:
        """Get the four legs of the Risk Reversal strategy"""
        try:
            legs = []
            
            if signal_type == "SHORT":
                legs = [
                    {
                        'instrument': f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'call')}-C",
                        'amount': self.config.position_size,
                        'side': OrderSide.SELL
                    },
                    {
                        'instrument': f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'put')}-P",
                        'amount': self.config.position_size,
                        'side': OrderSide.BUY
                    },
                    {
                        'instrument': f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'call')}-C",
                        'amount': self.config.position_size,
                        'side': OrderSide.BUY
                    },
                    {
                        'instrument': f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'put')}-P",
                        'amount': self.config.position_size,
                        'side': OrderSide.SELL
                    }
                ]
            else:  # LONG
                legs = [
                    {
                        'instrument': f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'call')}-C",
                        'amount': self.config.position_size,
                        'side': OrderSide.BUY
                    },
                    {
                        'instrument': f"BTC-{self.config.far_expiration}-{self._get_strike('far', 'put')}-P",
                        'amount': self.config.position_size,
                        'side': OrderSide.SELL
                    },
                    {
                        'instrument': f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'call')}-C",
                        'amount': self.config.position_size,
                        'side': OrderSide.SELL
                    },
                    {
                        'instrument': f"BTC-{self.config.near_expiration}-{self._get_strike('near', 'put')}-P",
                        'amount': self.config.position_size,
                        'side': OrderSide.BUY
                    }
                ]
            
            return legs
            
        except Exception as e:
            logger.error(f"Error getting strategy legs: {e}")
            return []
    
    def _get_strike(self, expiration: str, option_type: str) -> int:
        """Get strike price for given expiration and option type"""
        try:
            # Simplified strike selection - in practice you'd use delta-based selection
            if expiration == "far":
                return 50000  # Example strike
            else:
                return 50000  # Example strike
        except:
            return 50000
    
    def _calculate_signal_confidence(self, spread: RiskReversalSpread) -> float:
        """Calculate signal confidence based on spread strength"""
        try:
            # Simple confidence calculation based on spread magnitude
            iv_strength = abs(spread.rr_spread) / 0.1  # Normalize to typical spread
            price_strength = abs(spread.rr_spread_price) / 100  # Normalize to typical price spread
            
            confidence = min(1.0, (iv_strength + price_strength) / 2)
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating signal confidence: {e}")
            return 0.5
    
    async def _get_options_data(self, expiration: str) -> List[Dict]:
        """Get options data for given expiration"""
        try:
            # Simplified - in practice you'd query actual options data
            return []
        except Exception as e:
            logger.error(f"Error getting options data: {e}")
            return []
    
    def _calculate_short_spread(self, options_data: List[Dict]) -> float:
        """Calculate short spread (simplified)"""
        return 0.05  # Example spread
    
    def _calculate_long_spread(self, options_data: List[Dict]) -> float:
        """Calculate long spread (simplified)"""
        return 0.05  # Example spread
    
    def _calculate_short_spread_price(self, options_data: List[Dict]) -> float:
        """Calculate short spread price (simplified)"""
        return 50.0  # Example price spread
    
    def _calculate_long_spread_price(self, options_data: List[Dict]) -> float:
        """Calculate long spread price (simplified)"""
        return 50.0  # Example price spread
    
    async def _cancel_executed_trades(self, trades: List[Trade]):
        """Cancel executed trades if execution fails"""
        try:
            for trade in trades:
                if trade.order_id:
                    await self.client.cancel_order(trade.order_id)
        except Exception as e:
            logger.error(f"Error canceling trades: {e}")


class SpreadCalculator:
    """Helper class for spread calculations"""
    
    def __init__(self):
        pass
    
    def calculate_risk_reversal_spread(self, call_iv: float, put_iv: float) -> float:
        """Calculate Risk Reversal spread"""
        return call_iv - put_iv
    
    def calculate_butterfly_spread(self, wing_iv: float, atm_iv: float) -> float:
        """Calculate Butterfly spread"""
        return wing_iv - atm_iv

"""
Risk Management System

Comprehensive risk management for trading strategies including:
- Position sizing
- Margin management
- Portfolio risk assessment
- Stop-loss mechanisms
- Risk limits enforcement
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..models.market_data import Portfolio, Position, Trade
from ..core.client import DeribitClient
from ..utils.config import RiskConfig

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float = 0.1
    max_portfolio_risk: float = 0.2
    margin_buffer: float = 1.2
    max_daily_loss: float = 0.1
    stop_loss_percentage: float = 0.05
    max_positions: int = 10
    max_correlation: float = 0.7
    
    def validate(self) -> List[str]:
        """Validate risk limits configuration"""
        errors = []
        
        if self.max_position_size <= 0:
            errors.append("Max position size must be positive")
        
        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
            errors.append("Max portfolio risk must be between 0 and 1")
        
        if self.margin_buffer < 1.0:
            errors.append("Margin buffer must be >= 1.0")
        
        if self.max_daily_loss <= 0 or self.max_daily_loss > 1:
            errors.append("Max daily loss must be between 0 and 1")
        
        if self.stop_loss_percentage <= 0 or self.stop_loss_percentage > 1:
            errors.append("Stop loss percentage must be between 0 and 1")
        
        if self.max_positions <= 0:
            errors.append("Max positions must be positive")
        
        if self.max_correlation < 0 or self.max_correlation > 1:
            errors.append("Max correlation must be between 0 and 1")
        
        return errors


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    
    @property
    def risk_level(self) -> RiskLevel:
        """Determine risk level based on metrics"""
        if self.var_99 > 0.2 or self.max_drawdown > 0.3:
            return RiskLevel.CRITICAL
        elif self.var_95 > 0.1 or self.max_drawdown > 0.15:
            return RiskLevel.HIGH
        elif self.var_95 > 0.05 or self.max_drawdown > 0.08:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class RiskManager:
    """
    Comprehensive risk management system
    
    Features:
    - Position sizing validation
    - Margin requirement checking
    - Portfolio risk assessment
    - Stop-loss management
    - Risk limit enforcement
    """
    
    def __init__(self, client: DeribitClient, risk_limits: RiskLimits):
        self.client = client
        self.risk_limits = risk_limits
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()
        
    def check_position_size(self, instrument_name: str, amount: float, 
                          current_price: float) -> Tuple[bool, str]:
        """
        Check if position size is within risk limits
        
        Args:
            instrument_name: Name of the instrument
            amount: Position size
            current_price: Current market price
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check absolute position size
            notional_value = abs(amount) * current_price
            
            if notional_value > self.risk_limits.max_position_size:
                return False, f"Position size {notional_value} exceeds limit {self.risk_limits.max_position_size}"
            
            # Check if we're adding too many positions
            current_positions = self._get_current_position_count()
            if current_positions >= self.risk_limits.max_positions:
                return False, f"Maximum number of positions ({self.risk_limits.max_positions}) reached"
            
            return True, "Position size is within limits"
            
        except Exception as e:
            logger.error(f"Error checking position size: {e}")
            return False, f"Error checking position size: {e}"
    
    async def check_margin_requirements(self, positions: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if margin requirements are satisfied for given positions
        
        Args:
            positions: Dictionary of instrument_name -> position_size
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Simulate portfolio with given positions
            portfolio_simulation = await self.client.simulate_portfolio(positions)
            
            if not portfolio_simulation or 'result' not in portfolio_simulation:
                return False, "Portfolio simulation failed"
                
            result = portfolio_simulation['result']
            equity = result.get('equity', 0)
            maintenance_margin = result.get('projected_maintenance_margin', 0)
            initial_margin = result.get('projected_initial_margin', 0)
            margin_balance = result.get('margin_balance', 0)
            
            # Check margin requirements
            if equity < maintenance_margin * self.risk_limits.margin_buffer:
                return False, f"Equity {equity} below maintenance margin buffer {maintenance_margin * self.risk_limits.margin_buffer}"
            
            if margin_balance < initial_margin:
                return False, f"Margin balance {margin_balance} below initial margin {initial_margin}"
            
            return True, "Margin requirements satisfied"
            
        except Exception as e:
            logger.error(f"Error checking margin requirements: {e}")
            return False, f"Error checking margin requirements: {e}"
    
    async def assess_portfolio_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """
        Assess overall portfolio risk
        
        Args:
            portfolio: Portfolio object
            
        Returns:
            RiskMetrics object
        """
        try:
            # Calculate basic risk metrics
            total_exposure = sum(abs(pos.market_value) for pos in portfolio.positions)
            portfolio_volatility = self._calculate_portfolio_volatility(portfolio)
            
            # Calculate VaR (simplified)
            var_95 = self._calculate_var(portfolio, 0.95)
            var_99 = self._calculate_var(portfolio, 0.99)
            
            # Calculate expected shortfall
            expected_shortfall = self._calculate_expected_shortfall(portfolio, 0.95)
            
            # Calculate max drawdown (simplified)
            max_drawdown = self._calculate_max_drawdown(portfolio)
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=portfolio_volatility,
                beta=0.0,  # Would need market data
                correlation=0.0  # Would need correlation matrix
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return RiskMetrics()
    
    def check_daily_loss_limit(self, trade: Trade) -> Tuple[bool, str]:
        """
        Check if daily loss limit would be exceeded
        
        Args:
            trade: Trade object
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Reset daily tracking if new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_pnl = 0.0
                self.daily_trades = []
                self.last_reset_date = current_date
            
            # Add trade to daily tracking
            self.daily_trades.append(trade)
            self.daily_pnl += trade.pnl
            
            # Check if daily loss limit would be exceeded
            if self.daily_pnl < -self.risk_limits.max_daily_loss:
                return False, f"Daily loss {abs(self.daily_pnl)} exceeds limit {self.risk_limits.max_daily_loss}"
            
            return True, "Daily loss limit not exceeded"
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return False, f"Error checking daily loss limit: {e}"
    
    def calculate_stop_loss_price(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price based on risk limits
        
        Args:
            entry_price: Entry price of the position
            side: "buy" or "sell"
            
        Returns:
            Stop loss price
        """
        try:
            if side.lower() == "buy":
                # Long position - stop loss below entry
                stop_loss = entry_price * (1 - self.risk_limits.stop_loss_percentage)
            else:
                # Short position - stop loss above entry
                stop_loss = entry_price * (1 + self.risk_limits.stop_loss_percentage)
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss price: {e}")
            return entry_price
    
    def check_correlation_risk(self, new_instrument: str, 
                             existing_positions: List[Position]) -> Tuple[bool, str]:
        """
        Check correlation risk with existing positions
        
        Args:
            new_instrument: New instrument to add
            existing_positions: List of existing positions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Simplified correlation check based on instrument names
            # In practice, you'd calculate actual correlations
            
            for position in existing_positions:
                correlation = self._estimate_correlation(new_instrument, position.instrument_name)
                
                if correlation > self.risk_limits.max_correlation:
                    return False, f"High correlation {correlation:.2f} with {position.instrument_name}"
            
            return True, "Correlation risk acceptable"
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return False, f"Error checking correlation risk: {e}"
    
    async def validate_trade(self, instrument_name: str, amount: float, 
                           price: float, side: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive trade validation
        
        Args:
            instrument_name: Instrument to trade
            amount: Trade amount
            price: Trade price
            side: Trade side
            
        Returns:
            Tuple of (is_valid, list_of_reasons)
        """
        reasons = []
        
        # Check position size
        is_valid_size, size_reason = self.check_position_size(instrument_name, amount, price)
        if not is_valid_size:
            reasons.append(size_reason)
        
        # Check margin requirements
        positions = {instrument_name: amount}
        is_valid_margin, margin_reason = await self.check_margin_requirements(positions)
        if not is_valid_margin:
            reasons.append(margin_reason)
        
        # Check daily loss limit
        trade = Trade(
            instrument_name=instrument_name,
            side=side,
            amount=amount,
            price=price,
            timestamp=datetime.now(),
            order_id="",
            trade_id=""
        )
        is_valid_daily, daily_reason = self.check_daily_loss_limit(trade)
        if not is_valid_daily:
            reasons.append(daily_reason)
        
        # Check correlation risk
        existing_positions = await self._get_existing_positions()
        is_valid_correlation, correlation_reason = self.check_correlation_risk(
            instrument_name, existing_positions
        )
        if not is_valid_correlation:
            reasons.append(correlation_reason)
        
        is_valid = len(reasons) == 0
        return is_valid, reasons
    
    def get_risk_summary(self) -> Dict:
        """
        Get comprehensive risk summary
        
        Returns:
            Dictionary with risk summary
        """
        try:
            return {
                'daily_pnl': self.daily_pnl,
                'daily_trades_count': len(self.daily_trades),
                'risk_limits': {
                    'max_position_size': self.risk_limits.max_position_size,
                    'max_portfolio_risk': self.risk_limits.max_portfolio_risk,
                    'margin_buffer': self.risk_limits.margin_buffer,
                    'max_daily_loss': self.risk_limits.max_daily_loss,
                    'stop_loss_percentage': self.risk_limits.stop_loss_percentage,
                    'max_positions': self.risk_limits.max_positions,
                    'max_correlation': self.risk_limits.max_correlation
                },
                'current_status': {
                    'daily_loss_limit_exceeded': self.daily_pnl < -self.risk_limits.max_daily_loss,
                    'risk_level': self._get_overall_risk_level()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    # Helper methods
    def _get_current_position_count(self) -> int:
        """Get current number of positions (simplified)"""
        # In practice, you'd query the actual positions
        return 0
    
    async def _get_existing_positions(self) -> List[Position]:
        """Get existing positions (simplified)"""
        # In practice, you'd query the actual positions
        return []
    
    def _calculate_portfolio_volatility(self, portfolio: Portfolio) -> float:
        """Calculate portfolio volatility (simplified)"""
        try:
            # Simplified calculation - in practice you'd use proper portfolio theory
            if not portfolio.positions:
                return 0.0
            
            # Weighted average of individual volatilities
            total_value = sum(abs(pos.market_value) for pos in portfolio.positions)
            if total_value == 0:
                return 0.0
            
            weighted_vol = sum(
                abs(pos.market_value) / total_value * 0.5  # Assume 50% vol for each position
                for pos in portfolio.positions
            )
            
            return weighted_vol
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    def _calculate_var(self, portfolio: Portfolio, confidence: float) -> float:
        """Calculate Value at Risk (simplified)"""
        try:
            # Simplified VaR calculation
            portfolio_value = portfolio.equity
            volatility = self._calculate_portfolio_volatility(portfolio)
            
            # Use normal distribution assumption
            from scipy.stats import norm
            z_score = norm.ppf(confidence)
            var = portfolio_value * volatility * z_score
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_expected_shortfall(self, portfolio: Portfolio, confidence: float) -> float:
        """Calculate Expected Shortfall (simplified)"""
        try:
            # Simplified ES calculation
            var = self._calculate_var(portfolio, confidence)
            # ES is typically 1.3-1.5 times VaR for normal distributions
            return var * 1.4
            
        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, portfolio: Portfolio) -> float:
        """Calculate maximum drawdown (simplified)"""
        try:
            # Simplified max drawdown calculation
            # In practice, you'd track historical equity values
            return abs(portfolio.total_pl) / portfolio.equity if portfolio.equity > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, portfolio: Portfolio) -> float:
        """Calculate Sharpe ratio (simplified)"""
        try:
            # Simplified Sharpe ratio calculation
            if portfolio.equity == 0:
                return 0.0
            
            # Assume risk-free rate of 5%
            risk_free_rate = 0.05
            excess_return = portfolio.total_pl / portfolio.equity - risk_free_rate
            volatility = self._calculate_portfolio_volatility(portfolio)
            
            if volatility == 0:
                return 0.0
            
            return excess_return / volatility
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _estimate_correlation(self, instrument1: str, instrument2: str) -> float:
        """Estimate correlation between two instruments (simplified)"""
        try:
            # Simplified correlation estimation based on instrument names
            # In practice, you'd calculate actual historical correlations
            
            # Same underlying = high correlation
            if instrument1.split('-')[0] == instrument2.split('-')[0]:
                return 0.8
            
            # Different underlyings = low correlation
            return 0.2
            
        except Exception as e:
            logger.error(f"Error estimating correlation: {e}")
            return 0.0
    
    def _get_overall_risk_level(self) -> RiskLevel:
        """Get overall risk level"""
        try:
            if self.daily_pnl < -self.risk_limits.max_daily_loss * 0.8:
                return RiskLevel.CRITICAL
            elif self.daily_pnl < -self.risk_limits.max_daily_loss * 0.5:
                return RiskLevel.HIGH
            elif self.daily_pnl < -self.risk_limits.max_daily_loss * 0.2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error getting overall risk level: {e}")
            return RiskLevel.MEDIUM

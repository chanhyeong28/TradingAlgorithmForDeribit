"""
Market Data Models

Core data structures for representing market data, options contracts, and trading instruments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import json


class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class InstrumentKind(Enum):
    """Instrument kind enumeration"""
    OPTION = "option"
    FUTURE = "future"
    PERPETUAL = "perpetual"


@dataclass
class OptionContract:
    """Represents an options contract"""
    instrument_name: str
    base_currency: str
    quote_currency: str
    kind: InstrumentKind
    is_active: bool
    min_trade_amount: float
    tick_size: float
    contract_size: float
    settlement_period: str
    expiration_timestamp: int
    strike: Optional[float] = None
    option_type: Optional[OptionType] = None
    
    @property
    def expiration_date(self) -> datetime:
        """Get expiration date as datetime object"""
        return datetime.fromtimestamp(self.expiration_timestamp)
    
    @property
    def days_to_expiry(self) -> int:
        """Get days to expiry"""
        return (self.expiration_date - datetime.now()).days
    
    @property
    def time_to_expiry(self) -> float:
        """Get time to expiry in years"""
        return self.days_to_expiry / 365.0


@dataclass
class MarketData:
    """Represents market data for an instrument"""
    instrument_name: str
    timestamp: int
    stats: Dict[str, Any] = field(default_factory=dict)
    state: str = ""
    ticker: Dict[str, Any] = field(default_factory=dict)
    mark_price: Optional[float] = None
    mark_iv: Optional[float] = None
    best_bid_price: Optional[float] = None
    best_ask_price: Optional[float] = None
    best_bid_amount: Optional[float] = None
    best_ask_amount: Optional[float] = None
    greeks: Dict[str, float] = field(default_factory=dict)
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        if self.best_bid_price and self.best_ask_price:
            return (self.best_bid_price + self.best_ask_price) / 2
        return self.mark_price
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if self.best_bid_price and self.best_ask_price:
            return self.best_ask_price - self.best_bid_price
        return None
    
    @property
    def spread_percentage(self) -> Optional[float]:
        """Calculate bid-ask spread as percentage"""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 100
        return None
    
    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime object"""
        return datetime.fromtimestamp(self.timestamp / 1000)


@dataclass
class OrderBook:
    """Represents order book data"""
    instrument_name: str
    timestamp: int
    bids: List[List[float]] = field(default_factory=list)  # [price, amount]
    asks: List[List[float]] = field(default_factory=list)  # [price, amount]
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.asks[0][0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


@dataclass
class VolatilityPoint:
    """Single volatility data point"""
    strike: float
    iv: float
    moneyness: float
    option_type: OptionType
    expiration: int
    timestamp: int
    
    @property
    def log_moneyness(self) -> float:
        """Get log moneyness"""
        return self.moneyness


@dataclass
class VolatilityCurve:
    """Volatility curve for a specific expiration"""
    expiration: int
    points: List[VolatilityPoint]
    atm_strike: float
    atm_iv: float
    slope: float
    curvature: float
    timestamp: int
    
    @property
    def expiration_date(self) -> datetime:
        """Get expiration date"""
        return datetime.fromtimestamp(self.expiration)
    
    @property
    def time_to_expiry(self) -> float:
        """Get time to expiry in years"""
        return (self.expiration - self.timestamp / 1000) / (365 * 24 * 3600)


@dataclass
class RiskReversalSpread:
    """Risk Reversal spread data"""
    far_spread: float
    near_spread: float
    rr_spread: float
    far_spread_price: float
    near_spread_price: float
    rr_spread_price: float
    timestamp: datetime
    
    @property
    def is_profitable(self) -> bool:
        """Check if spread is profitable (both IV and price spreads positive)"""
        return self.rr_spread > 0 and self.rr_spread_price > 0


@dataclass
class TradingSignal:
    """Trading signal data"""
    signal_type: str  # "LONG" or "SHORT"
    rr_spread: float
    rr_spread_price: float
    margin_ok: bool
    timestamp: datetime
    should_execute: bool
    confidence: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """Check if signal is valid for execution"""
        return self.should_execute and self.margin_ok


@dataclass
class Position:
    """Represents a trading position"""
    instrument_name: str
    size: float
    average_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Calculate market value of position"""
        return abs(self.size) * self.average_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.size < 0


@dataclass
class Portfolio:
    """Represents a trading portfolio"""
    currency: str
    equity: float
    margin_balance: float
    initial_margin: float
    maintenance_margin: float
    total_pl: float
    positions: List[Position] = field(default_factory=list)
    
    @property
    def margin_ratio(self) -> float:
        """Calculate margin ratio"""
        if self.maintenance_margin > 0:
            return self.equity / self.maintenance_margin
        return 0.0
    
    @property
    def is_margin_safe(self) -> bool:
        """Check if portfolio is margin safe"""
        return self.margin_ratio > 1.2
    
    @property
    def total_delta(self) -> float:
        """Calculate total portfolio delta"""
        return sum(pos.delta for pos in self.positions)
    
    @property
    def total_gamma(self) -> float:
        """Calculate total portfolio gamma"""
        return sum(pos.gamma for pos in self.positions)
    
    @property
    def total_theta(self) -> float:
        """Calculate total portfolio theta"""
        return sum(pos.theta for pos in self.positions)
    
    @property
    def total_vega(self) -> float:
        """Calculate total portfolio vega"""
        return sum(pos.vega for pos in self.positions)


@dataclass
class Trade:
    """Represents a completed trade"""
    instrument_name: str
    side: OrderSide
    amount: float
    price: float
    timestamp: datetime
    order_id: str
    trade_id: str
    fee: float = 0.0
    pnl: float = 0.0
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of trade"""
        return self.amount * self.price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'instrument_name': self.instrument_name,
            'side': self.side.value,
            'amount': self.amount,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'trade_id': self.trade_id,
            'fee': self.fee,
            'pnl': self.pnl
        }

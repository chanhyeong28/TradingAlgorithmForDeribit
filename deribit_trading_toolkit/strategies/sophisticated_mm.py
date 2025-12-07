"""
Sophisticated Market Making Strategy

A theory-driven market making strategy based on:
- Ho & Stoll (1981) inventory risk model
- Avellaneda & Stoikov (2008) optimal market making
- Glosten-Milgrom adverse selection model

Features:
- Real-time orderbook subscription via WebSocket
- Volatility-based spread adjustment
- Inventory-based quote skewing
- Risk management with kill switches
- Realized volatility tracking
"""

import asyncio
import logging
import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .base import BaseStrategy, StrategyConfig
from ..models.market_data import OrderBook, OrderType, OrderSide, Trade
from ..core.client import DeribitClient
from ..risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class MMState:
    """Market maker state variables"""
    # Position & PnL
    position_btc: float = 0.0
    cash_usdc: float = 0.0
    unrealized_pnl_usdc: float = 0.0
    realized_pnl_usdc: float = 0.0
    
    # Market data
    mid_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    last_price: Optional[float] = None
    
    # Volatility tracking
    short_term_vol: float = 0.0
    price_history: List[Tuple[int, float]] = field(default_factory=list)  # [(timestamp_ms, mid_price)]
    
    # Active orders
    active_bid_order_id: Optional[str] = None
    active_ask_order_id: Optional[str] = None
    active_bid_price: Optional[float] = None
    active_ask_price: Optional[float] = None
    active_bid_size: float = 0.0
    active_ask_size: float = 0.0
    active_bid_timestamp: Optional[int] = None
    active_ask_timestamp: Optional[int] = None
    
    # Equity tracking
    starting_equity_usdc: Optional[float] = None
    current_equity_usdc: Optional[float] = None
    
    # Timing
    last_risk_check_ts: int = 0
    last_quote_update_ts: int = 0


@dataclass
class SophisticatedMMConfig(StrategyConfig):
    """Sophisticated market maker configuration"""
    instrument_name: str = "BTC_USDC-PERPETUAL"  # Default to BTC_USDC-PERPETUAL
    quote_currency: str = "USDC"
    
    # Risk / inventory
    target_inventory: float = 0.0  # in BTC
    max_inventory_abs: float = 1.0  # hard cap (BTC)
    inventory_bucket_1: float = 0.25  # mild skew threshold
    inventory_bucket_2: float = 0.5  # strong skew threshold
    
    # Spread & size
    base_spread_bps: float = 3.0  # 3 bps each side (0.03%) around mid
    min_spread_bps: float = 1.0
    max_spread_bps: float = 20.0
    base_quote_size: float = 0.01  # 0.01 BTC per quote
    max_quote_size: float = 0.05  # cap per order
    
    # Volatility / regime
    vol_lookback_secs: int = 60  # short-term realized vol window
    vol_scale_factor: float = 1.0  # multiply vol to get spread adjustment
    max_allowed_vol: float = 0.03  # 3% per vol_lookback: beyond this, widen/turn off
    
    # Time controls
    quote_refresh_interval_ms: int = 300  # how often to reconsider quotes
    quote_max_age_ms: int = 2000  # cancel if older than this
    risk_check_interval_secs: int = 5
    
    # Fees
    maker_fee_rate: float = -0.00005  # -0.5 bps (rebate)
    taker_fee_rate: float = 0.00025  # 2.5 bps
    slippage_buffer_bps: float = 0.5  # buffer for effective spread
    
    # Kill switch
    max_intraday_drawdown_pct: float = 2.0
    max_realized_loss_usdc: float = 1000.0
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = super().validate()
        if not self.instrument_name:
            errors.append("Instrument name is required")
        return errors


class SophisticatedMarketMaker(BaseStrategy):
    """
    Sophisticated market making strategy with:
    - Real-time orderbook subscription
    - Volatility-based spread adjustment
    - Inventory-based quote skewing
    - Risk management
    """
    
    def __init__(self, config: SophisticatedMMConfig, client: DeribitClient,
                 risk_manager: RiskManager):
        super().__init__(config, client, risk_manager)
        self.mm_config: SophisticatedMMConfig = config
        self.state = MMState()
        self.running = False
        self.orderbook_subscribed = False
        self._cached_instrument_info: Optional[Dict] = None
        
    async def generate_signals(self):
        """Market maker doesn't generate signals"""
        return []
    
    async def execute_signal(self, signal):
        """Market maker doesn't execute signals"""
        return False
    
    def get_strategy_positions(self) -> Dict[str, float]:
        """Get strategy positions"""
        if self.state.mid_price and self.state.position_btc != 0:
            return {self.mm_config.instrument_name: self.state.position_btc}
        return {}
    
    async def start(self):
        """Start the sophisticated market maker"""
        if self.running:
            logger.warning("Market maker is already running")
            return
        
        self.running = True
        logger.info(f"Starting sophisticated market maker for {self.mm_config.instrument_name}")
        
        try:
            # Initialize state
            await self._initialize_state()
            
            # Subscribe to orderbook
            await self._subscribe_orderbook()
            
            # Register WebSocket handlers
            self._register_handlers()
            
            # Start main loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error in market maker loop: {e}", exc_info=True)
        finally:
            await self._cleanup()
            self.running = False
    
    async def stop(self):
        """Stop the market maker"""
        logger.info("Stopping sophisticated market maker...")
        self.running = False
        await self._cancel_all_orders()
        await self._cleanup()
    
    async def _initialize_state(self):
        """Initialize market maker state"""
        try:
            # Get account summary
            account = await self.client.get_account_summary("BTC")
            # For USDC accounts, we need to get USDC balance
            # Deribit uses BTC as base currency, so we need to convert
            equity_btc = account.get('equity', 0.0)
            available_btc = account.get('available_funds', 0.0)
            
            # Get current positions
            positions = await self.client.get_positions("BTC")
            for pos in positions:
                if pos.get('instrument_name') == self.mm_config.instrument_name:
                    self.state.position_btc = pos.get('size', 0.0)
                    break
            
            # Initialize equity - get current price to convert BTC to USDC
            orderbook = await self.client.get_order_book(self.mm_config.instrument_name, depth=1)
            if orderbook.mid_price:
                self.state.mid_price = orderbook.mid_price
                self.state.best_bid = orderbook.best_bid
                self.state.best_ask = orderbook.best_ask
                # Convert BTC balance to USDC
                self.state.cash_usdc = available_btc * self.state.mid_price
                self.state.current_equity_usdc = self._compute_equity_usdc()
                self.state.starting_equity_usdc = self.state.current_equity_usdc
            else:
                # Fallback: use a default price
                self.state.cash_usdc = available_btc * 90000  # Rough estimate
                self.state.current_equity_usdc = self.state.cash_usdc
                self.state.starting_equity_usdc = self.state.cash_usdc
            
            logger.info(f"Initialized state:")
            logger.info(f"  Position: {self.state.position_btc:.6f} BTC")
            logger.info(f"  Cash: {self.state.cash_usdc:.2f} USDC")
            logger.info(f"  Equity: {self.state.current_equity_usdc:.2f} USDC")
            
        except Exception as e:
            logger.error(f"Error initializing state: {e}")
    
    async def _subscribe_orderbook(self):
        """Subscribe to orderbook updates via WebSocket"""
        try:
            # Connect WebSocket if not connected
            if not self.client.websocket:
                await self.client.connect_websocket()
            
            # Subscribe to orderbook channel (private subscription for authenticated user)
            channel = f"book.{self.mm_config.instrument_name}.none.10.100ms"
            await self.client.subscribe([channel], private=True)
            self.orderbook_subscribed = True
            logger.info(f"Subscribed to orderbook: {channel}")
            
            # Start listening for messages in background if not already running
            # Check if listen_for_messages is already running
            if not hasattr(self.client, '_listening') or not self.client._listening:
                self.client._listening = True
                asyncio.create_task(self.client.listen_for_messages())
            
        except Exception as e:
            logger.error(f"Error subscribing to orderbook: {e}")
    
    async def _subscribe_user_trades(self):
        """Subscribe to user trades updates"""
        try:
            if not self.client.websocket:
                await self.client.connect_websocket()
            
            # Subscribe to user trades for the instrument
            channel = f"user.trades.{self.mm_config.instrument_name}.raw"
            await self.client.subscribe([channel], private=True)
            
            # Register handler - escape special chars for regex
            inst_escaped = self.mm_config.instrument_name.replace('-', r'\-').replace('_', '_')
            pattern = f"user\\.trades\\.{inst_escaped}.*"
            self.client.register_message_handler(
                pattern,
                self._on_trade_update
            )
            
            logger.info(f"Subscribed to user trades: {channel}")
        except Exception as e:
            logger.error(f"Error subscribing to user trades: {e}")
    
    def _register_handlers(self):
        """Register WebSocket message handlers"""
        # Register orderbook handler (pattern matching)
        # Escape special characters in instrument name for regex
        inst_escaped = self.mm_config.instrument_name.replace('-', r'\-').replace('_', '_')
        pattern = f"book\\.{inst_escaped}.*"
        self.client.register_message_handler(
            pattern,
            self._on_orderbook_update
        )
        
        # Register user trades handler (subscribe separately)
        # Note: user.trades subscription needs to be done separately
        asyncio.create_task(self._subscribe_user_trades())
    
    async def _on_orderbook_update(self, channel: str, message: Dict):
        """Handle orderbook update from WebSocket"""
        try:
            data = message.get('params', {}).get('data', {})
            if not data:
                return
            
            now_ts_ms = int(datetime.now().timestamp() * 1000)
            
            # Extract best bid/ask
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2.0
                
                self.state.best_bid = best_bid
                self.state.best_ask = best_ask
                self.state.mid_price = mid_price
                
                # Store for volatility computation
                self.state.price_history.append((now_ts_ms, mid_price))
                self._trim_price_history(now_ts_ms)
                self.state.short_term_vol = self._compute_realized_vol()
                
                # Trigger quote update if needed
                await self._check_and_update_quotes(now_ts_ms)
                
        except Exception as e:
            logger.error(f"Error handling orderbook update: {e}")
    
    async def _on_trade_update(self, channel: str, message: Dict):
        """Handle trade update (our order was filled)"""
        try:
            data = message.get('params', {}).get('data', {})
            if not data:
                return
            
            direction = data.get('direction', '')
            price = data.get('price', 0.0)
            amount = data.get('amount', 0.0)
            fee = data.get('fee', 0.0)
            
            if direction == 'buy':
                self.state.position_btc += amount
                self.state.cash_usdc -= price * amount
                if self.state.active_bid_order_id:
                    self.state.active_bid_order_id = None
            elif direction == 'sell':
                self.state.position_btc -= amount
                self.state.cash_usdc += price * amount
                if self.state.active_ask_order_id:
                    self.state.active_ask_order_id = None
            
            # Update fee
            self.state.cash_usdc -= fee
            
            # Update equity
            self.state.current_equity_usdc = self._compute_equity_usdc()
            
            logger.info(f"Trade filled: {direction} {amount:.6f} @ {price:.2f}, Position: {self.state.position_btc:.6f} BTC")
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    def _trim_price_history(self, now_ts_ms: int):
        """Trim price history to lookback window"""
        cutoff = now_ts_ms - self.mm_config.vol_lookback_secs * 1000
        self.state.price_history = [
            (t, p) for (t, p) in self.state.price_history 
            if t >= cutoff
        ]
    
    def _compute_realized_vol(self) -> float:
        """Compute realized volatility from price history"""
        prices = [p for (_, p) in self.state.price_history]
        if len(prices) < 2:
            return 0.0
        
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                r = math.log(prices[i] / prices[i-1])
                log_returns.append(r)
        
        if len(log_returns) < 1:
            return 0.0
        
        mean_r = sum(log_returns) / len(log_returns)
        var_r = sum((r - mean_r) ** 2 for r in log_returns) / len(log_returns)
        vol = math.sqrt(var_r)
        
        return vol
    
    def _compute_equity_usdc(self) -> float:
        """Compute current equity in USDC"""
        if self.state.mid_price is None:
            return self.state.cash_usdc + self.state.realized_pnl_usdc
        
        inventory_value = self.state.position_btc * self.state.mid_price
        return self.state.cash_usdc + self.state.realized_pnl_usdc + inventory_value
    
    def _risk_limits_violated(self) -> bool:
        """Check if risk limits are violated"""
        # Inventory cap
        if abs(self.state.position_btc) > self.mm_config.max_inventory_abs:
            logger.warning(f"Inventory limit violated: {abs(self.state.position_btc):.6f} > {self.mm_config.max_inventory_abs}")
            return True
        
        # Equity drawdown
        self.state.current_equity_usdc = self._compute_equity_usdc()
        if self.state.starting_equity_usdc is None:
            self.state.starting_equity_usdc = self.state.current_equity_usdc
        
        if self.state.starting_equity_usdc > 0:
            drawdown_pct = 100 * (self.state.starting_equity_usdc - self.state.current_equity_usdc) / self.state.starting_equity_usdc
            if drawdown_pct > self.mm_config.max_intraday_drawdown_pct:
                logger.warning(f"Drawdown limit violated: {drawdown_pct:.2f}% > {self.mm_config.max_intraday_drawdown_pct}%")
                return True
        
        # Realized loss cap
        if -self.state.realized_pnl_usdc > self.mm_config.max_realized_loss_usdc:
            logger.warning(f"Realized loss limit violated: {-self.state.realized_pnl_usdc:.2f} > {self.mm_config.max_realized_loss_usdc}")
            return True
        
        return False
    
    async def _handle_risk_violation(self):
        """Handle risk limit violation - kill switch"""
        logger.error("Risk limits violated - activating kill switch")
        await self._cancel_all_orders()
        # Could add hedging logic here if needed
    
    def _compute_base_spread_bps(self) -> float:
        """Compute base spread based on volatility"""
        vol_component_bps = 10000 * self.mm_config.vol_scale_factor * self.state.short_term_vol
        raw_spread_bps = self.mm_config.base_spread_bps + vol_component_bps
        
        # If volatility is extreme, widen further
        if self.state.short_term_vol > self.mm_config.max_allowed_vol:
            raw_spread_bps *= 2.0
        
        spread_bps = max(
            self.mm_config.min_spread_bps,
            min(raw_spread_bps, self.mm_config.max_spread_bps)
        )
        
        return spread_bps
    
    def _compute_inventory_skew_bps(self) -> float:
        """Compute inventory-based skew"""
        q = self.state.position_btc
        q_abs = abs(q)
        
        if q_abs < self.mm_config.inventory_bucket_1:
            factor = 0.0  # no skew
        elif q_abs < self.mm_config.inventory_bucket_2:
            factor = 1.0  # moderate skew
        else:
            factor = 2.0  # strong skew
        
        # Direction: long (q>0) => negative skew (move quotes down)
        direction = -1 if q > 0 else (1 if q < 0 else 0)
        
        # Magnitude of skew per BTC unit
        base_skew_per_btc_bps = 2.0
        skew_bps = direction * factor * base_skew_per_btc_bps * q_abs
        
        return skew_bps
    
    def _compute_quotes(self) -> Tuple[Optional[float], Optional[float], float, float]:
        """Compute bid/ask prices and sizes"""
        if self.state.mid_price is None:
            return None, None, 0.0, 0.0
        
        mid = self.state.mid_price
        
        # Base spread from vol
        base_spread_bps = self._compute_base_spread_bps()
        
        # Inventory skew
        skew_bps = self._compute_inventory_skew_bps()
        
        # Effective half-spread in price
        half_spread = (base_spread_bps / 20000.0) * mid  # (bps -> fraction -> half)
        
        # Skew in price (applied symmetrically)
        skew_px = (skew_bps / 10000.0) * mid
        
        # Final quotes
        bid_px = mid - half_spread + skew_px
        ask_px = mid + half_spread + skew_px
        
        # Quote size adapted to proximity to inventory cap
        inv_frac = min(1.0, abs(self.state.position_btc) / self.mm_config.max_inventory_abs)
        size_scale = max(0.2, 1.0 - inv_frac)  # smaller size when inventory large
        
        bid_size = self.mm_config.base_quote_size * size_scale
        ask_size = self.mm_config.base_quote_size * size_scale
        
        # If already long, be more cautious on bid size
        if self.state.position_btc > 0:
            bid_size *= 0.5
        elif self.state.position_btc < 0:
            ask_size *= 0.5
        
        # Cap sizes
        bid_size = min(bid_size, self.mm_config.max_quote_size)
        ask_size = min(ask_size, self.mm_config.max_quote_size)
        
        return bid_px, ask_px, bid_size, ask_size
    
    async def _round_price_to_tick(self, price: float) -> float:
        """Round price to tick size"""
        try:
            inst_info = await self._get_instrument_info()
            if inst_info:
                tick_size = inst_info.get('tick_size', 1.0)
            else:
                # Fallback: use 1.0 for BTC_USDC-PERPETUAL (1 USDC tick size)
                tick_size = 1.0
            
            if tick_size and tick_size > 0:
                rounded = round(price / tick_size) * tick_size
                # Ensure we don't lose precision
                if abs(rounded - price) > tick_size * 0.5:
                    logger.warning(f"Price rounding may be incorrect: {price:.2f} -> {rounded:.2f} (tick_size={tick_size})")
                return rounded
            
            return price
        except Exception as e:
            logger.warning(f"Error rounding price to tick: {e}, using fallback")
            # Fallback: round to 1.0 (1 USDC tick size for BTC_USDC-PERPETUAL)
            return round(price)
    
    async def _get_instrument_info(self) -> Optional[Dict]:
        """Get instrument information (cached)"""
        # Return cached info if available
        if self._cached_instrument_info:
            return self._cached_instrument_info
        
        try:
            # Try get_instruments with valid kinds only
            for kind in ["future", "option", "spot"]:
                try:
                    instruments = await self.client.get_instruments("BTC", kind)
                    for inst in instruments:
                        if inst['instrument_name'] == self.mm_config.instrument_name:
                            self._cached_instrument_info = inst
                            return inst
                except:
                    continue
            
            # If not found, use defaults for BTC_USDC-PERPETUAL
            if "BTC_USDC" in self.mm_config.instrument_name and "PERPETUAL" in self.mm_config.instrument_name:
                # Default values for BTC_USDC-PERPETUAL based on typical Deribit settings
                default_info = {
                    'instrument_name': self.mm_config.instrument_name,
                    'tick_size': 1.0,  # 1 USDC tick size
                    'min_trade_amount': 0.0001,  # 0.0001 BTC minimum
                    'contract_size': 0.0001,  # Contract size in BTC
                }
                self._cached_instrument_info = default_info
                logger.info(f"Using default instrument info for {self.mm_config.instrument_name}: tick_size=1.0, min_trade=0.0001")
                return default_info
            
            return None
        except Exception as e:
            logger.debug(f"Error getting instrument info: {e}")
            return None
    
    async def _check_and_update_quotes(self, now_ts_ms: int):
        """Check if quotes need updating and update them"""
        # Don't update too frequently
        if now_ts_ms - self.state.last_quote_update_ts < self.mm_config.quote_refresh_interval_ms:
            return
        
        # Periodic risk check
        if now_ts_ms - self.state.last_risk_check_ts > self.mm_config.risk_check_interval_secs * 1000:
            if self._risk_limits_violated():
                await self._handle_risk_violation()
                return
            self.state.last_risk_check_ts = now_ts_ms
        
        # Compute new quotes
        bid_px, ask_px, bid_size, ask_size = self._compute_quotes()
        
        if bid_px is None:
            return
        
        # If quotes too small, cancel all
        if bid_size <= 0.0 and ask_size <= 0.0:
            await self._cancel_all_orders()
            return
        
        # Round prices to tick size
        bid_px = await self._round_price_to_tick(bid_px)
        ask_px = await self._round_price_to_tick(ask_px)
        
        # Check if existing orders are stale or far from targets
        need_new_bid = True
        need_new_ask = True
        
        if self.state.active_bid_order_id is not None:
            age = now_ts_ms - (self.state.active_bid_timestamp or 0)
            if self.state.mid_price and self.state.active_bid_price:
                price_diff = abs(self.state.active_bid_price - bid_px) / self.state.mid_price
                if age < self.mm_config.quote_max_age_ms and price_diff < 0.0005:  # 5 bps tolerance
                    need_new_bid = False
        
        if self.state.active_ask_order_id is not None:
            age = now_ts_ms - (self.state.active_ask_timestamp or 0)
            if self.state.mid_price and self.state.active_ask_price:
                price_diff = abs(self.state.active_ask_price - ask_px) / self.state.mid_price
                if age < self.mm_config.quote_max_age_ms and price_diff < 0.0005:
                    need_new_ask = False
        
        # Cancel & replace as needed
        if need_new_bid and self.state.active_bid_order_id:
            try:
                await self.client.cancel_order(self.state.active_bid_order_id)
            except:
                pass
            self.state.active_bid_order_id = None
        
        if need_new_ask and self.state.active_ask_order_id:
            try:
                await self.client.cancel_order(self.state.active_ask_order_id)
            except:
                pass
            self.state.active_ask_order_id = None
        
        # Place new orders
        if need_new_bid and bid_size > 0:
            await self._place_bid_order(bid_px, bid_size, now_ts_ms)
        
        if need_new_ask and ask_size > 0:
            await self._place_ask_order(ask_px, ask_size, now_ts_ms)
        
        self.state.last_quote_update_ts = now_ts_ms
    
    async def _get_order_amount(self, size_btc: float) -> float:
        """
        Calculate order amount in the correct format for the instrument
        
        For BTC_USDC-PERPETUAL: amount is in BTC (base currency)
        For BTC-PERPETUAL: amount is in USD (quote currency)
        
        Returns:
            Amount in the correct currency/format, rounded to contract size
        """
        try:
            inst_info = await self._get_instrument_info()
            if not inst_info:
                logger.warning("Could not get instrument info, using size as-is")
                return size_btc
            
            instrument_name = inst_info['instrument_name']
            min_trade = inst_info.get('min_trade_amount', 0.0001)
            contract_size = inst_info.get('contract_size', 1.0)
            
            # Check if it's BTC_USDC (amount in BTC)
            if "BTC_USDC" in instrument_name:
                # Amount is in BTC for BTC_USDC instruments
                # Round to multiple of contract_size (which is the increment for amount)
                # For BTC_USDC-PERPETUAL, contract_size might be in BTC terms (e.g., 0.0001)
                # Use contract_size if > 0, otherwise fall back to min_trade_amount
                increment = contract_size if contract_size > 0 else min_trade
                amount_btc = round(size_btc / increment) * increment
                amount_btc = max(amount_btc, min_trade)
                logger.debug(f"Calculated BTC amount: {amount_btc:.6f} (from {size_btc:.6f}, increment={increment}, min_trade={min_trade})")
                return amount_btc
            
            # For BTC-PERPETUAL (inverse perpetual), amount is in USD
            elif instrument_name == "BTC-PERPETUAL":
                # Convert order_size (in BTC) to USD
                # We need current price, use mid_price if available
                price = self.state.mid_price if self.state.mid_price else 90000
                amount_usd = size_btc * price
                # Round to multiple of contract_size
                amount_usd = round(amount_usd / contract_size) * contract_size
                # Ensure minimum
                amount_usd = max(amount_usd, min_trade)
                logger.debug(f"Calculated USD amount: {amount_usd:.2f} (from {size_btc:.6f} BTC @ {price:.2f})")
                return amount_usd
            
            # For other futures/perpetuals, check if they're linear (USDC/USDT) or inverse
            elif "PERPETUAL" in instrument_name or "FUTURE" in instrument_name:
                # Check if it's a linear perpetual (USDC/USDT) - amount in base currency
                if "USDC" in instrument_name or "USDT" in instrument_name:
                    # Use contract_size for rounding
                    increment = contract_size if contract_size > 0 else min_trade
                    amount_btc = round(size_btc / increment) * increment
                    amount_btc = max(amount_btc, min_trade)
                    logger.debug(f"Calculated BTC amount (linear): {amount_btc:.6f} (increment={increment})")
                    return amount_btc
                else:
                    # Inverse perpetual - amount in USD
                    price = self.state.mid_price if self.state.mid_price else 90000
                    amount_usd = size_btc * price
                    amount_usd = round(amount_usd / contract_size) * contract_size
                    amount_usd = max(amount_usd, min_trade)
                    logger.debug(f"Calculated USD amount (inverse): {amount_usd:.2f}")
                    return amount_usd
            
            # Default: amount is in base currency (for options/spot)
            amount_btc = round(size_btc / min_trade) * min_trade
            return max(amount_btc, min_trade)
            
        except Exception as e:
            logger.error(f"Error calculating order amount: {e}", exc_info=True)
            return size_btc
    
    async def _place_bid_order(self, price: float, size: float, timestamp_ms: int):
        """Place a bid order"""
        try:
            # Calculate correct amount format and round to contract size
            amount = await self._get_order_amount(size)
            
            result = await self.client.place_order(
                instrument_name=self.mm_config.instrument_name,
                amount=amount,
                order_type=OrderType.LIMIT,
                price=price,
                side=OrderSide.BUY
            )
            
            order_id = result.get('order', {}).get('order_id')
            if order_id:
                self.state.active_bid_order_id = order_id
                self.state.active_bid_price = price
                self.state.active_bid_size = amount
                self.state.active_bid_timestamp = timestamp_ms
                logger.info(f"Placed bid: {order_id} @ {price:.2f} for {amount:.6f}")
        except Exception as e:
            logger.error(f"Error placing bid order: {e}")
    
    async def _place_ask_order(self, price: float, size: float, timestamp_ms: int):
        """Place an ask order"""
        try:
            # Calculate correct amount format and round to contract size
            amount = await self._get_order_amount(size)
            
            result = await self.client.place_order(
                instrument_name=self.mm_config.instrument_name,
                amount=amount,
                order_type=OrderType.LIMIT,
                price=price,
                side=OrderSide.SELL
            )
            
            order_id = result.get('order', {}).get('order_id')
            if order_id:
                self.state.active_ask_order_id = order_id
                self.state.active_ask_price = price
                self.state.active_ask_size = amount
                self.state.active_ask_timestamp = timestamp_ms
                logger.info(f"Placed ask: {order_id} @ {price:.2f} for {amount:.6f}")
        except Exception as e:
            logger.error(f"Error placing ask order: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            if self.state.active_bid_order_id:
                await self.client.cancel_order(self.state.active_bid_order_id)
                self.state.active_bid_order_id = None
            if self.state.active_ask_order_id:
                await self.client.cancel_order(self.state.active_ask_order_id)
                self.state.active_ask_order_id = None
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    async def _main_loop(self):
        """Main event loop"""
        while self.running:
            try:
                now_ts_ms = int(datetime.now().timestamp() * 1000)
                
                # Periodic quote update check (WebSocket events also trigger updates)
                await self._check_and_update_quotes(now_ts_ms)
                
                # Sleep to avoid busy-waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup(self):
        """Cleanup resources"""
        await self._cancel_all_orders()
        if self.orderbook_subscribed:
            # Unsubscribe if needed
            pass
    
    def get_status(self) -> Dict:
        """Get current market maker status"""
        return {
            'running': self.running,
            'instrument': self.mm_config.instrument_name,
            'position_btc': self.state.position_btc,
            'equity_usdc': self.state.current_equity_usdc,
            'mid_price': self.state.mid_price,
            'volatility': self.state.short_term_vol,
            'active_bid': self.state.active_bid_order_id,
            'active_ask': self.state.active_ask_order_id,
        }


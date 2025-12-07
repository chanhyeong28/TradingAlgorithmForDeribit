"""
Simple Market Making Strategy

A basic market making strategy that:
- Uses orderbook to determine bid/ask prices
- Places limit orders on both sides
- Manages orders (cancel and replace)
- Includes basic risk management
"""

import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .base import BaseStrategy, StrategyConfig
from ..models.market_data import OrderBook, OrderType, OrderSide, Trade
from ..core.client import DeribitClient
from ..risk.manager import RiskManager

logger = logging.getLogger(__name__)


class MarketMakerState(Enum):
    """Market maker state"""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class MarketMakerConfig(StrategyConfig):
    """Market maker configuration"""
    instrument_name: str = ""  # Instrument to trade
    order_size: float = 0.1  # Size per order
    spread_bps: float = 10.0  # Spread in basis points (0.1%)
    max_position: float = 1.0  # Maximum position size
    update_interval: float = 2.0  # Update interval in seconds
    max_spread_bps: float = 50.0  # Maximum spread to trade (0.5%)
    min_orderbook_depth: int = 5  # Minimum orderbook depth required
    skew_offset_bps: float = 0.0  # Offset to skew prices (positive = higher ask)
    
    def validate(self) -> List[str]:
        """Validate market maker configuration"""
        errors = super().validate()
        
        if not self.instrument_name or self.instrument_name == "":
            errors.append("Instrument name is required")
        
        if self.order_size <= 0:
            errors.append("Order size must be positive")
        
        if self.spread_bps <= 0:
            errors.append("Spread must be positive")
        
        if self.max_spread_bps <= self.spread_bps:
            errors.append("Max spread must be greater than target spread")
        
        return errors


class SimpleMarketMaker(BaseStrategy):
    """
    Simple market making strategy using orderbook
    
    Strategy:
    1. Get current orderbook
    2. Calculate mid price from best bid/ask
    3. Place limit orders at mid ± spread/2
    4. Monitor orders and update as needed
    5. Manage position limits
    """
    
    def __init__(self, config: MarketMakerConfig, client: DeribitClient,
                 risk_manager: RiskManager):
        super().__init__(config, client, risk_manager)
        self.mm_config: MarketMakerConfig = config
        self.state = MarketMakerState.IDLE
        self.active_orders: Dict[str, Dict] = {}  # order_id -> order_info
        self.bid_order_id: Optional[str] = None
        self.ask_order_id: Optional[str] = None
        self.last_orderbook: Optional[OrderBook] = None
        self.running = False
        
    async def generate_signals(self):
        """Market maker doesn't generate signals, it continuously quotes"""
        return []
    
    async def execute_signal(self, signal):
        """Market maker doesn't execute signals"""
        return False
    
    def get_strategy_positions(self) -> Dict[str, float]:
        """
        Get simulated positions for risk checking
        
        Returns:
            Dictionary of instrument_name -> position_size
        """
        try:
            # For market maker, we return empty dict as positions are managed
            # through the actual trading, not simulated
            # The risk manager will check actual positions from the client
            return {}
        except Exception as e:
            logger.error(f"Error getting strategy positions: {e}")
            return {}
    
    async def start(self):
        """Start the market maker"""
        if self.running:
            logger.warning("Market maker is already running")
            return
        
        self.running = True
        self.state = MarketMakerState.ACTIVE
        logger.info(f"Starting market maker for {self.mm_config.instrument_name}")
        
        try:
            while self.running:
                await self._update_quotes()
                await asyncio.sleep(self.mm_config.update_interval)
        except Exception as e:
            logger.error(f"Error in market maker loop: {e}", exc_info=True)
            self.state = MarketMakerState.ERROR
        finally:
            await self._cancel_all_orders()
            self.running = False
            self.state = MarketMakerState.IDLE
    
    async def stop(self):
        """Stop the market maker"""
        logger.info("Stopping market maker...")
        self.running = False
        await self._cancel_all_orders()
        self.state = MarketMakerState.IDLE
    
    async def _update_quotes(self):
        """Update bid/ask quotes based on current orderbook"""
        try:
            # Get current orderbook
            orderbook = await self.client.get_order_book(
                self.mm_config.instrument_name,
                depth=self.mm_config.min_orderbook_depth
            )
            
            if not orderbook.bids or not orderbook.asks:
                logger.warning("Insufficient orderbook depth")
                return
            
            # Check if spread is acceptable
            if orderbook.spread is None:
                logger.warning("Cannot calculate spread")
                return
            
            mid_price = orderbook.mid_price
            if mid_price is None:
                logger.warning("Cannot calculate mid price")
                return
            
            # Calculate spread percentage
            spread_pct = (orderbook.spread / mid_price) * 10000  # in basis points
            
            if spread_pct > self.mm_config.max_spread_bps:
                logger.debug(f"Spread too wide: {spread_pct:.2f} bps > {self.mm_config.max_spread_bps} bps")
                await self._cancel_all_orders()
                return
            
            # Check position limits
            positions = await self.client.get_positions("BTC")
            current_position = 0.0
            for pos in positions:
                if pos.get('instrument_name') == self.mm_config.instrument_name:
                    current_position = pos.get('size', 0.0)
                    break
            
            # Get instrument info for tick size
            inst_info = await self._get_instrument_info()
            tick_size = inst_info.get('tick_size', 0.1) if inst_info else 0.1
            
            # Calculate bid/ask prices
            half_spread = (self.mm_config.spread_bps / 10000) * mid_price
            skew_offset = (self.mm_config.skew_offset_bps / 10000) * mid_price
            
            bid_price = mid_price - half_spread - skew_offset
            ask_price = mid_price + half_spread + skew_offset
            
            # Round prices to tick size
            bid_price = await self._round_price_to_tick(bid_price, tick_size)
            ask_price = await self._round_price_to_tick(ask_price, tick_size)
            
            # Adjust for position (skew prices to reduce position)
            if abs(current_position) > 0.1:
                # If we have a long position, lower ask price to encourage selling
                # If we have a short position, raise bid price to encourage buying
                position_skew = (current_position / self.mm_config.max_position) * half_spread * 0.5
                bid_price += position_skew
                ask_price -= position_skew
            
            # Check if we should place orders
            should_place_bid = abs(current_position) < self.mm_config.max_position
            should_place_ask = abs(current_position) < self.mm_config.max_position
            
            # Update or place bid order
            if should_place_bid:
                if self.bid_order_id and self.bid_order_id in self.active_orders:
                    # Check if price changed significantly
                    old_bid = self.active_orders[self.bid_order_id].get('price', 0)
                    if abs(bid_price - old_bid) / old_bid > 0.001:  # 0.1% change
                        await self._cancel_order(self.bid_order_id)
                        self.bid_order_id = None
                
                if not self.bid_order_id:
                    await self._place_bid_order(bid_price)
            else:
                if self.bid_order_id:
                    await self._cancel_order(self.bid_order_id)
                    self.bid_order_id = None
            
            # Update or place ask order
            if should_place_ask:
                if self.ask_order_id and self.ask_order_id in self.active_orders:
                    # Check if price changed significantly
                    old_ask = self.active_orders[self.ask_order_id].get('price', 0)
                    if abs(ask_price - old_ask) / old_ask > 0.001:  # 0.1% change
                        await self._cancel_order(self.ask_order_id)
                        self.ask_order_id = None
                
                if not self.ask_order_id:
                    await self._place_ask_order(ask_price)
            else:
                if self.ask_order_id:
                    await self._cancel_order(self.ask_order_id)
                    self.ask_order_id = None
            
            self.last_orderbook = orderbook
            
        except Exception as e:
            logger.error(f"Error updating quotes: {e}", exc_info=True)
    
    async def _get_instrument_info(self) -> Optional[Dict]:
        """Get instrument information"""
        try:
            # Try futures first
            instruments = await self.client.get_instruments("BTC", "future")
            for inst in instruments:
                if inst['instrument_name'] == self.mm_config.instrument_name:
                    return inst
            
            # Try options
            instruments = await self.client.get_instruments("BTC", "option")
            for inst in instruments:
                if inst['instrument_name'] == self.mm_config.instrument_name:
                    return inst
            
            return None
        except Exception as e:
            logger.error(f"Error getting instrument info: {e}")
            return None
    
    async def _round_price_to_tick(self, price: float, tick_size: float) -> float:
        """Round price to conform to tick size"""
        if tick_size <= 0:
            return price
        return round(price / tick_size) * tick_size
    
    async def _get_order_amount(self, price: float) -> float:
        """
        Calculate order amount in the correct format for the instrument
        
        For BTC_USDC-PERPETUAL: amount is in BTC (base currency)
        For BTC-PERPETUAL: amount is in USD (quote currency)
        For options/spot: amount is in base currency
        
        Returns:
            Amount in the correct currency/format
        """
        try:
            inst_info = await self._get_instrument_info()
            if not inst_info:
                # Default: assume amount is in base currency
                return self.mm_config.order_size
            
            instrument_name = inst_info['instrument_name']
            min_trade = inst_info.get('min_trade_amount', 0.0001)
            
            # Check if it's BTC_USDC (amount in BTC)
            if "BTC_USDC" in instrument_name:
                # Amount is in BTC for BTC_USDC instruments
                # Ensure it meets minimum trade amount
                amount_btc = max(self.mm_config.order_size, min_trade)
                return amount_btc
            
            # For BTC-PERPETUAL (inverse perpetual), amount is in USD
            elif instrument_name == "BTC-PERPETUAL":
                contract_size = inst_info.get('contract_size', 10.0)
                # Convert order_size (in BTC) to USD
                amount_usd = self.mm_config.order_size * price
                # Round to multiple of contract_size
                amount_usd = round(amount_usd / contract_size) * contract_size
                # Ensure minimum
                amount_usd = max(amount_usd, min_trade)
                return amount_usd
            
            # For other futures/perpetuals, check if they're linear (USDC/USDT) or inverse
            elif "PERPETUAL" in instrument_name or "FUTURE" in instrument_name:
                # Check if it's a linear perpetual (USDC/USDT) - amount in base currency
                if "USDC" in instrument_name or "USDT" in instrument_name:
                    amount_btc = max(self.mm_config.order_size, min_trade)
                    return amount_btc
                else:
                    # Inverse perpetual - amount in USD
                    contract_size = inst_info.get('contract_size', 10.0)
                    amount_usd = self.mm_config.order_size * price
                    amount_usd = round(amount_usd / contract_size) * contract_size
                    amount_usd = max(amount_usd, min_trade)
                    return amount_usd
            
            # Default: amount is in base currency (for options/spot)
            return max(self.mm_config.order_size, min_trade)
            
        except Exception as e:
            logger.error(f"Error calculating order amount: {e}")
            # Fallback: use order_size directly
            return self.mm_config.order_size
    
    async def _place_bid_order(self, price: float):
        """Place a bid (buy) order"""
        try:
            # Calculate correct amount format
            amount = await self._get_order_amount(price)
            
            result = await self.client.place_order(
                instrument_name=self.mm_config.instrument_name,
                amount=amount,
                order_type=OrderType.LIMIT,
                price=price,
                side=OrderSide.BUY
            )
            
            order_id = result.get('order', {}).get('order_id') or result.get('order_id')
            if order_id:
                self.bid_order_id = order_id
                self.active_orders[order_id] = {
                    'side': 'buy',
                    'price': price,
                    'amount': amount,
                    'timestamp': datetime.now()
                }
                logger.info(f"Placed bid order: {order_id} @ {price:.2f} for {amount}")
            else:
                logger.warning(f"Bid order placed but no order_id returned: {result}")
                
        except Exception as e:
            logger.error(f"Error placing bid order: {e}")
    
    async def _place_ask_order(self, price: float):
        """Place an ask (sell) order"""
        try:
            # Calculate correct amount format
            amount = await self._get_order_amount(price)
            
            result = await self.client.place_order(
                instrument_name=self.mm_config.instrument_name,
                amount=amount,
                order_type=OrderType.LIMIT,
                price=price,
                side=OrderSide.SELL
            )
            
            order_id = result.get('order', {}).get('order_id') or result.get('order_id')
            if order_id:
                self.ask_order_id = order_id
                self.active_orders[order_id] = {
                    'side': 'sell',
                    'price': price,
                    'amount': amount,
                    'timestamp': datetime.now()
                }
                logger.info(f"Placed ask order: {order_id} @ {price:.2f} for {amount}")
            else:
                logger.warning(f"Ask order placed but no order_id returned: {result}")
                
        except Exception as e:
            logger.error(f"Error placing ask order: {e}")
    
    async def _cancel_order(self, order_id: str):
        """Cancel a specific order"""
        try:
            await self.client.cancel_order(order_id)
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            if order_id == self.bid_order_id:
                self.bid_order_id = None
            if order_id == self.ask_order_id:
                self.ask_order_id = None
            logger.info(f"Cancelled order: {order_id}")
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            # Cancel bid order
            if self.bid_order_id:
                await self._cancel_order(self.bid_order_id)
            
            # Cancel ask order
            if self.ask_order_id:
                await self._cancel_order(self.ask_order_id)
            
            # Cancel any remaining orders
            for order_id in list(self.active_orders.keys()):
                await self._cancel_order(order_id)
                
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
    
    def get_status(self) -> Dict:
        """Get current market maker status"""
        return {
            'state': self.state.value,
            'instrument': self.mm_config.instrument_name,
            'active_orders': len(self.active_orders),
            'bid_order_id': self.bid_order_id,
            'ask_order_id': self.ask_order_id,
            'last_orderbook': {
                'best_bid': self.last_orderbook.best_bid if self.last_orderbook else None,
                'best_ask': self.last_orderbook.best_ask if self.last_orderbook else None,
                'spread': self.last_orderbook.spread if self.last_orderbook else None,
            } if self.last_orderbook else None
        }


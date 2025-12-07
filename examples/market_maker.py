#!/usr/bin/env python3
"""
Simple Market Making Strategy Example

This script demonstrates a basic market making strategy on Deribit testnet.
The strategy:
- Monitors the orderbook for a specified instrument
- Places limit orders on both bid and ask sides
- Adjusts prices based on current position
- Manages risk with position limits

Usage:
    python examples/market_maker.py

Make sure to set DERIBIT_TESTNET=true in your .env file!
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deribit_trading_toolkit import DeribitClient, DeribitAuth, ConfigManager
from deribit_trading_toolkit.strategies.market_maker import SimpleMarketMaker, MarketMakerConfig
from deribit_trading_toolkit.risk.manager import RiskManager, RiskLimits

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Force testnet mode
os.environ['DERIBIT_TESTNET'] = 'true'


async def main():
    """Main function to run market maker"""
    client = None
    market_maker = None
    
    try:
        logger.info("=" * 70)
        logger.info("Simple Market Making Strategy - Testnet")
        logger.info("=" * 70)
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        config.deribit.testnet = True
        
        logger.info(f"Testnet mode: {config.deribit.testnet}")
        logger.info(f"API URL: {config.deribit.effective_api_url}")
        
        # Initialize client
        auth = DeribitAuth(
            client_id=config.deribit.effective_client_id,
            private_key_path=config.deribit.effective_private_key_path,
            private_key=config.deribit.effective_private_key
        )
        
        client = DeribitClient(config.deribit, auth)
        await client.connect()
        logger.info("✅ Connected to Deribit")
        
        # Get account summary
        account = await client.get_account_summary("BTC")
        logger.info(f"Account equity: {account.get('equity', 0):.4f} BTC")
        logger.info(f"Available balance: {account.get('available_funds', 0):.4f} BTC")
        
        # Get available instruments
        instruments = await client.get_instruments("BTC", "future")
        if not instruments:
            logger.error("No instruments found")
            return
        
        # Try to find BTC_USDC-PERPETUAL first (for USDC profits)
        # User wants to trade BTC_USDC-PERPETUAL with amount in BTC
        instrument_name = None
        for inst in instruments:
            if "BTC_USDC" in inst['instrument_name'] and "PERPETUAL" in inst['instrument_name']:
                instrument_name = inst['instrument_name']
                logger.info(f"✅ Found BTC_USDC perpetual: {instrument_name}")
                break
        
        if not instrument_name:
            # Fallback to BTC-PERPETUAL (note: amount will be in USD for this)
            if "BTC-PERPETUAL" in [inst['instrument_name'] for inst in instruments]:
                instrument_name = "BTC-PERPETUAL"
                logger.info(f"⚠️  BTC_USDC-PERPETUAL not found, using BTC-PERPETUAL (amount in USD)")
            else:
                instrument_name = instruments[0]['instrument_name'] if instruments else "BTC-PERPETUAL"
                logger.info(f"⚠️  Using {instrument_name}")
        
        logger.info(f"Selected instrument: {instrument_name}")
        
        # Show instrument details
        for inst in instruments:
            if inst['instrument_name'] == instrument_name:
                logger.info(f"Instrument details:")
                logger.info(f"  Tick size: {inst.get('tick_size', 'N/A')}")
                logger.info(f"  Min trade amount: {inst.get('min_trade_amount', 'N/A')}")
                logger.info(f"  Contract size: {inst.get('contract_size', 'N/A')}")
                break
        
        # Get current orderbook to show initial state
        orderbook = await client.get_order_book(instrument_name, depth=5)
        logger.info(f"Current orderbook:")
        logger.info(f"  Best bid: {orderbook.best_bid}")
        logger.info(f"  Best ask: {orderbook.best_ask}")
        logger.info(f"  Spread: {orderbook.spread:.2f} ({orderbook.spread/orderbook.mid_price*10000:.2f} bps)")
        
        # Create market maker configuration
        mm_config = MarketMakerConfig(
            name="SimpleMarketMaker",
            instrument_name=instrument_name,
            order_size=0.0001,  # Small size for testnet (in BTC, will be converted appropriately)
            spread_bps=20.0,  # 0.2% spread
            max_position=0.1,  # Max 0.1 BTC position
            update_interval=3.0,  # Update every 3 seconds
            max_spread_bps=100.0,  # Don't trade if spread > 1%
            min_orderbook_depth=3,
            skew_offset_bps=0.0
        )
        
        # Create risk limits
        risk_limits = RiskLimits(
            max_position_size=0.1,
            max_daily_loss=0.01,
            max_positions=5,
            margin_buffer=1.2,
            stop_loss_percentage=0.05
        )
        
        # Create risk manager
        risk_manager = RiskManager(client, risk_limits)
        
        # Create and start market maker
        market_maker = SimpleMarketMaker(mm_config, client, risk_manager)
        
        logger.info("\n" + "=" * 70)
        logger.info("Starting Market Maker")
        logger.info("=" * 70)
        logger.info(f"Instrument: {instrument_name}")
        logger.info(f"Order size: {mm_config.order_size}")
        logger.info(f"Target spread: {mm_config.spread_bps} bps")
        logger.info(f"Update interval: {mm_config.update_interval}s")
        logger.info(f"Max position: {mm_config.max_position}")
        logger.info("\nPress Ctrl+C to stop...\n")
        
        # Start market maker in background
        market_maker_task = asyncio.create_task(market_maker.start())
        
        # Monitor status
        try:
            while True:
                await asyncio.sleep(10)
                status = market_maker.get_status()
                logger.info(f"Status: {status['state']} | Active orders: {status['active_orders']}")
                
                # Get current position
                positions = await client.get_positions("BTC")
                for pos in positions:
                    if pos.get('instrument_name') == instrument_name:
                        size = pos.get('size', 0)
                        if abs(size) > 0.001:
                            logger.info(f"Current position: {size:.4f} BTC")
                        break
                        
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal, stopping market maker...")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Cleanup
        if market_maker:
            await market_maker.stop()
            logger.info("Market maker stopped")
        
        if client:
            await client.disconnect()
            logger.info("Client disconnected")
        
        logger.info("Exiting...")


if __name__ == "__main__":
    asyncio.run(main())


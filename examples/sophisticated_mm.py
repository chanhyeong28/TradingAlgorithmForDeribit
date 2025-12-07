#!/usr/bin/env python3
"""
Sophisticated Market Making Strategy Example

This script implements a theory-driven market making strategy with:
- Real-time orderbook subscription via WebSocket
- Volatility-based spread adjustment
- Inventory-based quote skewing
- Risk management with kill switches

Usage:
    python examples/sophisticated_mm.py

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
from deribit_trading_toolkit.strategies.sophisticated_mm import SophisticatedMarketMaker, SophisticatedMMConfig
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
    """Main function to run sophisticated market maker"""
    client = None
    market_maker = None
    
    try:
        logger.info("=" * 70)
        logger.info("Sophisticated Market Making Strategy - Testnet")
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
        
        # Find BTC_USDC-PERPETUAL (required for this strategy)
        instrument_name = "BTC_USDC-PERPETUAL"
        
        # Verify instrument exists by trying to access its orderbook
        # (BTC_USDC-PERPETUAL may not appear in get_instruments but is accessible)
        inst_info = None
        found = False
        
        try:
            # Try to get orderbook - if this works, the instrument exists
            orderbook = await client.get_order_book(instrument_name, depth=1)
            if orderbook and orderbook.mid_price:
                found = True
                logger.info(f"✅ Verified {instrument_name} exists (via orderbook)")
                logger.info(f"  Current mid price: {orderbook.mid_price:.2f} USDC")
                logger.info(f"  Best bid: {orderbook.best_bid:.2f} USDC")
                logger.info(f"  Best ask: {orderbook.best_ask:.2f} USDC")
                
                # Try to find instrument details in get_instruments
                for kind in ["future", "option"]:
                    try:
                        instruments = await client.get_instruments("BTC", kind)
                        for inst in instruments:
                            if inst['instrument_name'] == instrument_name:
                                inst_info = inst
                                logger.info(f"  Tick size: {inst_info.get('tick_size', 'N/A')}")
                                logger.info(f"  Min trade: {inst_info.get('min_trade_amount', 'N/A')}")
                                logger.info(f"  Contract size: {inst_info.get('contract_size', 'N/A')}")
                                break
                        if inst_info:
                            break
                    except:
                        continue
                
                if not inst_info:
                    logger.info(f"  (Instrument details not found in get_instruments, using defaults)")
        except Exception as e:
            logger.error(f"❌ Cannot access {instrument_name}: {e}")
            logger.error("This strategy requires BTC_USDC-PERPETUAL. Please check if it's available on testnet.")
            return
        
        if not found:
            logger.error(f"❌ {instrument_name} not accessible. Cannot run sophisticated market maker.")
            return
        
        # Create market maker configuration
        mm_config = SophisticatedMMConfig(
            name="SophisticatedMM",
            instrument_name=instrument_name,
            quote_currency="USDC",
            
            # Risk / inventory
            target_inventory=0.0,
            max_inventory_abs=0.1,  # Max 0.1 BTC position
            inventory_bucket_1=0.025,  # 0.025 BTC threshold
            inventory_bucket_2=0.05,   # 0.05 BTC threshold
            
            # Spread & size
            base_spread_bps=5.0,  # 5 bps base spread
            min_spread_bps=2.0,
            max_spread_bps=50.0,
            base_quote_size=0.01,  # 0.01 BTC per quote
            max_quote_size=0.05,
            
            # Volatility
            vol_lookback_secs=60,
            vol_scale_factor=1.0,
            max_allowed_vol=0.05,  # 5% max vol
            
            # Time controls
            quote_refresh_interval_ms=300,
            quote_max_age_ms=2000,
            risk_check_interval_secs=5,
            
            # Kill switch
            max_intraday_drawdown_pct=5.0,
            max_realized_loss_usdc=500.0,
        )
        
        # Create risk manager
        risk_limits = RiskLimits(
            max_position_size=0.1,
            max_daily_loss=0.01,
            max_positions=1,
            margin_buffer=1.2
        )
        risk_manager = RiskManager(client, risk_limits)
        
        # Create and start market maker
        market_maker = SophisticatedMarketMaker(mm_config, client, risk_manager)
        
        logger.info("\n" + "=" * 70)
        logger.info("Starting Sophisticated Market Maker")
        logger.info("=" * 70)
        logger.info(f"Instrument: {instrument_name}")
        logger.info(f"Base spread: {mm_config.base_spread_bps} bps")
        logger.info(f"Base quote size: {mm_config.base_quote_size} BTC")
        logger.info(f"Max inventory: {mm_config.max_inventory_abs} BTC")
        logger.info("\nPress Ctrl+C to stop...\n")
        
        # Start market maker
        await market_maker.start()
        
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


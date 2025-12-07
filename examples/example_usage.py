#!/usr/bin/env python3
"""
Example usage of the Deribit Trading Toolkit

This script demonstrates how to use the refactored OOP trading system.
"""

import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deribit_trading_toolkit import (
    TradingApp, AppConfig, ConfigManager, DeribitClient, DeribitAuth,
    RiskReversalStrategy, RiskReversalConfig, RiskManager, RiskLimits,
    VolatilityAnalyzer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Example of basic toolkit usage"""
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Create and run the trading application
        app = TradingApp(config)
        await app.run()
        
    except Exception as e:
        logger.error(f"Error in basic usage example: {e}")


async def example_manual_setup():
    """Example of manual component setup"""
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Initialize authentication
        auth = DeribitAuth(
            client_id=config.deribit.effective_client_id,
            private_key_path=config.deribit.effective_private_key_path,
            private_key=config.deribit.effective_private_key
        )
        
        # Initialize client
        async with DeribitClient(config.deribit, auth) as client:
            # Get option chain
            options = await client.get_option_chain("BTC")
            logger.info(f"Retrieved {len(options)} BTC options")
            
            # Initialize analytics
            volatility_analyzer = VolatilityAnalyzer()
            
            # Initialize risk management
            risk_limits = RiskLimits()
            risk_manager = RiskManager(client, risk_limits)
            
            # Example: Build volatility curve
            if options:
                # Get options for a specific expiration
                expiration = options[0].expiration_timestamp
                expiration_options = [opt for opt in options if opt.expiration_timestamp == expiration]
                
                underlying_price = 50000  # Example price
                curve = volatility_analyzer.build_volatility_curve(
                    expiration_options, underlying_price, expiration
                )
                
                if curve:
                    logger.info(f"Built volatility curve with ATM slope: {curve.slope}")
            
    except Exception as e:
        logger.error(f"Error in manual setup example: {e}")


async def example_strategy_setup():
    """Example of strategy setup and execution"""
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Initialize components
        auth = DeribitAuth(
            client_id=config.deribit.effective_client_id,
            private_key_path=config.deribit.effective_private_key_path,
            private_key=config.deribit.effective_private_key
        )
        
        async with DeribitClient(config.deribit, auth) as client:
            # Initialize risk management
            risk_limits = RiskLimits()
            risk_manager = RiskManager(client, risk_limits)
            
            # Create strategy configuration
            strategy_config = RiskReversalConfig(
                name="Example Risk Reversal",
                enabled=True,
                position_size=0.1,
                near_expiration="27SEP24",
                far_expiration="25OCT24",
                spread_way="SHORT"
            )
            
            # Create strategy
            strategy = RiskReversalStrategy(
                config=strategy_config,
                client=client,
                risk_manager=risk_manager
            )
            
            # Run strategy
            result = await strategy.run_strategy()
            logger.info(f"Strategy execution result: {result}")
            
            # Get performance metrics
            metrics = strategy.get_performance_metrics()
            logger.info(f"Strategy performance: {metrics}")
            
    except Exception as e:
        logger.error(f"Error in strategy setup example: {e}")


async def main():
    """Main example function"""
    print("Deribit Trading Toolkit Examples")
    print("=" * 40)
    
    # Example 1: Basic usage with full application
    print("\n1. Basic Usage Example:")
    print("Running full trading application...")
    # Uncomment to run: await example_basic_usage()
    
    # Example 2: Manual component setup
    print("\n2. Manual Setup Example:")
    print("Setting up components manually...")
    await example_manual_setup()
    
    # Example 3: Strategy setup
    print("\n3. Strategy Setup Example:")
    print("Setting up and running strategy...")
    await example_strategy_setup()
    
    print("\nExamples completed!")


if __name__ == "__main__":
    asyncio.run(main())

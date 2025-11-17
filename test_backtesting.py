#!/usr/bin/env python3
"""
Test script for BacktestingEnvironment

This script demonstrates how to use the BacktestingEnvironment class
to construct backtesting environments for options trading.
"""

import asyncio
import logging
from deribit_trading_toolkit import BacktestingEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run backtesting environment construction"""
    try:
        logger.info("=" * 60)
        logger.info("Backtesting Environment Construction")
        logger.info("=" * 60)
        
        async with BacktestingEnvironment() as env:
            # Build backtesting environment for specified expirations
            results = await env.build_environment(
                expirations=["26DEC25", "27MAR26"],
                days_back=1,
                resolution=60,  # 1-minute bars
                time_window_seconds=3600,  # Build curve every hour
                save_prices=True,
                save_curves=True
            )
            
            # Print individual results
            logger.info("\n" + "=" * 60)
            logger.info("Individual Results")
            logger.info("=" * 60)
            for result in results:
                if result.success:
                    logger.info(
                        f"{result.expiration_str}: "
                        f"{result.futures_records} futures, "
                        f"{result.option_records} options, "
                        f"{result.curves_saved} curves"
                    )
                else:
                    logger.error(f"{result.expiration_str}: Failed - {result.error}")
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())


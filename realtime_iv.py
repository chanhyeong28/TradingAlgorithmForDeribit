#!/usr/bin/env python3
"""
Real-time Implied Volatility Visualization - Entry Point

This script provides a simple entry point to the RealTimeIVApp.
For advanced usage, import RealTimeIVApp directly from deribit_trading_toolkit.
"""

import asyncio
import logging

from deribit_trading_toolkit import RealTimeIVApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """
    Main entry point for real-time IV visualization.
    
    Uses automatic expiration selection based on rules:
    1. Daily options this week (Mon-Fri)
    2. Weekly options this month (every Friday)
    3. Monthly options in next 2 months (last Friday)
    4. Quarterly options within a year (last Friday of each quarter)
    """
    app = None
    try:
        logger.info("Initializing RealTimeIVApp...")
        app = RealTimeIVApp(
            refresh_seconds=2,
            futures_refresh_seconds=15,
            use_auto_expirations=True
        )
        
        logger.info("Starting application...")
        await app.start()
        
        logger.info("Application started successfully!")
        logger.info("Dashboard should be available at http://127.0.0.1:8050")
        logger.info("If the browser didn't open automatically, please open it manually.")
        
        # Access SSR results
        try:
            ssr_daily = app.ssr_calculator.get_current_SSR_daily()
            eod_results = app.ssr_calculator.compute_SSR_EOD_all()
            logger.info(f"SSR daily: {ssr_daily}")
            logger.info(f"SSR EOD: {eod_results}")
        except Exception as e:
            logger.warning(f"Could not get SSR results: {e}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        if app:
            try:
                await app.stop()
                logger.info("Application stopped")
            except Exception as e:
                logger.error(f"Error stopping application: {e}")


if __name__ == "__main__":
    asyncio.run(main())

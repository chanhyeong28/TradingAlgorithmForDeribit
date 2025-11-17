"""
Backtesting Environment Module

Provides OOP interface for constructing backtesting environments for options trading.
Handles collecting historical data, building volatility curves, and saving to MySQL.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..analytics.backtesting import HistoricalDataCollector

logger = logging.getLogger(__name__)


@dataclass
class BacktestingResult:
    """Result of backtesting environment construction"""
    expiration_str: str
    futures_records: int
    option_records: int
    curves_saved: int
    success: bool
    error: Optional[str] = None


class BacktestingEnvironment:
    """
    Constructs backtesting environments for options trading.
    
    This class handles:
    - Collecting historical futures prices
    - Collecting historical option prices
    - Building volatility curves over time
    - Saving all data to MySQL database
    
    Example:
        ```python
        async with BacktestingEnvironment() as env:
            result = await env.build_environment(
                expirations=["26DEC25", "27MAR26"],
                days_back=40,
                resolution=60
            )
        ```
    """
    
    def __init__(self, collector: Optional[HistoricalDataCollector] = None):
        """
        Initialize backtesting environment builder.
        
        Args:
            collector: HistoricalDataCollector instance (optional, will create if not provided)
        """
        self.collector = collector
        self._should_close_collector = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.collector:
            self.collector = HistoricalDataCollector()
            await self.collector.__aenter__()
            self._should_close_collector = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._should_close_collector and self.collector:
            await self.collector.__aexit__(exc_type, exc_val, exc_tb)
    
    async def build_environment(
        self,
        expirations: List[str],
        days_back: int,
        resolution: int = 60,
        time_window_seconds: int = 3600,
        save_prices: bool = True,
        save_curves: bool = True
    ) -> List[BacktestingResult]:
        """
        Build backtesting environment for given expirations.
        
        This method:
        1. Collects and saves futures prices
        2. Collects and saves option prices
        3. Builds and saves volatility curves
        
        Args:
            expirations: List of expiration strings (e.g., ["26DEC25", "27MAR26"])
            days_back: Number of days to look back
            resolution: Resolution in seconds for data collection (default: 60)
            time_window_seconds: Time window for building each curve (default: 3600 = 1 hour)
            save_prices: Whether to save price data (default: True)
            save_curves: Whether to save volatility curves (default: True)
            
        Returns:
            List of BacktestingResult objects, one per expiration
        """
        # Calculate timestamps
        end_timestamp = int(datetime.now().timestamp() * 1000)
        start_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        logger.info(f"Building backtesting environment for {len(expirations)} expirations")
        logger.info(f"Expirations: {', '.join(expirations)}")
        logger.info(f"Period: {days_back} days (from {datetime.fromtimestamp(start_timestamp/1000)} to {datetime.fromtimestamp(end_timestamp/1000)})")
        logger.info(f"Resolution: {resolution}s, Time window: {time_window_seconds}s")
        logger.info(f"Save prices: {save_prices}, Save curves: {save_curves}\n")
        
        results = []
        
        for expiration_str in expirations:
            logger.info("\n" + "=" * 60)
            logger.info(f"Processing expiration: {expiration_str}")
            logger.info("=" * 60)
            
            try:
                result = await self._build_for_expiration(
                    expiration_str=expiration_str,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    resolution=resolution,
                    time_window_seconds=time_window_seconds,
                    save_prices=save_prices,
                    save_curves=save_curves
                )
                results.append(result)
                
                if result.success:
                    logger.info(
                        f"\n✓ Completed {expiration_str}: "
                        f"{result.futures_records} futures, "
                        f"{result.option_records} options, "
                        f"{result.curves_saved} curves"
                    )
                else:
                    logger.error(f"\n✗ Failed {expiration_str}: {result.error}")
                    
            except Exception as e:
                logger.error(f"Error processing {expiration_str}: {e}", exc_info=True)
                results.append(BacktestingResult(
                    expiration_str=expiration_str,
                    futures_records=0,
                    option_records=0,
                    curves_saved=0,
                    success=False,
                    error=str(e)
                ))
        
        # Final summary
        self._print_summary(results)
        
        return results
    
    async def _build_for_expiration(
        self,
        expiration_str: str,
        start_timestamp: int,
        end_timestamp: int,
        resolution: int,
        time_window_seconds: int,
        save_prices: bool,
        save_curves: bool
    ) -> BacktestingResult:
        """Build environment for a single expiration"""
        # Parse expiration to get expiration timestamp
        expiration_epoch = self.collector._extract_expiration(f"BTC-{expiration_str}")
        if not expiration_epoch:
            return BacktestingResult(
                expiration_str=expiration_str,
                futures_records=0,
                option_records=0,
                curves_saved=0,
                success=False,
                error=f"Failed to parse expiration: {expiration_str}"
            )
        
        futures_records = 0
        option_records = 0
        curves_saved = 0
        
        # 1. Collect and save futures prices
        if save_prices:
            logger.info(f"\n[1/3] Collecting futures prices for {expiration_str}...")
            futures_name = f"BTC-{expiration_str}"
            futures_data = await self.collector.collect_instrument_data(
                instrument_name=futures_name,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                months_back=0,
                resolution=resolution
            )
            
            if not futures_data:
                # Try perpetual if future doesn't exist
                logger.info(f"Future {futures_name} not found, trying perpetual...")
                futures_data = await self.collector.collect_instrument_data(
                    instrument_name="BTC-PERPETUAL",
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    months_back=0,
                    resolution=resolution
                )
                futures_name = "BTC-PERPETUAL"
            
            if futures_data:
                futures_records = self.collector.save_futures_prices(
                    instrument_name=futures_name,
                    data_points=futures_data,
                    resolution_seconds=resolution,
                    expiration_timestamp=expiration_epoch if 'PERPETUAL' not in futures_name else None
                )
                logger.info(f"✓ Saved {futures_records} futures price records")
            else:
                logger.warning(f"⚠ No futures data collected for {expiration_str}")
        
        # 2. Collect and save option prices
        if save_prices:
            logger.info(f"\n[2/3] Collecting option prices for {expiration_str}...")
            instruments = await self.collector.client.get_instruments(currency="BTC", kind="option")
            target_options = [instr.get('instrument_name', '') for instr in instruments 
                             if expiration_str.upper() in instr.get('instrument_name', '').upper()]
            
            logger.info(f"Found {len(target_options)} options for {expiration_str}")
            
            option_type_map = {'C': 'call', 'P': 'put'}
            
            for idx, option_name in enumerate(target_options, 1):
                if idx % 10 == 0:
                    logger.info(f"  Progress: {idx}/{len(target_options)} options processed...")
                
                option_data = await self.collector.collect_instrument_data(
                    instrument_name=option_name,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    months_back=0,
                    resolution=resolution
                )
                
                if option_data:
                    # Extract option details
                    option_type_raw = self.collector._extract_option_type(option_name)
                    strike = self.collector._extract_strike_from_name(option_name)
                    
                    # Convert C/P to call/put
                    option_type = option_type_map.get(option_type_raw, 'call')
                    
                    if strike and option_type_raw in ['C', 'P']:
                        records = self.collector.save_option_prices(
                            instrument_name=option_name,
                            data_points=option_data,
                            resolution_seconds=resolution,
                            expiration_timestamp=expiration_epoch,
                            option_type=option_type,
                            strike_price=strike
                        )
                        option_records += records
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            logger.info(f"✓ Saved {option_records} option price records for {expiration_str}")
        
        # 3. Build and save volatility curves
        if save_curves:
            logger.info(f"\n[3/3] Building and saving volatility curves for {expiration_str}...")
            curves = await self.collector.build_volatility_curves_over_period(
                expiration_str=expiration_str,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                months_back=0,
                resolution=resolution,
                time_window_seconds=time_window_seconds
            )
            
            curves_saved = self.collector.save_volatility_curves(
                curves=curves,
                expiration_str=expiration_str
            )
            logger.info(f"✓ Saved {curves_saved} volatility curves for {expiration_str}")
        
        return BacktestingResult(
            expiration_str=expiration_str,
            futures_records=futures_records,
            option_records=option_records,
            curves_saved=curves_saved,
            success=True
        )
    
    def _print_summary(self, results: List[BacktestingResult]):
        """Print final summary of results"""
        logger.info("\n" + "=" * 60)
        logger.info("Final Summary")
        logger.info("=" * 60)
        
        total_futures = sum(r.futures_records for r in results)
        total_options = sum(r.option_records for r in results)
        total_curves = sum(r.curves_saved for r in results)
        successful = sum(1 for r in results if r.success)
        
        logger.info(f"Total expirations processed: {len(results)}")
        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Total futures records saved: {total_futures}")
        logger.info(f"Total option records saved: {total_options}")
        logger.info(f"Total volatility curves saved: {total_curves}")
        
        if successful == len(results):
            logger.info("\n✓ All data saved to MySQL database successfully!")
        else:
            logger.warning(f"\n⚠ {len(results) - successful} expirations failed. Check logs above for details.")


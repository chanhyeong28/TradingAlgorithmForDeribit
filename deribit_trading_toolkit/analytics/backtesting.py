"""
Backtesting Module

Provides tools for backtesting trading strategies using historical data from Deribit.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import json
import mysql.connector

from ..core.client import DeribitClient, DeribitAuth
from ..utils.config import ConfigManager
from ..analytics.volatility import VolatilityAnalyzer
from ..models.market_data import MarketData, VolatilityCurve

logger = logging.getLogger(__name__)


@dataclass
class HistoricalOHLCV:
    """Historical OHLCV data point"""
    timestamp: int  # Milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    cost: float


class HistoricalDataCollector:
    """
    Collects historical price data from Deribit API.
    
    Uses TradingView chart data API to fetch historical OHLCV data
    for futures and options instruments.
    """
    
    def __init__(self, client: Optional[DeribitClient] = None, 
                 db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize historical data collector.
        
        Args:
            client: DeribitClient instance (optional, will create if not provided)
            db_config: Database configuration dict with keys: host, user, password, database
                      (optional, will use defaults from config if not provided)
        """
        self.client = client
        self._should_close_client = False
        
        # Database configuration
        if db_config:
            self.db_config = db_config
        else:
            cfg = ConfigManager().get_config()
            self.db_config = {
                'host': cfg.database.host,
                'user': cfg.database.user,
                'password': cfg.database.password,
                'database': cfg.database.database
            }
        
        self.db_connection: Optional[mysql.connector.MySQLConnection] = None
        self.db_cursor: Optional[mysql.connector.cursor.MySQLCursor] = None
    
    def _connect_db(self):
        """Connect to MySQL database"""
        if self.db_connection is None or not self.db_connection.is_connected():
            try:
                self.db_connection = mysql.connector.connect(
                    host=self.db_config['host'],
                    user=self.db_config['user'],
                    password=self.db_config['password'],
                    database=self.db_config['database']
                )
                self.db_cursor = self.db_connection.cursor()
                logger.info("Database connected successfully")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                raise
    
    def _disconnect_db(self):
        """Disconnect from MySQL database"""
        if self.db_cursor:
            self.db_cursor.close()
            self.db_cursor = None
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            self.db_connection = None
        logger.debug("Database disconnected")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.client:
            cfg = ConfigManager().get_config()
            auth = DeribitAuth(
                client_id=cfg.deribit.effective_client_id,
                private_key_path=cfg.deribit.effective_private_key_path,
                private_key=cfg.deribit.effective_private_key
            )
            self.client = DeribitClient(cfg.deribit, auth)
            await self.client.connect()
            self._should_close_client = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._should_close_client and self.client:
            await self.client.disconnect()
        self._disconnect_db()
    
    async def collect_instrument_data(
        self,
        instrument_name: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        months_back: int = 1,
        resolution: int = 10
    ) -> List[HistoricalOHLCV]:
        """
        Collect historical data for a single instrument.
        
        Args:
            instrument_name: Instrument name (e.g., "BTC-25SEP26-150000-C" or "BTC-25SEP26")
            start_timestamp: Start timestamp in milliseconds (optional)
            end_timestamp: End timestamp in milliseconds (optional)
            months_back: Number of months to look back (default: 1)
            resolution: Resolution in seconds (default: 10)
            
        Returns:
            List of HistoricalOHLCV data points
        """
        # Calculate timestamps if not provided
        if end_timestamp is None:
            end_timestamp = int(datetime.now().timestamp() * 1000)
        
        if start_timestamp is None:
            # Default to last N months
            start_date = datetime.now() - timedelta(days=30 * months_back)
            start_timestamp = int(start_date.timestamp() * 1000)
        
        logger.info(
            f"Collecting historical data for {instrument_name} "
            f"from {datetime.fromtimestamp(start_timestamp/1000)} "
            f"to {datetime.fromtimestamp(end_timestamp/1000)} "
            f"(resolution: {resolution}s)"
        )
        
        try:
            result = await self.client.get_tradingview_chart_data(
                instrument_name=instrument_name,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                resolution=resolution
            )
            
            if result.get('status') != 'ok':
                logger.warning(f"Non-ok status for {instrument_name}: {result.get('status')}")
                return []
            
            # Parse the response
            ticks = result.get('ticks', [])
            opens = result.get('open', [])
            highs = result.get('high', [])
            lows = result.get('low', [])
            closes = result.get('close', [])
            volumes = result.get('volume', [])
            costs = result.get('cost', [])
            
            # Build data points
            data_points = []
            for i, tick in enumerate(ticks):
                if i < len(closes):
                    data_points.append(HistoricalOHLCV(
                        timestamp=tick,
                        open=opens[i] if i < len(opens) else closes[i],
                        high=highs[i] if i < len(highs) else closes[i],
                        low=lows[i] if i < len(lows) else closes[i],
                        close=closes[i],
                        volume=volumes[i] if i < len(volumes) else 0.0,
                        cost=costs[i] if i < len(costs) else 0.0
                    ))
            
            logger.info(f"Collected {len(data_points)} data points for {instrument_name}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error collecting data for {instrument_name}: {e}")
            return []
    
    async def collect_multiple_instruments(
        self,
        instrument_names: List[str],
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        months_back: int = 1,
        resolution: int = 10
    ) -> Dict[str, List[HistoricalOHLCV]]:
        """
        Collect historical data for multiple instruments.
        
        Args:
            instrument_names: List of instrument names
            start_timestamp: Start timestamp in milliseconds (optional)
            end_timestamp: End timestamp in milliseconds (optional)
            months_back: Number of months to look back (default: 1)
            resolution: Resolution in seconds (default: 10)
            
        Returns:
            Dictionary mapping instrument names to their historical data
        """
        results = {}
        
        for instrument_name in instrument_names:
            data = await self.collect_instrument_data(
                instrument_name=instrument_name,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                months_back=months_back,
                resolution=resolution
            )
            results[instrument_name] = data
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        return results
    
    def to_dataframe(self, data: List[HistoricalOHLCV]) -> pd.DataFrame:
        """
        Convert historical data to pandas DataFrame.
        
        Args:
            data: List of HistoricalOHLCV data points
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, cost
        """
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'datetime': datetime.fromtimestamp(d.timestamp / 1000),
            'open': d.open,
            'high': d.high,
            'low': d.low,
            'close': d.close,
            'volume': d.volume,
            'cost': d.cost
        } for d in data])
        
        df.set_index('datetime', inplace=True)
        return df
    
    def _extract_expiration(self, instrument_name: str) -> Optional[int]:
        """Extract expiration timestamp from instrument name or expiration string"""
        # Pattern: BTC-DDMMMYY-STRIKE-C/P or BTC-DDMMMYY or DDMMMYY
        # Try to extract expiration string
        exp_str = None
        
        # Remove BTC- prefix if present
        if instrument_name.startswith('BTC-'):
            # Extract expiration part (before strike or end)
            parts = instrument_name.upper().split('-')
            if len(parts) >= 2:
                # Format: BTC-DDMMMYY-STRIKE-C/P or BTC-DDMMMYY
                exp_str = parts[1]
        else:
            # Direct expiration string like "25SEP26" or "3NOV25"
            exp_str = instrument_name.upper()
        
        if not exp_str:
            return None
        
        try:
            # Parse date: DDMMMYY or DMMMYY (single or double digit day)
            # Use regex to properly extract day, month, year
            match = re.match(r'(\d{1,2})([A-Z]{3})(\d{2})', exp_str)
            if not match:
                logger.debug(f"Could not match expiration pattern in: {exp_str}")
                return None
            
            day = int(match.group(1))
            month_str = match.group(2)
            year_str = match.group(3)
            
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            month = month_map.get(month_str)
            if not month:
                logger.debug(f"Invalid month: {month_str}")
                return None
            
            year = 2000 + int(year_str)
            
            exp_date = datetime(year, month, day)
            return int(exp_date.timestamp())
            
        except Exception as e:
            logger.debug(f"Error parsing expiration from {instrument_name}: {e}")
            return None
    
    def _extract_option_type(self, instrument_name: str) -> str:
        """Extract option type (C or P) from instrument name"""
        if '-C' in instrument_name.upper() and not instrument_name.upper().endswith('-P'):
            return 'C'
        elif '-P' in instrument_name.upper() or instrument_name.upper().endswith('-P'):
            return 'P'
        return 'UNKNOWN'
    
    async def build_volatility_curves_over_period(
        self,
        expiration_str: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        months_back: int = 1,
        resolution: int = 10,
        time_window_seconds: int = 3600  # Build curve every hour
    ) -> List[Tuple[int, Optional[VolatilityCurve]]]:
        """
        Build volatility curves over a period for a specific expiration.
        
        Uses cubic spline interpolation (newton-labs method) to build volatility curves
        from historical option price data.
        
        Args:
            expiration_str: Expiration string (e.g., "25SEP26")
            start_timestamp: Start timestamp in milliseconds (optional)
            end_timestamp: End timestamp in milliseconds (optional)
            months_back: Number of months to look back (default: 1)
            resolution: Resolution in seconds for data collection (default: 10)
            time_window_seconds: Time window for building each curve (default: 3600 = 1 hour)
            
        Returns:
            List of tuples: (timestamp_ms, VolatilityCurve or None)
            Each tuple represents a volatility curve at a specific point in time
        """
        # Calculate timestamps if not provided
        if end_timestamp is None:
            end_timestamp = int(datetime.now().timestamp() * 1000)
        
        if start_timestamp is None:
            start_date = datetime.now() - timedelta(days=30 * months_back)
            start_timestamp = int(start_date.timestamp() * 1000)
        
        # Parse expiration
        expiration_epoch = self._extract_expiration(f"BTC-{expiration_str}")
        if not expiration_epoch:
            logger.error(f"Invalid expiration format: {expiration_str}")
            return []
        
        logger.info(
            f"Building volatility curves for expiration {expiration_str} "
            f"from {datetime.fromtimestamp(start_timestamp/1000)} "
            f"to {datetime.fromtimestamp(end_timestamp/1000)}"
        )
        
        # Get all available BTC options for this expiration
        try:
            instruments = await self.client.get_instruments(currency="BTC", kind="option")
            
            # Filter options for this expiration - get ALL strikes
            target_options = []
            for instr in instruments:
                instr_name = instr.get('instrument_name', '')
                if expiration_str.upper() in instr_name.upper():
                    target_options.append(instr_name)
            
            logger.info(f"Found {len(target_options)} options for expiration {expiration_str}")
            
            if len(target_options) < 2:
                logger.warning(f"Insufficient options for expiration {expiration_str}")
                return []
            
            # Log strike distribution
            strikes = set()
            for opt_name in target_options:
                # Extract strike from option name (format: BTC-DDMMMYY-STRIKE-C/P)
                parts = opt_name.split('-')
                if len(parts) >= 3:
                    try:
                        strike = float(parts[-2])  # Strike is second to last before C/P
                        strikes.add(strike)
                    except:
                        pass
            
            logger.info(f"Found {len(strikes)} unique strikes: {sorted(strikes)[:10]}..." if len(strikes) > 10 else f"Found {len(strikes)} unique strikes: {sorted(strikes)}")
            
            # Collect historical data for ALL options
            logger.info("Collecting historical data for all options...")
            all_option_data: Dict[str, List[HistoricalOHLCV]] = {}
            
            for idx, option_name in enumerate(target_options, 1):
                logger.info(f"[{idx}/{len(target_options)}] Collecting data for {option_name}...")
                try:
                    data = await self.collect_instrument_data(
                        instrument_name=option_name,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        months_back=0,  # Don't double-count months
                        resolution=resolution
                    )
                    if data:
                        all_option_data[option_name] = data
                        logger.info(f"  ✓ Collected {len(data)} data points for {option_name}")
                    else:
                        logger.warning(f"  ✗ No data collected for {option_name}")
                except Exception as e:
                    logger.warning(f"  ✗ Error collecting {option_name}: {e}")
                
                # Rate limiting - small delay between requests
                await asyncio.sleep(0.1)
                
                # Progress update every 10 options
                if idx % 10 == 0:
                    logger.info(f"  Progress: {idx}/{len(target_options)} ({idx*100//len(target_options)}%) - {len(all_option_data)} successful")
            
            logger.info(f"✓ Successfully collected data for {len(all_option_data)}/{len(target_options)} options")
            
            if not all_option_data:
                logger.warning("No historical data collected")
                return []
            
            # Get underlying price (use futures or perpetual)
            underlying_name = f"BTC-{expiration_str}"
            underlying_data = await self.collect_instrument_data(
                instrument_name=underlying_name,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                resolution=resolution
            )
            
            # If no future, try perpetual
            if not underlying_data:
                underlying_data = await self.collect_instrument_data(
                    instrument_name="BTC-PERPETUAL",
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    resolution=resolution
                )
            
            # Build curves over time windows using SVI
            analyzer = VolatilityAnalyzer(use_svi=True)
            curves = []
            
            # Create time windows
            current_time = start_timestamp
            time_window_ms = time_window_seconds * 1000
            
            # Calculate total windows for progress tracking
            total_time_ms = end_timestamp - start_timestamp
            total_windows = int(total_time_ms / time_window_ms) + 1
            
            logger.info(f"Building curves for {total_windows} time windows...")
            
            window_idx = 0
            while current_time < end_timestamp:
                window_idx += 1
                window_end = min(current_time + time_window_ms, end_timestamp)
                window_dt = datetime.fromtimestamp(current_time / 1000)
                
                # Progress update
                if window_idx % 10 == 0 or window_idx == 1:
                    logger.info(f"[Window {window_idx}/{total_windows}] Processing {window_dt.strftime('%Y-%m-%d %H:%M:%S')}...")
                
                # Collect MarketData for this time window
                options_market_data: List[MarketData] = []
                underlying_price = None
                
                # Get underlying price at this time
                for point in underlying_data:
                    if current_time <= point.timestamp <= window_end:
                        underlying_price = point.close
                        break
                
                if underlying_price is None:
                    # Try to get closest underlying price
                    for point in underlying_data:
                        if point.timestamp >= current_time:
                            underlying_price = point.close
                            break
                
                if underlying_price is None:
                    logger.debug(f"  ✗ No underlying price at {window_dt}, skipping")
                    curves.append((current_time, None))
                    current_time = window_end
                    continue
                
                # Collect option data for this window with retry logic
                successful_ivs = 0
                failed_ivs = 0
                
                for option_name, option_history in all_option_data.items():
                    # Find the closest data point to window start
                    closest_point = None
                    min_diff = float('inf')
                    
                    for point in option_history:
                        if current_time <= point.timestamp <= window_end:
                            diff = abs(point.timestamp - current_time)
                            if diff < min_diff:
                                min_diff = diff
                                closest_point = point
                    
                    if closest_point:
                        # Try to get IV from ticker with retry logic for 502 errors
                        max_retries = 3
                        retry_delay = 1.0
                        
                        for retry in range(max_retries):
                            try:
                                ticker = await self.client.get_ticker(option_name)
                                if ticker.mark_iv:
                                    md = MarketData(
                                        instrument_name=option_name,
                                        timestamp=closest_point.timestamp,
                                        mark_price=closest_point.close,
                                        mark_iv=ticker.mark_iv,
                                    )
                                    options_market_data.append(md)
                                    successful_ivs += 1
                                    break
                            except Exception as e:
                                error_msg = str(e)
                                if '502' in error_msg or 'Bad Gateway' in error_msg:
                                    if retry < max_retries - 1:
                                        logger.warning(f"  ⚠ [Window {window_idx}] Retry {retry+1}/{max_retries} for {option_name} (502 Bad Gateway)")
                                        await asyncio.sleep(retry_delay)
                                        retry_delay *= 2  # Exponential backoff
                                        continue
                                    else:
                                        failed_ivs += 1
                                        logger.warning(f"  ✗ [Window {window_idx}] Failed to get IV for {option_name} after {max_retries} retries (502)")
                                elif '429' in error_msg or 'Too Many Requests' in error_msg:
                                    # Rate limit - wait longer
                                    logger.warning(f"  ⚠ [Window {window_idx}] Rate limited for {option_name}, waiting {retry_delay}s...")
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                    if retry < max_retries - 1:
                                        continue
                                    else:
                                        failed_ivs += 1
                                else:
                                    failed_ivs += 1
                                    logger.debug(f"  ✗ Could not get IV for {option_name}: {e}")
                                    break
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.05)
                
                # Log progress for this window
                if len(options_market_data) >= 2:
                    logger.info(f"  ✓ Window {window_idx}/{total_windows} ({window_dt.strftime('%H:%M')}): {len(options_market_data)} options with IV ({successful_ivs} success, {failed_ivs} failed)")
                else:
                    logger.warning(f"  ✗ Window {window_idx}/{total_windows} ({window_dt.strftime('%H:%M')}): Insufficient IV data ({len(options_market_data)} < 2, {successful_ivs} success, {failed_ivs} failed)")
                
                # Build volatility curve for this window
                if len(options_market_data) >= 2:
                    curve = analyzer.build_volatility_curve(
                        options_data=options_market_data,
                        underlying_price=underlying_price,
                        expiration=expiration_epoch
                    )
                    if curve:
                        logger.info(f"  ✓ Built curve with {len(curve.points)} points (ATM IV: {curve.atm_iv:.4f})")
                        curves.append((current_time, curve))
                    else:
                        logger.warning(f"  ✗ Failed to build curve (insufficient valid data)")
                        curves.append((current_time, None))
                else:
                    curves.append((current_time, None))
                
                current_time = window_end
                
                # Periodic summary
                if window_idx % 50 == 0:
                    valid_count = len([c for _, c in curves if c is not None])
                    logger.info(f"  Progress: {window_idx}/{total_windows} windows processed, {valid_count} valid curves built")
            
            valid_count = len([c for _, c in curves if c is not None])
            logger.info(f"✓ Built {valid_count}/{len(curves)} volatility curves ({valid_count*100//len(curves) if curves else 0}% success rate)")
            return curves
            
        except Exception as e:
            logger.error(f"Error building volatility curves: {e}", exc_info=True)
            return []
    
    def plot_volatility_curves_over_time(
        self,
        curves: List[Tuple[int, Optional[VolatilityCurve]]],
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot volatility curves over time.
        
        Args:
            curves: List of (timestamp_ms, VolatilityCurve) tuples
            title: Optional title for the plot
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Filter out None curves
        valid_curves = [(ts, c) for ts, c in curves if c is not None]
        
        if not valid_curves:
            logger.warning("No valid curves to plot")
            return fig
        
        # Extract data
        timestamps = [datetime.fromtimestamp(ts / 1000) for ts, _ in valid_curves]
        atm_ivs = [c.atm_iv for _, c in valid_curves]
        slopes = [c.slope for _, c in valid_curves]
        num_points = [len(c.points) for _, c in valid_curves]
        
        # Log curve statistics
        if num_points:
            logger.info(f"Curve statistics: min_strikes={min(num_points)}, max_strikes={max(num_points)}, avg_strikes={np.mean(num_points):.1f}")
        
        # Plot 1: ATM IV over time
        ax1 = axes[0]
        ax1.plot(timestamps, atm_ivs, marker='o', linestyle='-', linewidth=2, markersize=4)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('ATM Implied Volatility', fontsize=12)
        ax1.set_title('ATM Implied Volatility Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(timestamps) // 10)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add statistics
        mean_iv = np.mean(atm_ivs)
        std_iv = np.std(atm_ivs)
        ax1.axhline(y=mean_iv, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_iv:.4f}')
        ax1.fill_between(timestamps, mean_iv - std_iv, mean_iv + std_iv, alpha=0.2, color='gray', label=f'±1σ: {std_iv:.4f}')
        ax1.legend()
        
        # Plot 2: ATM Slope over time
        ax2 = axes[1]
        ax2.plot(timestamps, slopes, marker='s', linestyle='-', linewidth=2, markersize=4, color='green')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('ATM Slope (Skew)', fontsize=12)
        ax2.set_title('ATM Slope (Volatility Skew) Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(timestamps) // 10)))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add statistics
        mean_slope = np.mean(slopes)
        ax2.axhline(y=mean_slope, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_slope:.4f}')
        ax2.legend()
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        else:
            fig.suptitle('Volatility Curves Analysis Over Time', fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_volatility_surface_evolution(
        self,
        curves: List[Tuple[int, Optional[VolatilityCurve]]],
        max_curves: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot volatility surface evolution by showing multiple curves at different times.
        
        Args:
            curves: List of (timestamp_ms, VolatilityCurve) tuples
            max_curves: Maximum number of curves to plot (default: 10)
            title: Optional title for the plot
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        valid_curves = [(ts, c) for ts, c in curves if c is not None]
        
        if not valid_curves:
            logger.warning("No valid curves to plot")
            return plt.figure()
        
        # Select curves to plot (evenly spaced)
        if len(valid_curves) > max_curves:
            indices = np.linspace(0, len(valid_curves) - 1, max_curves, dtype=int)
            selected_curves = [valid_curves[i] for i in indices]
        else:
            selected_curves = valid_curves
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for different curves
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_curves)))
        
        for idx, (timestamp, curve) in enumerate(selected_curves):
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # Extract moneyness and IV from curve points
            if not curve.points:
                continue
            
            sorted_points = sorted(curve.points, key=lambda p: p.moneyness)
            moneyness = [p.moneyness for p in sorted_points]
            iv = [p.iv for p in sorted_points]
            
            # Separate calls and puts for better visualization
            call_points = [p for p in sorted_points if p.option_type and p.option_type.value == 'call']
            put_points = [p for p in sorted_points if p.option_type and p.option_type.value == 'put']
            
            # Plot full curve with all strikes
            if call_points and put_points:
                # Plot calls and puts separately for clarity
                call_moneyness = [p.moneyness for p in call_points]
                call_iv = [p.iv for p in call_points]
                put_moneyness = [p.moneyness for p in put_points]
                put_iv = [p.iv for p in put_points]
                
                ax.plot(call_moneyness, call_iv, marker='o', linestyle='-', linewidth=2, 
                       markersize=4, color=colors[idx], alpha=0.7,
                       label=f'{dt.strftime("%Y-%m-%d %H:%M")} (IV={curve.atm_iv:.3f}, {len(curve.points)} strikes)')
                ax.plot(put_moneyness, put_iv, marker='s', linestyle='--', linewidth=2, 
                       markersize=3, color=colors[idx], alpha=0.5)
            else:
                # Plot combined curve
                ax.plot(moneyness, iv, marker='o', linestyle='-', linewidth=2, 
                       markersize=4, color=colors[idx], 
                       label=f'{dt.strftime("%Y-%m-%d %H:%M")} (IV={curve.atm_iv:.3f}, {len(curve.points)} strikes)')
        
        ax.set_xlabel('Log-Moneyness (k)', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        ax.set_title('Volatility Surface Evolution Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='ATM')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def save_futures_prices(
        self,
        instrument_name: str,
        data_points: List[HistoricalOHLCV],
        resolution_seconds: int,
        expiration_timestamp: Optional[int] = None
    ) -> int:
        """Save historical futures prices to MySQL database."""
        if not data_points:
            logger.warning(f"No data points to save for {instrument_name}")
            return 0
        
        self._connect_db()
        
        try:
            sql = """
                INSERT INTO btc_historical_futures_prices
                (timestamp, instrument_name, expiration_timestamp, open_price, high_price, 
                 low_price, close_price, volume, cost, resolution_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    open_price=VALUES(open_price),
                    high_price=VALUES(high_price),
                    low_price=VALUES(low_price),
                    close_price=VALUES(close_price),
                    volume=VALUES(volume),
                    cost=VALUES(cost)
            """
            
            values = [(p.timestamp, instrument_name, expiration_timestamp, p.open, p.high, 
                      p.low, p.close, p.volume, p.cost, resolution_seconds) for p in data_points]
            
            self.db_cursor.executemany(sql, values)
            self.db_connection.commit()
            
            rows_inserted = self.db_cursor.rowcount
            logger.info(f"✓ Saved {rows_inserted} futures price records for {instrument_name}")
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Error saving futures prices for {instrument_name}: {e}")
            self.db_connection.rollback()
            raise
    
    def save_option_prices(
        self,
        instrument_name: str,
        data_points: List[HistoricalOHLCV],
        resolution_seconds: int,
        expiration_timestamp: int,
        option_type: str,
        strike_price: float
    ) -> int:
        """Save historical option prices to MySQL database."""
        if not data_points:
            logger.warning(f"No data points to save for {instrument_name}")
            return 0
        
        self._connect_db()
        
        try:
            sql = """
                INSERT INTO btc_historical_option_prices
                (timestamp, instrument_name, expiration_timestamp, option_type, strike_price,
                 open_price, high_price, low_price, close_price, volume, cost, resolution_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    open_price=VALUES(open_price),
                    high_price=VALUES(high_price),
                    low_price=VALUES(low_price),
                    close_price=VALUES(close_price),
                    volume=VALUES(volume),
                    cost=VALUES(cost)
            """
            
            values = [(p.timestamp, instrument_name, expiration_timestamp, option_type.lower(), 
                      strike_price, p.open, p.high, p.low, p.close, p.volume, p.cost, 
                      resolution_seconds) for p in data_points]
            
            self.db_cursor.executemany(sql, values)
            self.db_connection.commit()
            
            rows_inserted = self.db_cursor.rowcount
            logger.info(f"✓ Saved {rows_inserted} option price records for {instrument_name}")
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Error saving option prices for {instrument_name}: {e}")
            self.db_connection.rollback()
            raise
    
    def save_volatility_curves(
        self,
        curves: List[Tuple[int, Optional[VolatilityCurve]]],
        expiration_str: str
    ) -> int:
        """Save historical volatility curves to MySQL database."""
        if not curves:
            logger.warning(f"No volatility curves to save for {expiration_str}")
            return 0
        
        self._connect_db()
        
        try:
            sql = """
                INSERT INTO btc_historical_volatility_curves
                (timestamp, expiration_timestamp, expiration_str, underlying_price,
                 atm_iv, atm_slope, curvature, num_points, curve_data,
                 svi_a, svi_b, svi_rho, svi_m, svi_sigma)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    underlying_price=VALUES(underlying_price),
                    atm_iv=VALUES(atm_iv),
                    atm_slope=VALUES(atm_slope),
                    curvature=VALUES(curvature),
                    num_points=VALUES(num_points),
                    curve_data=VALUES(curve_data),
                    svi_a=VALUES(svi_a),
                    svi_b=VALUES(svi_b),
                    svi_rho=VALUES(svi_rho),
                    svi_m=VALUES(svi_m),
                    svi_sigma=VALUES(svi_sigma)
            """
            
            values = []
            saved_count = 0
            
            for timestamp, curve in curves:
                if curve is None:
                    continue
                
                curve_data = [{
                    'strike': p.strike,
                    'log_moneyness': p.moneyness,
                    'iv': p.iv,
                    'option_type': p.option_type.value if p.option_type else None
                } for p in curve.points]
                
                underlying_price = curve.atm_strike if curve.atm_strike else 0.0
                
                # Get SVI parameters if available
                svi_params = getattr(curve, 'svi_params', None)
                if svi_params:
                    svi_a = svi_params.a
                    svi_b = svi_params.b
                    svi_rho = svi_params.rho
                    svi_m = svi_params.m
                    svi_sigma = svi_params.sigma
                else:
                    svi_a = svi_b = svi_rho = svi_m = svi_sigma = None
                
                values.append((
                    timestamp, curve.expiration, expiration_str, underlying_price,
                    curve.atm_iv, curve.slope, curve.curvature,
                    len(curve.points), json.dumps(curve_data),
                    svi_a, svi_b, svi_rho, svi_m, svi_sigma
                ))
                saved_count += 1
            
            if values:
                self.db_cursor.executemany(sql, values)
                self.db_connection.commit()
                logger.info(f"✓ Saved {saved_count} volatility curves for {expiration_str}")
            
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving volatility curves for {expiration_str}: {e}")
            self.db_connection.rollback()
            raise
    
    def _extract_strike_from_name(self, instrument_name: str) -> Optional[float]:
        """Extract strike price from instrument name"""
        parts = instrument_name.split('-')
        if len(parts) >= 3:
            try:
                return float(parts[-2])
            except:
                pass
        return None


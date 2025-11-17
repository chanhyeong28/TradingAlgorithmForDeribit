"""
Backtesting Module for Options Trading

Provides OOP interfaces for:
- SimpleOptionBacktester: Buy and hold specific options with optional delta hedging
"""

import asyncio
import logging
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import mysql.connector
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..analytics.min_var_delta import MinimumVarianceDeltaCalculator, MinimumVarianceDeltaResult
from ..analytics.ssr import SSRCalculator
from ..analytics.pnl_decomposition import PnLDecomposer, PnLDecompositionResult
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class BacktestPosition:
    """Represents an option position in backtesting"""
    instrument_name: str
    expiration_timestamp: int
    strike: float
    option_type: str  # 'call' or 'put'
    quantity: float  # Positive for long, negative for short
    entry_price: float
    entry_timestamp: int
    current_price: Optional[float] = None
    current_delta: Optional[float] = None
    min_var_delta: Optional[float] = None


@dataclass
class HedgePosition:
    """Represents a delta hedge position"""
    timestamp: int
    total_delta: float
    hedge_quantity: float  # Futures quantity to hedge
    hedge_cost: float


@dataclass
class FuturesPosition:
    """Represents a futures position for delta hedging"""
    instrument_name: str  # e.g., "BTC-26DEC25" or "BTC-PERPETUAL"
    expiration_str: str  # e.g., "26DEC25"
    quantity: float  # Positive for long, negative for short
    entry_price: float
    entry_timestamp: int
    current_price: Optional[float] = None


@dataclass
class BacktestTrade:
    """Represents a trade execution in backtesting"""
    timestamp: int
    instrument_name: str
    quantity: float
    price: float
    trade_type: str  # 'open', 'close', 'hedge'
    pnl: float = 0.0


@dataclass
class DailyPnLDecomposition:
    """Daily PnL decomposition result"""
    timestamp: int
    total_pnl: float
    funding: float
    ir_theta: float
    delta: float
    gamma: float
    vol_block: float
    vanna_block: float
    positions_pnl: Dict[str, float]  # PnL per position
    decomposed_pnl_sum: float = 0.0  # Sum of all decomposition buckets
    pnl_error: float = 0.0  # Difference between actual PnL and decomposed sum


@dataclass
class BacktestResult:
    """Result of backtesting"""
    total_pnl: float
    total_trades: int
    total_hedges: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: List[BacktestTrade]
    daily_pnl: List[Tuple[int, float]]
    daily_pnl_decomposition: List[DailyPnLDecomposition]
    positions_history: List[Dict[str, Any]]

@dataclass
class OptionSpec:
    """
    Specification for an option to trade.
    
    Attributes:
        expiration_str: Expiration date string (e.g., "26DEC25")
        strike: Strike price
        option_type: 'call' or 'put'
        quantity: Position size. Positive for long, negative for short.
                  Example: quantity=1.0 means long 1 option, quantity=-1.0 means short 1 option
    """
    expiration_str: str
    strike: float
    option_type: str  # 'call' or 'put'
    quantity: float  # Positive for long, negative for short


# ============================================================================
# Plotting Class
# ============================================================================

class BacktestPlotter:
    """
    OOP plotting class for backtest results.
    
    Provides methods to visualize:
    - Cumulative PnL over time
    - Daily actual PnL vs PnL decomposition with Greek proportions
    """
    
    def __init__(self, result: BacktestResult):
        """Initialize plotter with backtest result"""
        self.result = result
    
    def plot_cumulative_pnl(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot cumulative PnL over time.
        
        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if not self.result.daily_pnl:
            ax.text(0.5, 0.5, 'No PnL data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cumulative PnL', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Filter out None timestamps
        valid_pnl = [(ts, pnl) for ts, pnl in self.result.daily_pnl if ts is not None]
        
        if not valid_pnl:
            ax.text(0.5, 0.5, 'No valid PnL data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cumulative PnL', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        timestamps = [datetime.fromtimestamp(ts / 1000) for ts, _ in valid_pnl]
        pnl_values = [pnl for _, pnl in valid_pnl]
        
        # daily_pnl contains daily changes, so cumsum gives cumulative PnL
        cumulative_pnl = np.cumsum(pnl_values)
        
        ax.plot(timestamps, cumulative_pnl, linewidth=2.5, label='Cumulative PnL', color='#2c3e50', marker='o', markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(timestamps, 0, cumulative_pnl, alpha=0.3, color='#3498db', where=(cumulative_pnl >= 0))
        ax.fill_between(timestamps, 0, cumulative_pnl, alpha=0.3, color='#e74c3c', where=(cumulative_pnl < 0))
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative PnL ($)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative PnL Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps) // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add statistics text box
        total_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0.0
        max_pnl = np.max(cumulative_pnl) if len(cumulative_pnl) > 0 else 0.0
        min_pnl = np.min(cumulative_pnl) if len(cumulative_pnl) > 0 else 0.0
        
        stats_text = f'Total PnL: ${total_pnl:,.2f}\nMax PnL: ${max_pnl:,.2f}\nMin PnL: ${min_pnl:,.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cumulative PnL plot saved to {save_path}")
        
        return fig
    
    def plot_daily_pnl_decomposition(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Plot daily actual PnL vs PnL decomposition with proportion of each Greek.
        
        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        if not self.result.daily_pnl_decomposition:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, 'No PnL decomposition data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Daily PnL Decomposition', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Filter out None timestamps
        valid_decomp = [d for d in self.result.daily_pnl_decomposition if d.timestamp is not None]
        
        if not valid_decomp:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.text(0.5, 0.5, 'No valid PnL decomposition data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Daily PnL Decomposition', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        timestamps = [datetime.fromtimestamp(d.timestamp / 1000) for d in valid_decomp]
        timestamps_ms = [d.timestamp for d in valid_decomp]
        
        # Real Option PnL: calculated from actual option price changes (option only, no futures)
        # This is the true PnL from differentiating option prices
        # Use sum of positions_pnl which only includes option PnL
        real_option_pnl = [sum(d.positions_pnl.values()) for d in valid_decomp]
        
        # Decomposed PnL: Taylor expansion approximation
        # This is the approximation using Greeks
        decomposed_pnl = [d.decomposed_pnl_sum for d in valid_decomp]
        
        # Extract Greek components
        funding = [d.funding for d in valid_decomp]
        ir_theta = [d.ir_theta for d in valid_decomp]
        delta = [d.delta for d in valid_decomp]
        gamma = [d.gamma for d in valid_decomp]
        vol_block = [d.vol_block for d in valid_decomp]
        vanna_block = [d.vanna_block for d in valid_decomp]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Real Option PnL vs Decomposed PnL (Taylor Expansion Comparison)
        # This compares the actual option price movement with the Taylor expansion approximation
        axes[0].plot(timestamps, real_option_pnl, label='Real Option PnL (Price Change)', linewidth=2.5, 
                    color='#2c3e50', marker='o', markersize=5, linestyle='-')
        axes[0].plot(timestamps, decomposed_pnl, label='Decomposed PnL (Taylor Expansion)', linewidth=2.5,
                    color='#e74c3c', marker='s', markersize=5, linestyle='--', alpha=0.8)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].fill_between(timestamps, 0, real_option_pnl, alpha=0.2, color='#3498db', where=(np.array(real_option_pnl) >= 0))
        axes[0].fill_between(timestamps, 0, real_option_pnl, alpha=0.2, color='#e74c3c', where=(np.array(real_option_pnl) < 0))
        
        axes[0].set_ylabel('Daily PnL ($)', fontsize=12, fontweight='bold')
        axes[0].set_title('Real Option PnL vs Decomposed PnL (Taylor Expansion Approximation)', fontsize=14, fontweight='bold', pad=15)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Calculate and display statistics
        # Compare real option PnL with Taylor expansion approximation
        pnl_errors = [r - d for r, d in zip(real_option_pnl, decomposed_pnl)]
        mean_error = np.mean(pnl_errors)
        std_error = np.std(pnl_errors)
        max_error = np.max(np.abs(pnl_errors))
        mean_abs_error = np.mean(np.abs(pnl_errors))
        
        # Calculate R-squared to measure approximation quality
        real_array = np.array(real_option_pnl)
        decomposed_array = np.array(decomposed_pnl)
        ss_res = np.sum((real_array - decomposed_array) ** 2)
        ss_tot = np.sum((real_array - np.mean(real_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        
        stats_text = (f'Taylor Expansion Fit:\n'
                     f'Mean Error: ${mean_error:.2f}\n'
                     f'Std Error: ${std_error:.2f}\n'
                     f'Max |Error|: ${max_error:.2f}\n'
                     f'Mean |Error|: ${mean_abs_error:.2f}\n'
                     f'R²: {r_squared:.4f}')
        axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: PnL Decomposition with Greek proportions (option only, no futures)
        # This shows the non-hedged option portfolio
        # Calculate cumulative sums for stacked area (option Greeks only)
        cumulative_funding = np.cumsum(funding)
        cumulative_ir_theta = np.cumsum(ir_theta)
        cumulative_delta = np.cumsum(delta)
        cumulative_gamma = np.cumsum(gamma)
        cumulative_vol_block = np.cumsum(vol_block)
        cumulative_vanna_block = np.cumsum(vanna_block)
        
        # Stacked area chart showing cumulative contribution of each Greek (option only, no futures)
        axes[1].fill_between(timestamps, 0, cumulative_funding, label='Funding', alpha=0.7, color='#1f77b4')
        axes[1].fill_between(timestamps, cumulative_funding,
                            [f + it for f, it in zip(cumulative_funding, cumulative_ir_theta)],
                            label='IR Theta', alpha=0.7, color='#ff7f0e')
        axes[1].fill_between(timestamps,
                            [f + it for f, it in zip(cumulative_funding, cumulative_ir_theta)],
                            [f + it + d for f, it, d in zip(cumulative_funding, cumulative_ir_theta, cumulative_delta)],
                            label='Delta', alpha=0.7, color='#2ca02c')
        axes[1].fill_between(timestamps,
                            [f + it + d for f, it, d in zip(cumulative_funding, cumulative_ir_theta, cumulative_delta)],
                            [f + it + d + g for f, it, d, g in zip(cumulative_funding, cumulative_ir_theta, cumulative_delta, cumulative_gamma)],
                            label='Gamma', alpha=0.7, color='#d62728')
        axes[1].fill_between(timestamps,
                            [f + it + d + g for f, it, d, g in zip(cumulative_funding, cumulative_ir_theta, cumulative_delta, cumulative_gamma)],
                            [f + it + d + g + v for f, it, d, g, v in zip(cumulative_funding, cumulative_ir_theta, cumulative_delta, cumulative_gamma, cumulative_vol_block)],
                            label='Vol Block', alpha=0.7, color='#9467bd')
        axes[1].fill_between(timestamps,
                            [f + it + d + g + v for f, it, d, g, v in zip(cumulative_funding, cumulative_ir_theta, cumulative_delta, cumulative_gamma, cumulative_vol_block)],
                            [f + it + d + g + v + va for f, it, d, g, v, va in zip(cumulative_funding, cumulative_ir_theta, cumulative_delta, cumulative_gamma, cumulative_vol_block, cumulative_vanna_block)],
                            label='Vanna Block', alpha=0.7, color='#8c564b')
        
        # Overlay cumulative decomposed PnL (Taylor expansion) and real option PnL
        cumulative_decomposed_pnl = np.cumsum(decomposed_pnl)
        cumulative_real_option_pnl = np.cumsum(real_option_pnl)
        
        axes[1].plot(timestamps, cumulative_real_option_pnl, label='Real Option PnL (Price Change)', 
                    linewidth=3, color='#2c3e50', linestyle='-', marker='o', markersize=6, zorder=10)
        axes[1].plot(timestamps, cumulative_decomposed_pnl, label='Decomposed PnL (Taylor Expansion)', 
                    linewidth=2.5, color='#e74c3c', linestyle='--', marker='s', markersize=5, zorder=9, alpha=0.8)
        
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Cumulative PnL ($)', fontsize=12, fontweight='bold')
        axes[1].set_title('PnL Decomposition: Cumulative Contribution of Each Greek (Option Only, Non-Hedged Portfolio)', fontsize=14, fontweight='bold', pad=15)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
        
        # Format x-axis
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps) // 10)))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add proportion statistics (option only, no futures)
        total_funding = sum(funding)
        total_ir_theta = sum(ir_theta)
        total_delta = sum(delta)
        total_gamma = sum(gamma)
        total_vol_block = sum(vol_block)
        total_vanna_block = sum(vanna_block)
        total_decomposed_option = sum(decomposed_pnl)  # Option-only decomposed PnL
        
        if abs(total_decomposed_option) > 1e-6:
            prop_funding = (total_funding / total_decomposed_option) * 100
            prop_ir_theta = (total_ir_theta / total_decomposed_option) * 100
            prop_delta = (total_delta / total_decomposed_option) * 100
            prop_gamma = (total_gamma / total_decomposed_option) * 100
            prop_vol_block = (total_vol_block / total_decomposed_option) * 100
            prop_vanna_block = (total_vanna_block / total_decomposed_option) * 100
            
            prop_text = (f'Option Portfolio Proportions:\n'
                        f'Funding: {prop_funding:.1f}%\n'
                        f'IR Theta: {prop_ir_theta:.1f}%\n'
                        f'Delta: {prop_delta:.1f}%\n'
                        f'Gamma: {prop_gamma:.1f}%\n'
                        f'Vol Block: {prop_vol_block:.1f}%\n'
                        f'Vanna Block: {prop_vanna_block:.1f}%')
            
            axes[1].text(0.98, 0.02, prop_text, transform=axes[1].transAxes,
                        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Daily PnL decomposition plot saved to {save_path}")
        
        return fig


# ============================================================================
# SimpleOptionBacktester
# ============================================================================



class SimpleOptionBacktester:
    """
    Simple backtester for buying and holding specific options.
    
    Features:
    - Buy and hold specific options
    - Optional delta hedging (Black-Scholes or minimum variance)
    - Daily PnL decomposition
    - Performance metrics
    """
    
    def __init__(
        self,
        options: List[OptionSpec],
        use_delta_hedge: bool = False,
        hedge_method: str = "bs",  # "bs" or "min_var"
        risk_free_rate: float = 0.05,
        db_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize simple option backtester"""
        self.options = options
        self.use_delta_hedge = use_delta_hedge
        self.hedge_method = hedge_method
        self.risk_free_rate = risk_free_rate
        
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
        
        # State tracking
        self.positions: List[BacktestPosition] = []
        self.futures_positions: Dict[str, FuturesPosition] = {}  # Key: expiration_str
        self.trades: List[BacktestTrade] = []
        self.hedges: List[HedgePosition] = []
        self.daily_pnl: List[Tuple[int, float]] = []
        self.daily_pnl_decomposition: List[DailyPnLDecomposition] = []
        
        # Calculators
        self.pnl_decomposer = PnLDecomposer(risk_free_rate=risk_free_rate)
        if hedge_method == "min_var":
            self.min_var_delta_calc = MinimumVarianceDeltaCalculator(risk_free_rate=risk_free_rate)
        else:
            self.min_var_delta_calc = None
        
    
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
                self.db_cursor = self.db_connection.cursor(dictionary=True)
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
    
    def load_volatility_curves(
        self,
        start_timestamp: int,
        end_timestamp: int,
        expiration_str: str
    ) -> List[Dict[str, Any]]:
        """Load volatility curves from database for a given expiration"""
        self._connect_db()
        
        try:
            sql = """
                SELECT timestamp, expiration_timestamp, expiration_str, underlying_price,
                       atm_iv, atm_slope, curvature, num_points, curve_data,
                       svi_a, svi_b, svi_rho, svi_m, svi_sigma
                FROM btc_historical_volatility_curves
                WHERE expiration_str = %s
                  AND timestamp >= %s
                  AND timestamp <= %s
                ORDER BY timestamp ASC
            """
            
            self.db_cursor.execute(sql, (expiration_str, start_timestamp, end_timestamp))
            results = self.db_cursor.fetchall()
            
            curves = []
            for row in results:
                curve_data = json.loads(row['curve_data']) if row['curve_data'] else []
                curve_dict = {
                    'timestamp': row['timestamp'],
                    'expiration_timestamp': row['expiration_timestamp'],
                    'underlying_price': row['underlying_price'],
                    'atm_iv': row['atm_iv'],
                    'atm_slope': row['atm_slope'],
                    'curvature': row['curvature'],
                    'num_points': row['num_points'],
                    'curve_data': curve_data
                }
                
                # Add SVI parameters if available
                if row.get('svi_a') is not None and row.get('svi_b') is not None:
                    from ..analytics.svi import SVIParams
                    curve_dict['svi_params'] = SVIParams(
                        a=row['svi_a'],
                        b=row['svi_b'],
                        rho=row['svi_rho'],
                        m=row['svi_m'],
                        sigma=row['svi_sigma']
                    )
                
                curves.append(curve_dict)
            
            logger.info(f"Loaded {len(curves)} volatility curves for {expiration_str}")
            return curves
            
        except Exception as e:
            logger.error(f"Error loading volatility curves: {e}")
            return []
    
    def load_option_price(
        self,
        timestamp: int,
        expiration_str: str,
        strike: float,
        option_type: str,
        convert_to_usd: bool = True
    ) -> Optional[float]:
        """Load option price from database and convert to USD if requested"""
        self._connect_db()
        
        try:
            # Convert expiration_str to expiration_timestamp
            expiration_epoch = self._get_expiration_epoch(expiration_str)
            if not expiration_epoch:
                logger.debug(f"Could not convert expiration_str '{expiration_str}' to timestamp")
                return None
            
            sql = """
                SELECT close_price
                FROM btc_historical_option_prices
                WHERE expiration_timestamp = %s
                  AND strike_price = %s
                  AND option_type = %s
                  AND timestamp >= %s - 3600000
                  AND timestamp <= %s + 3600000
                ORDER BY ABS(timestamp - %s) ASC
                LIMIT 1
            """
            
            self.db_cursor.execute(sql, (
                expiration_epoch, strike, option_type.lower(),
                timestamp, timestamp, timestamp
            ))
            result = self.db_cursor.fetchone()
            
            if not result:
                logger.debug(f"No option price found for {expiration_str} {strike} {option_type} at timestamp {timestamp}")
                return None
            
            price_btc = result['close_price']
            
            if convert_to_usd:
                underlying_price = self.load_futures_price(timestamp, expiration_str)
                if underlying_price:
                    return price_btc * underlying_price
                # Fallback to any futures/perpetual price
                underlying_price = self.load_futures_price(timestamp)
                if underlying_price:
                    return price_btc * underlying_price
                logger.debug(f"Could not find underlying price for USD conversion")
                return None
            
            return price_btc
            
        except Exception as e:
            logger.debug(f"Error loading option price: {e}")
            return None
    
    def find_earliest_option_data(
        self,
        expiration_str: str,
        strike: float,
        option_type: str
    ) -> Optional[int]:
        """Find earliest available timestamp for an option"""
        self._connect_db()
        
        try:
            expiration_epoch = self._get_expiration_epoch(expiration_str)
            if not expiration_epoch:
                return None
            
            sql = """
                SELECT MIN(timestamp) as earliest_timestamp
                FROM btc_historical_option_prices
                WHERE expiration_timestamp = %s
                  AND strike_price = %s
                  AND option_type = %s
            """
            
            self.db_cursor.execute(sql, (expiration_epoch, strike, option_type.lower()))
            result = self.db_cursor.fetchone()
            
            return result['earliest_timestamp'] if result and result['earliest_timestamp'] else None
            
        except Exception as e:
            logger.debug(f"Error finding earliest option data: {e}")
            return None
    
    def load_futures_price(self, timestamp: int, expiration_str: Optional[str] = None) -> Optional[float]:
        """
        Load futures/perpetual price at a specific timestamp.
        
        Args:
            timestamp: Timestamp in milliseconds
            expiration_str: Optional expiration string (e.g., "26DEC25") to load specific futures.
                          If None, loads any available futures/perpetual.
        
        Returns:
            Futures price or None if not found
        """
        self._connect_db()
        
        try:
            if expiration_str:
                # Try to load specific futures contract for this expiration
                # First, get expiration timestamp
                expiration_epoch = self._get_expiration_epoch(expiration_str)
                if expiration_epoch:
                    sql = """
                        SELECT close_price
                        FROM btc_historical_futures_prices
                        WHERE expiration_timestamp = %s
                          AND timestamp >= %s - 3600000
                          AND timestamp <= %s + 3600000
                        ORDER BY ABS(timestamp - %s) ASC
                        LIMIT 1
                    """
                    self.db_cursor.execute(sql, (expiration_epoch, timestamp, timestamp, timestamp))
                    result = self.db_cursor.fetchone()
                    if result:
                        return result['close_price']
            
            # Fallback: load any futures/perpetual (for daily options or if specific futures not found)
            sql = """
                SELECT close_price
                FROM btc_historical_futures_prices
                WHERE timestamp >= %s - 3600000
                  AND timestamp <= %s + 3600000
                ORDER BY ABS(timestamp - %s) ASC
                LIMIT 1
            """
            
            self.db_cursor.execute(sql, (timestamp, timestamp, timestamp))
            result = self.db_cursor.fetchone()
            
            return result['close_price'] if result else None
            
        except Exception as e:
            logger.debug(f"Error loading futures price: {e}")
            return None
    
    def _get_futures_instrument_name(self, expiration_str: str) -> str:
        """
        Get futures instrument name for a given expiration.
        
        Args:
            expiration_str: Expiration string (e.g., "26DEC25")
        
        Returns:
            Futures instrument name (e.g., "BTC-26DEC25") or "BTC-PERPETUAL" if no futures exists
        """
        # Check if futures exists for this expiration
        # Format: BTC-{expiration_str}
        futures_name = f"BTC-{expiration_str}"
        
        # For daily options, always use perpetual
        # We can check if it's a daily option by checking if it's within the next 7 days
        # But for simplicity, we'll try to load the futures price first
        # If it doesn't exist, we'll use perpetual
        
        return futures_name
    
    def calculate_black_scholes_delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> float:
        """Calculate Black-Scholes delta"""
        if T <= 0:
            return 1.0 if option_type == 'call' and S > K else 0.0
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return -norm.cdf(-d1)
    
    def _get_expiration_epoch(self, expiration_str: str) -> int:
        """Get expiration epoch timestamp"""
        from ..analytics.backtesting import HistoricalDataCollector
        collector = HistoricalDataCollector()
        epoch = collector._extract_expiration(f"BTC-{expiration_str}")
        return epoch if epoch else 0
    
    def establish_positions(self, timestamp: int, underlying_price: float):
        """Establish all option positions"""
        if self.positions:
            return  # Already established
        
        logger.info(f"Establishing {len(self.options)} option positions at {datetime.fromtimestamp(timestamp/1000)}")
        
        total_notional = 0.0
        for option_spec in self.options:
            # Try to get option price at exact timestamp first
            option_price_usd = self.load_option_price(
                timestamp,
                option_spec.expiration_str,
                option_spec.strike,
                option_spec.option_type,
                convert_to_usd=True
            )
            
            entry_timestamp = timestamp
            
            # If not found, find earliest available data
            if option_price_usd is None:
                logger.info(f"No price at {datetime.fromtimestamp(timestamp/1000)} for {option_spec.expiration_str} {option_spec.strike} {option_spec.option_type}, searching for earliest available data...")
                earliest_timestamp = self.find_earliest_option_data(
                    option_spec.expiration_str,
                    option_spec.strike,
                    option_spec.option_type
                )
                
                if earliest_timestamp:
                    option_price_usd = self.load_option_price(
                        earliest_timestamp,
                        option_spec.expiration_str,
                        option_spec.strike,
                        option_spec.option_type,
                        convert_to_usd=True
                    )
                    if option_price_usd:
                        logger.info(f"Using earliest available data at {datetime.fromtimestamp(earliest_timestamp/1000)}")
                        entry_timestamp = earliest_timestamp
                    else:
                        logger.warning(f"Could not load price even at earliest timestamp for {option_spec.expiration_str} {option_spec.strike} {option_spec.option_type}")
                        continue
                else:
                    logger.warning(f"Could not find any price data for {option_spec.expiration_str} {option_spec.strike} {option_spec.option_type}")
                    continue
            
            # Ensure USD price
            if option_price_usd < 100 and underlying_price:
                option_price_usd = option_price_usd * underlying_price
            
            expiration_epoch = self._get_expiration_epoch(option_spec.expiration_str)
            
            position = BacktestPosition(
                instrument_name=f"BTC-{option_spec.expiration_str}-{int(option_spec.strike)}-{option_spec.option_type[0].upper()}",
                expiration_timestamp=expiration_epoch,
                strike=option_spec.strike,
                option_type=option_spec.option_type,
                quantity=option_spec.quantity,
                entry_price=option_price_usd,
                entry_timestamp=entry_timestamp
            )
            
            self.positions.append(position)
            total_notional += abs(option_spec.quantity) * underlying_price  # Approximate notional
            
            # Record trade
            self.trades.append(BacktestTrade(
                timestamp=timestamp,
                instrument_name=position.instrument_name,
                quantity=option_spec.quantity,
                price=option_price_usd,
                trade_type='open'
            ))
            
            logger.info(f"  - Opened {option_spec.quantity:+.2f} {position.instrument_name} @ ${option_price_usd:,.2f} (Notional: ${abs(option_spec.quantity) * underlying_price:,.2f})")
        
        logger.info(f"✓ Portfolio established: {len(self.positions)} positions, Total Notional: ${total_notional:,.2f}")
    
    def update_positions_pnl(self, timestamp: int, underlying_price: float):
        """Update PnL for all positions (options and futures)"""
        # Update option positions
        for position in self.positions:
            expiration_str = self._extract_expiration_str(position.instrument_name)
            current_price_usd = self.load_option_price(
                timestamp,
                expiration_str,
                position.strike,
                position.option_type,
                convert_to_usd=True
            )
            
            if current_price_usd:
                if current_price_usd < 100 and underlying_price:
                    current_price_usd = current_price_usd * underlying_price
                position.current_price = current_price_usd
        
        # Update futures positions
        for expiration_str, futures_pos in self.futures_positions.items():
            # Load current futures price
            futures_price = self.load_futures_price(timestamp, expiration_str)
            if futures_price is None:
                # Fallback to perpetual
                futures_price = underlying_price
                futures_pos.instrument_name = "BTC-PERPETUAL"
            
            futures_pos.current_price = futures_price
    
    def _extract_expiration_str(self, instrument_name: str) -> str:
        """Extract expiration string from instrument name"""
        # Format: BTC-26DEC25-100000-C
        parts = instrument_name.split('-')
        if len(parts) >= 2:
            return parts[1]
        return ""
    
    def apply_delta_hedge(
        self,
        timestamp: int,
        underlying_price: float,
        curves: Dict[str, Dict[str, Any]]
    ) -> List[HedgePosition]:
        """
        Apply delta hedge if enabled.
        Hedges each option position with its corresponding futures contract.
        
        Returns:
            List of HedgePosition objects (one per expiration)
        """
        if not self.use_delta_hedge:
            return []
        
        # Group positions by expiration and calculate delta per expiration
        positions_by_exp: Dict[str, List[BacktestPosition]] = {}
        deltas_by_exp: Dict[str, float] = {}
        underlying_prices_by_exp: Dict[str, float] = {}
        
        for position in self.positions:
            if position.current_price is None:
                continue
            
            expiration_str = self._extract_expiration_str(position.instrument_name)
            curve = curves.get(expiration_str)
            
            if not curve:
                continue
            
            # Get underlying price for this expiration
            # Try to load futures price for this expiration first
            futures_price = self.load_futures_price(timestamp, expiration_str)
            if futures_price is None:
                # Fallback to provided underlying_price (perpetual)
                futures_price = underlying_price
            
            # Store underlying price for this expiration
            underlying_prices_by_exp[expiration_str] = futures_price
            
            # Initialize if needed
            if expiration_str not in positions_by_exp:
                positions_by_exp[expiration_str] = []
                deltas_by_exp[expiration_str] = 0.0
            
            positions_by_exp[expiration_str].append(position)
            
            T = max((position.expiration_timestamp - timestamp / 1000) / (365 * 24 * 3600), 0.001)
            
            # Calculate delta
            # First, calculate k_t (log-moneyness) and IV at strike for all methods
            k_t = math.log(position.strike / futures_price) if futures_price > 0 else 0.0
            
            # Calculate IV at strike (k_t) using SVI if available, otherwise linear approximation
            svi_params = curve.get('svi_params')
            if svi_params is not None:
                # Use exact SVI calculation for IV at strike
                from ..analytics.svi import svi_iv_at_moneyness, svi_sqrtw_at_moneyness, svi_d_sqrtw_dk_at_moneyness
                iv_at_strike = float(svi_iv_at_moneyness(np.array([k_t]), T, svi_params)[0])
                sqrtw_at_k = svi_sqrtw_at_moneyness(k_t, T, svi_params)
                d_sqrtw_dk_at_k = svi_d_sqrtw_dk_at_moneyness(k_t, T, svi_params)
            else:
                logger.warning("No SVI parameters found, falling back to linear approximation")
                # Fallback to linear approximation
                atm_iv = curve.get('atm_iv', 0.8)
                # Normalize if stored as percentage (e.g., 85 -> 0.85)
                if atm_iv > 1.0:
                    atm_iv = atm_iv / 100.0
                atm_slope = curve.get('atm_slope', 0.0)
                iv_at_strike = atm_iv + atm_slope * k_t
                sqrtw_at_k = iv_at_strike * math.sqrt(T)
                # d_sqrtw/dk = d(IV*sqrt(T))/dk = d_IV/dk * sqrt(T) = atm_slope * sqrt(T)
                d_sqrtw_dk_at_k = atm_slope * math.sqrt(T)
            
            if self.hedge_method == "min_var" and self.min_var_delta_calc:
                # Use minimum variance delta
                try:
                    min_var_result = self.min_var_delta_calc.calculate_delta_min(
                        F_t=futures_price,
                        K=position.strike,
                        tau_years=T,
                        sqrtw=sqrtw_at_k,
                        d_sqrtw_dk=d_sqrtw_dk_at_k,
                        expiry_ms=position.expiration_timestamp,
                        SSR_tau=1.0,  # Default SSR
                        option_type=position.option_type  # 'call' or 'put'
                    )
                    
                    if min_var_result:
                        delta = min_var_result.delta_min
                    else:
                        logger.warning("No minimum variance delta result, falling back to BS")
                        # Fallback to BS - use IV at strike (k_t), not ATM
                        delta = self.calculate_black_scholes_delta(
                            futures_price, position.strike, T,
                            self.risk_free_rate, iv_at_strike, position.option_type
                        )
                except Exception as e:
                    logger.warning(f"Error calculating min var delta: {e}, falling back to BS")
                    # Fallback to BS - use IV at strike (k_t), not ATM
                    delta = self.calculate_black_scholes_delta(
                        futures_price, position.strike, T,
                        self.risk_free_rate, iv_at_strike, position.option_type
                    )
            else:
                # Use Black-Scholes delta - use IV at strike (k_t), not ATM
                delta = self.calculate_black_scholes_delta(
                    futures_price,
                    position.strike,
                    T,
                    self.risk_free_rate,
                    iv_at_strike,  # IV at strike, not ATM
                    position.option_type
                )
            
            deltas_by_exp[expiration_str] += position.quantity * delta
        
        # Apply hedge for each expiration
        hedges = []
        for expiration_str, total_delta in deltas_by_exp.items():
            print(f"Expiration: {expiration_str}, Total Delta: {total_delta}")
            hedge_quantity = -total_delta
            
            if abs(hedge_quantity) > 0.01:
                futures_price = underlying_prices_by_exp[expiration_str]
                
                # Determine futures instrument name
                # Try to find futures contract for this expiration
                futures_name = self._get_futures_instrument_name(expiration_str)
                
                # Check if futures exists by trying to load its price
                futures_price_check = self.load_futures_price(timestamp, expiration_str)
                if futures_price_check is None:
                    # No futures exists for this expiration, use perpetual
                    futures_name = "BTC-PERPETUAL"
                    futures_price = underlying_price
                else:
                    futures_price = futures_price_check
                    futures_name = f"BTC-{expiration_str}"
                
                hedge = HedgePosition(
                    timestamp=timestamp,
                    total_delta=total_delta,
                    hedge_quantity=hedge_quantity,
                    hedge_cost=abs(hedge_quantity) * futures_price * 0.0001
                )
                
                self.hedges.append(hedge)
                
                self.trades.append(BacktestTrade(
                    timestamp=timestamp,
                    instrument_name=futures_name,
                    quantity=hedge_quantity,
                    price=futures_price,
                    trade_type='hedge'
                ))
                
                # Update or create futures position
                if expiration_str in self.futures_positions:
                    # Update existing position (rebalance)
                    existing_pos = self.futures_positions[expiration_str]
                    # Calculate the change needed
                    delta_quantity = hedge_quantity - existing_pos.quantity
                    print("Update existing futures position")
                    if abs(delta_quantity) > 0.01:
                        # Update position
                        existing_pos.quantity = hedge_quantity
                        existing_pos.current_price = futures_price
                        # If quantity changed sign, update entry price
                        if existing_pos.quantity * existing_pos.entry_price < 0:
                            existing_pos.entry_price = futures_price
                            existing_pos.entry_timestamp = timestamp
                else:
                    print("Create new futures position")
                    # Create new futures position
                    futures_pos = FuturesPosition(
                        instrument_name=futures_name,
                        expiration_str=expiration_str,
                        quantity=hedge_quantity,
                        entry_price=futures_price,
                        entry_timestamp=timestamp,
                        current_price=futures_price
                    )
                    self.futures_positions[expiration_str] = futures_pos
                
                hedges.append(hedge)
        
        return hedges
    
    def calculate_daily_pnl_decomposition(
        self,
        timestamp: int,
        current_underlying_price: float,
        curves: Dict[str, Dict[str, Any]],
        prev_underlying_price: Optional[float] = None,
        prev_curves: Optional[Dict[str, Dict[str, Any]]] = None,
        prev_positions_prices: Optional[Dict[str, float]] = None
    ) -> Optional[DailyPnLDecomposition]:
        """
        Calculate daily PnL decomposition.
        
        Args:
            timestamp: Current timestamp
            current_underlying_price: Current underlying price
            curves: Current volatility curves
            prev_underlying_price: Previous underlying price (if None, uses self.previous_underlying_price)
            prev_curves: Previous volatility curves (if None, uses self.previous_curves)
            prev_positions_prices: Previous position prices (if None, uses self.previous_positions_prices)
        """
        if not self.positions or prev_underlying_price is None or prev_curves is None or prev_positions_prices is None:
            return None
        
        total_funding = 0.0
        total_ir_theta = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_vol_block = 0.0
        total_vanna_block = 0.0
        positions_pnl = {}
        
        for position in self.positions:
            if position.current_price is None:
                continue
            
            expiration_str = self._extract_expiration_str(position.instrument_name)
            curve = curves.get(expiration_str)
            prev_curve = prev_curves.get(expiration_str) if prev_curves else None
            
            if not curve or not prev_curve:
                continue
            
            T = max((position.expiration_timestamp - timestamp / 1000) / (365 * 24 * 3600), 0.001)
            dt = 1.0 / 365.0
            
            k_t = math.log(position.strike / current_underlying_price)
            k_t_prev = math.log(position.strike / prev_underlying_price)
            
            # Try to use SVI parameters if available, otherwise use linear approximation
            svi_params = curve.get('svi_params')
            prev_svi_params = prev_curve.get('svi_params') if prev_curve else None
            
            if svi_params is not None and prev_svi_params is not None:
                # Use exact SVI calculation
                from ..analytics.svi import (
                    svi_iv_at_moneyness, svi_slope_at_moneyness, svi_curvature_at_moneyness
                )
                # svi_iv_at_moneyness expects numpy array, returns array
                iv_at_strike = float(svi_iv_at_moneyness(np.array([k_t]), T, svi_params)[0])
                prev_iv_at_strike = float(svi_iv_at_moneyness(np.array([k_t_prev]), T, prev_svi_params)[0])
                d_iv_dk = svi_slope_at_moneyness(k_t, T, svi_params)
                prev_d_iv_dk = svi_slope_at_moneyness(k_t_prev, T, prev_svi_params)
                d2_iv_dk2 = svi_curvature_at_moneyness(k_t, T, svi_params)
            else:
                logger.warning("No SVI parameters found, falling back to linear approximation")
                # Fallback to linear approximation
                atm_iv = curve.get('atm_iv', 0.8)
                # Normalize if stored as percentage (e.g., 85 -> 0.85)
                if atm_iv > 1.0:
                    atm_iv = atm_iv / 100.0
                atm_slope = curve.get('atm_slope', 0.0)
                prev_atm_iv = prev_curve.get('atm_iv', atm_iv)
                # Normalize if stored as percentage
                if prev_atm_iv > 1.0:
                    prev_atm_iv = prev_atm_iv / 100.0
                prev_atm_slope = prev_curve.get('atm_slope', atm_slope)
                
                iv_at_strike = atm_iv + atm_slope * k_t
                prev_iv_at_strike = prev_atm_iv + prev_atm_slope * k_t_prev
                d_iv_dk = atm_slope
                prev_d_iv_dk = prev_atm_slope
                curvature = curve.get('curvature', 0.0)
                d2_iv_dk2 = curvature if curvature else 0.0
            
            # Calculate IV moves at k0 (ATM) for decomposition
            # Use SVI at k=0 (ATM) if available, otherwise use curve values
            if svi_params is not None:
                from ..analytics.svi import svi_iv_at_moneyness, svi_slope_at_moneyness
                atm_iv_for_k0 = float(svi_iv_at_moneyness(np.array([0.0]), T, svi_params)[0])
                atm_slope_for_k0 = svi_slope_at_moneyness(0.0, T, svi_params)
            else:
                atm_iv_for_k0 = curve.get('atm_iv', 0.8)
                if atm_iv_for_k0 > 1.0:
                    atm_iv_for_k0 = atm_iv_for_k0 / 100.0
                atm_slope_for_k0 = curve.get('atm_slope', 0.0)
            
            if prev_svi_params is not None:
                from ..analytics.svi import svi_iv_at_moneyness, svi_slope_at_moneyness
                prev_atm_iv_for_k0 = float(svi_iv_at_moneyness(np.array([0.0]), T, prev_svi_params)[0])
                prev_atm_slope_for_k0 = svi_slope_at_moneyness(0.0, T, prev_svi_params)
            else:
                prev_atm_iv_for_k0 = prev_curve.get('atm_iv', atm_iv_for_k0)
                if prev_atm_iv_for_k0 > 1.0:
                    prev_atm_iv_for_k0 = prev_atm_iv_for_k0 / 100.0
                prev_atm_slope_for_k0 = prev_curve.get('atm_slope', atm_slope_for_k0)
            
            delta_iv_k0 = atm_iv_for_k0 - prev_atm_iv_for_k0
            delta_d_iv_dk_k0 = atm_slope_for_k0 - prev_atm_slope_for_k0
  
            
            prev_price_usd = prev_positions_prices.get(position.instrument_name)
            if prev_price_usd is None:
                prev_price_usd = position.entry_price
            
            current_price_usd = position.current_price
            if current_price_usd is None or prev_price_usd is None:
                continue
            
            # Ensure USD prices
            if prev_price_usd < 100 and prev_underlying_price:
                prev_price_usd = prev_price_usd * prev_underlying_price
            if current_price_usd < 100 and current_underlying_price:
                current_price_usd = current_price_usd * current_underlying_price
            
            try:
                # Calculate decomposition per unit (V_t and Pi_t should be per unit for decomposition)
                result = self.pnl_decomposer.decompose(
                    F_t=prev_underlying_price,
                    F_next=current_underlying_price,
                    K=position.strike,
                    tau_years=T,
                    dt=dt,
                    V_t=prev_price_usd,  # Per unit value
                    Pi_t=prev_price_usd,  # Per unit value
                    option_type=position.option_type,  # 'call' or 'put'
                    iv1y_t_kt=prev_iv_at_strike,
                    d_iv1y_dk_t_kt=prev_d_iv_dk,
                    d2_iv1y_dk2_t_kt=d2_iv_dk2,
                    delta_iv1y_k0=delta_iv_k0,
                    delta_d_iv1y_dk_k0=delta_d_iv_dk_k0,
                )
                
                # Scale all components by position quantity (sign and amount)
                # For short positions (negative quantity), all components are correctly inverted
                total_funding += result.funding * position.quantity
                total_ir_theta += result.ir_theta * position.quantity
                total_delta += result.delta * position.quantity
                total_gamma += result.gamma * position.quantity
                total_vol_block += result.vol_block * position.quantity
                total_vanna_block += result.vanna_block * position.quantity
    
                position_pnl = (current_price_usd - prev_price_usd) * position.quantity   
                positions_pnl[position.instrument_name] = position_pnl
                
            except Exception as e:
                logger.debug(f"Error decomposing PnL: {e}")
                continue
        
        total_pnl = sum(positions_pnl.values())
        decomposed_sum = total_funding + total_ir_theta + total_delta + total_gamma + total_vol_block + total_vanna_block
        pnl_error = total_pnl - decomposed_sum
        
        return DailyPnLDecomposition(
            timestamp=timestamp,
            total_pnl=total_pnl,
            funding=total_funding,
            ir_theta=total_ir_theta,
            delta=total_delta,
            gamma=total_gamma,
            vol_block=total_vol_block,
            vanna_block=total_vanna_block,
            positions_pnl=positions_pnl,
            decomposed_pnl_sum=decomposed_sum,
            pnl_error=pnl_error
        )

    
    def run_backtest(
        self,
        start_timestamp: int,
        end_timestamp: int,
        hedge_hour: int = 9  # Default: 9:00 AM
    ) -> BacktestResult:
        """
        Run backtest for buying and holding options.
        
        Calculates and hedges every 24 hours at specified hour (default 9:00 AM).
        PnL is calculated using Greeks: delta_pnl = dV_dF * dF where dV_dF is from previous iteration.
        
        Args:
            start_timestamp: Start timestamp in milliseconds
            end_timestamp: End timestamp in milliseconds
            hedge_hour: Hour of day to calculate and hedge (default: 9 for 9:00 AM)
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Simple Option Buy-and-Hold Backtest")
        logger.info("=" * 60)
        logger.info(f"Portfolio: {len(self.options)} option(s)")
        for i, opt in enumerate(self.options, 1):
            logger.info(f"  {i}. {opt.quantity:+.2f} {opt.expiration_str} {opt.strike} {opt.option_type}")
        logger.info(f"Delta Hedge: {self.use_delta_hedge} ({self.hedge_method})")
        logger.info(f"Hedge Time: {hedge_hour}:00 AM every 24 hours")
        logger.info(f"Period: {datetime.fromtimestamp(start_timestamp/1000)} to {datetime.fromtimestamp(end_timestamp/1000)}")
        
        # Get unique expirations
        expirations = list(set(opt.expiration_str for opt in self.options))
        logger.info(f"Expirations: {', '.join(expirations)}")
        
        # Load volatility curves for all expirations
        curves_by_exp = {}
        for exp_str in expirations:
            curves = self.load_volatility_curves(start_timestamp, end_timestamp, exp_str)
            if curves:
                curves_by_exp[exp_str] = {c['timestamp']: c for c in curves}
        
        if not curves_by_exp:
            logger.error("Could not load volatility curves")
            return BacktestResult(
                total_pnl=0.0,
                total_trades=0,
                total_hedges=0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                trades=[],
                daily_pnl=[],
                daily_pnl_decomposition=[],
                positions_history=[]
            )
        
        # Get all unique timestamps
        all_timestamps = sorted(set(
            ts for curves_map in curves_by_exp.values() for ts in curves_map.keys()
        ))
        all_timestamps = [t for t in all_timestamps if start_timestamp <= t <= end_timestamp]
        
        # Find closest timestamp to hedge_hour each day
        # Group timestamps by day
        timestamps_by_day: Dict[str, List[int]] = {}
        for ts in all_timestamps:
            dt = datetime.fromtimestamp(ts / 1000)
            day_key = dt.strftime('%Y-%m-%d')
            if day_key not in timestamps_by_day:
                timestamps_by_day[day_key] = []
            timestamps_by_day[day_key].append(ts)
        
        # For each day, find the timestamp closest to hedge_hour
        hedge_timestamps = []
        target_hour_seconds = hedge_hour * 3600  # Convert hour to seconds since midnight
        
        for day_key, day_timestamps in timestamps_by_day.items():
            if not day_timestamps:
                continue
            
            # Find timestamp closest to target hour
            best_ts = None
            min_diff = float('inf')
            
            for ts in day_timestamps:
                dt = datetime.fromtimestamp(ts / 1000)
                # Calculate seconds since midnight
                seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
                # Calculate difference (handle wrap-around at midnight)
                diff = abs(seconds_since_midnight - target_hour_seconds)
                # Also check wrap-around (e.g., if target is 9:00 and we have 23:00, diff is 10 hours)
                diff = min(diff, 86400 - diff)
                
                if diff < min_diff:
                    min_diff = diff
                    best_ts = ts
            
            if best_ts is not None:
                hedge_timestamps.append(best_ts)
        
        hedge_timestamps.sort()
        
        logger.info(f"Found {len(hedge_timestamps)} timestamps closest to {hedge_hour}:00 AM (one per day)")
        
        if not hedge_timestamps:
            logger.error("No timestamps found at specified hedge hour")
            return BacktestResult(
                total_pnl=0.0,
                total_trades=0,
                total_hedges=0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                trades=[],
                daily_pnl=[],
                daily_pnl_decomposition=[],
                positions_history=[]
            )
        
        # Track previous state for PnL calculation using Greeks
        F_t_prev: Optional[float] = None
        curves_prev: Dict[str, Dict[str, Any]] = {}
        positions_prices_prev: Dict[str, float] = {}
        futures_positions_prev: Dict[str, FuturesPosition] = {}
        positions_established = False
        
        # Process only hedge timestamps (every 24 hours at hedge_hour)
        for idx, timestamp in enumerate(hedge_timestamps):
            # Get curves and underlying price for this timestamp
            curves = {}
            F_t_current = None
            
            for exp_str in expirations:
                curve_map = curves_by_exp.get(exp_str, {})
                curve = curve_map.get(timestamp)
                if curve:
                    curves[exp_str] = curve
                    if F_t_current is None:
                        F_t_current = curve.get('underlying_price')
            
            if not curves or not F_t_current:
                F_t_current = self.load_futures_price(timestamp)
            
            if not F_t_current:
                continue
            
            # Establish positions on first timestamp
            if not positions_established:
                self.establish_positions(timestamp, F_t_current)
                positions_established = True
                if not self.positions:
                    logger.error("Failed to establish any positions")
                    break
                
                # Initialize previous state with entry state
                F_t_prev = F_t_current
                curves_prev = curves.copy()
                positions_prices_prev = {pos.instrument_name: pos.entry_price 
                                       for pos in self.positions if pos.entry_price is not None}
                print(f"positions_prices_prev: {positions_prices_prev}")
                futures_positions_prev = {}
                
                logger.info(f"Established positions at {datetime.fromtimestamp(timestamp/1000)}")
                continue  # Skip PnL calculation on first timestamp
            
            # Update positions to get current prices
            self.update_positions_pnl(timestamp, F_t_current)
            
            # Calculate PnL using Greeks from previous iteration
            # delta_pnl = dV_dF * dF where dF = F_t_current - F_t_prev
            dF = F_t_current - F_t_prev

            # Calculate PnL decomposition using previous state
            daily_decomposition = self.calculate_daily_pnl_decomposition(
                timestamp=timestamp,
                current_underlying_price=F_t_current,
                curves=curves,
                prev_underlying_price=F_t_prev,
                prev_curves=curves_prev,
                prev_positions_prices=positions_prices_prev
            )
            
            # Calculate actual PnL from option price changes
            actual_pnl = 0.0
            for pos in self.positions:
                prev_price = positions_prices_prev.get(pos.instrument_name)
                if prev_price is None:
                    prev_price = pos.entry_price
                if prev_price is None or pos.current_price is None:
                    continue
                
                # Ensure USD prices
                if prev_price < 100:
                    prev_price = prev_price * F_t_prev
                if pos.current_price < 100:
                    current_price_usd = pos.current_price * F_t_current
                else:
                    current_price_usd = pos.current_price
                
                actual_pnl += (current_price_usd - prev_price) * pos.quantity

            # Add futures PnL if hedging
            if self.use_delta_hedge:
                for exp_str, futures_pos in self.futures_positions.items():
                    prev_futures_pos = futures_positions_prev.get(exp_str)
                    if prev_futures_pos and prev_futures_pos.current_price is not None:
                        if futures_pos.current_price is not None:
                            futures_pnl = (futures_pos.current_price - prev_futures_pos.current_price) * prev_futures_pos.quantity
                            actual_pnl += futures_pnl
            
            # Save daily PnL and decomposition
            day_timestamp = int(datetime.fromtimestamp(timestamp / 1000).replace(hour=0, minute=0, second=0).timestamp() * 1000)
            self.daily_pnl.append((day_timestamp, actual_pnl))
            
            if daily_decomposition:
                daily_decomposition.timestamp = day_timestamp
                daily_decomposition.total_pnl = actual_pnl
                self.daily_pnl_decomposition.append(daily_decomposition)
            
            # Apply delta hedge (if enabled)
            if self.use_delta_hedge:
                self.apply_delta_hedge(timestamp, F_t_current, curves)
            
            # Update previous state for next iteration
            # F_next should be exactly the same as F_t of next iteration
            F_t_prev = F_t_current
            curves_prev = curves.copy()
            positions_prices_prev = {pos.instrument_name: pos.current_price 
                                    for pos in self.positions if pos.current_price is not None}
            futures_positions_prev = {
                exp_str: FuturesPosition(
                    instrument_name=fp.instrument_name,
                    expiration_str=fp.expiration_str,
                    quantity=fp.quantity,
                    entry_price=fp.entry_price,
                    entry_timestamp=fp.entry_timestamp,
                    current_price=fp.current_price
                ) for exp_str, fp in self.futures_positions.items()
            }
            
            # Progress update
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx+1}/{len(hedge_timestamps)} ({idx*100//len(hedge_timestamps)}%) - "
                          f"PnL: ${actual_pnl:,.2f}, dF: ${dF:,.2f}")
        
        # Calculate final metrics
        pnl_series = [pnl for _, pnl in self.daily_pnl]
        total_pnl = sum(pnl_series) if pnl_series else 0.0
        
        if len(pnl_series) > 1 and np.std(pnl_series) > 0:
            sharpe_ratio = (np.mean(pnl_series) / np.std(pnl_series)) * math.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        logger.info("\n" + "=" * 60)
        logger.info("Backtest Complete")
        logger.info("=" * 60)
        logger.info(f"Portfolio: {len(self.positions)} position(s)")
        for pos in self.positions:
            final_pnl = (pos.current_price - pos.entry_price) * pos.quantity if pos.current_price and pos.entry_price else 0.0
            logger.info(f"  - {pos.instrument_name}: {pos.quantity:+.2f} @ ${pos.entry_price:,.2f} → ${pos.current_price or 0:,.2f} (PnL: ${final_pnl:,.2f})")
        logger.info(f"Total PnL: ${total_pnl:,.2f}")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Total Hedges: {len(self.hedges)}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: ${max_drawdown:,.2f}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        
        return BacktestResult(
            total_pnl=total_pnl,
            total_trades=len(self.trades),
            total_hedges=len(self.hedges),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades=self.trades,
            daily_pnl=self.daily_pnl,
            daily_pnl_decomposition=self.daily_pnl_decomposition,
            positions_history=[]
        )
    
    def plot_cumulative_pnl(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot cumulative PnL - uses BacktestPlotter"""
        plotter = BacktestPlotter(result)
        return plotter.plot_cumulative_pnl(save_path)
    
    def plot_daily_pnl_decomposition(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot daily PnL decomposition - uses BacktestPlotter"""
        plotter = BacktestPlotter(result)
        return plotter.plot_daily_pnl_decomposition(save_path)
    
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._disconnect_db()
        self,

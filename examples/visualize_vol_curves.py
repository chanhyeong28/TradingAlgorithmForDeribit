#!/usr/bin/env python3
"""
Visualize Volatility Curves from Database

This script loads historical volatility curves from the database and visualizes
them to check for anomalous ATM slope data and overall curve evolution.
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deribit_trading_toolkit import SimpleOptionBacktester
from deribit_trading_toolkit.models.market_data import VolatilityCurve, VolatilityPoint, OptionType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_volatility_curves_from_db(
    backtester: SimpleOptionBacktester,
    start_timestamp: int,
    end_timestamp: int,
    expiration_str: str
) -> List[Tuple[int, Optional[VolatilityCurve]]]:
    """
    Load volatility curves from database and convert to VolatilityCurve objects.
    
    Args:
        backtester: SimpleOptionBacktester instance with DB connection
        start_timestamp: Start timestamp in milliseconds
        end_timestamp: End timestamp in milliseconds
        expiration_str: Expiration string (e.g., "26DEC25")
        
    Returns:
        List of (timestamp_ms, VolatilityCurve) tuples
    """
    backtester._connect_db()
    
    try:
        sql = """
            SELECT timestamp, expiration_timestamp, expiration_str, underlying_price,
                   atm_iv, atm_slope, curvature, num_points, curve_data
            FROM btc_historical_volatility_curves
            WHERE expiration_str = %s
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp ASC
        """
        
        backtester.db_cursor.execute(sql, (expiration_str, start_timestamp, end_timestamp))
        results = backtester.db_cursor.fetchall()
        
        curves = []
        for row in results:
            try:
                # Get expiration timestamp first
                expiration_epoch = row['expiration_timestamp']
                if isinstance(expiration_epoch, int):
                    expiration_epoch_sec = expiration_epoch
                else:
                    expiration_epoch_sec = int(expiration_epoch)
                
                # Parse curve_data JSON
                curve_data_json = row['curve_data']
                if isinstance(curve_data_json, str):
                    curve_data = json.loads(curve_data_json)
                else:
                    curve_data = curve_data_json if curve_data_json else []
                
                # Convert curve_data to VolatilityPoint objects
                points = []
                for point_data in curve_data:
                    if isinstance(point_data, dict):
                        option_type = OptionType.CALL  # Default
                        if 'option_type' in point_data:
                            opt_type_str = point_data['option_type']
                            if opt_type_str == 'call':
                                option_type = OptionType.CALL
                            elif opt_type_str == 'put':
                                option_type = OptionType.PUT
                        
                        point = VolatilityPoint(
                            strike=point_data.get('strike', 0.0),
                            iv=point_data.get('iv', 0.0),
                            moneyness=point_data.get('log_moneyness', point_data.get('moneyness', 0.0)),
                            option_type=option_type,
                            expiration=expiration_epoch_sec,
                            timestamp=row['timestamp']
                        )
                        points.append(point)
                
                # Normalize atm_iv if stored as percentage
                atm_iv = row['atm_iv']
                if atm_iv > 1.0:
                    atm_iv = atm_iv / 100.0
                
                # Calculate ATM strike (approximate from underlying price)
                underlying_price = row['underlying_price']
                atm_strike = underlying_price  # Approximate
                
                # Create VolatilityCurve object
                curve = VolatilityCurve(
                    expiration=expiration_epoch_sec,
                    points=points,
                    atm_strike=atm_strike,
                    atm_iv=atm_iv,
                    slope=row['atm_slope'],
                    curvature=row.get('curvature', 0.0),
                    timestamp=row['timestamp']
                )
                
                curves.append((row['timestamp'], curve))
                
            except Exception as e:
                logger.warning(f"Error parsing curve at timestamp {row['timestamp']}: {e}")
                curves.append((row['timestamp'], None))
        
        logger.info(f"Loaded {len([c for _, c in curves if c is not None])} valid curves out of {len(curves)} total for {expiration_str}")
        return curves
        
    except Exception as e:
        logger.error(f"Error loading volatility curves: {e}")
        return []


def plot_volatility_surface_evolution(
    curves: List[Tuple[int, Optional[VolatilityCurve]]],
    expiration_str: str,
    max_curves: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot volatility surface evolution by showing multiple curves at different times.
    
    Args:
        curves: List of (timestamp_ms, VolatilityCurve) tuples
        expiration_str: Expiration string for title
        max_curves: Maximum number of curves to plot (None = plot all)
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    valid_curves = [(ts, c) for ts, c in curves if c is not None]
    
    if not valid_curves:
        logger.warning("No valid curves to plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No valid curves to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Select curves to plot (evenly spaced if max_curves is specified, otherwise plot all)
    if max_curves and len(valid_curves) > max_curves:
        indices = np.linspace(0, len(valid_curves) - 1, max_curves, dtype=int)
        selected_curves = [valid_curves[i] for i in indices]
        logger.info(f"Plotting {max_curves} curves out of {len(valid_curves)} total (evenly spaced)")
    else:
        selected_curves = valid_curves
        logger.info(f"Plotting all {len(valid_curves)} curves")
    
    # Use larger figure for many curves
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Color map for different curves (use colormap that works well with many curves)
    if len(selected_curves) > 50:
        # For many curves, use a continuous colormap with lower alpha
        colors = plt.cm.plasma(np.linspace(0, 1, len(selected_curves)))
        alpha = 0.4
        linewidth = 1.5
        markersize = 2
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_curves)))
        alpha = 0.7
        linewidth = 2
        markersize = 4
    
    # Track if we've added legend entry
    legend_added = False
    
    for idx, (timestamp, curve) in enumerate(selected_curves):
        dt = datetime.fromtimestamp(timestamp / 1000)
        
        # Extract moneyness and IV from curve points
        if not curve.points:
            continue
        
        sorted_points = sorted(curve.points, key=lambda p: p.moneyness)
        moneyness = [p.moneyness for p in sorted_points]
        iv = [p.iv for p in sorted_points]
        
        # Separate calls and puts for better visualization
        call_points = [p for p in sorted_points if p.option_type == OptionType.CALL]
        put_points = [p for p in sorted_points if p.option_type == OptionType.PUT]
        
        # Only add label for first, middle, and last curves to avoid cluttering
        add_label = (idx == 0 or idx == len(selected_curves) - 1 or 
                    (len(selected_curves) > 10 and idx == len(selected_curves) // 2))
        
        # Plot full curve with all strikes
        if call_points and put_points:
            # Plot calls and puts separately for clarity
            call_moneyness = [p.moneyness for p in call_points]
            call_iv = [p.iv for p in call_points]
            put_moneyness = [p.moneyness for p in put_points]
            put_iv = [p.iv for p in put_points]
            
            if add_label:
                label = f'{dt.strftime("%Y-%m-%d %H:%M")} (IV={curve.atm_iv:.3f}, Slope={curve.slope:.4f})'
            else:
                label = None
            
            ax.plot(call_moneyness, call_iv, marker='o', linestyle='-', linewidth=linewidth, 
                   markersize=markersize, color=colors[idx], alpha=alpha, label=label if add_label else None)
            ax.plot(put_moneyness, put_iv, marker='s', linestyle='--', linewidth=linewidth, 
                   markersize=markersize-1, color=colors[idx], alpha=alpha*0.7)
        else:
            # Plot combined curve
            if add_label:
                label = f'{dt.strftime("%Y-%m-%d %H:%M")} (IV={curve.atm_iv:.3f}, Slope={curve.slope:.4f})'
            else:
                label = None
            ax.plot(moneyness, iv, marker='o', linestyle='-', linewidth=linewidth, 
                   markersize=markersize, color=colors[idx], alpha=alpha, label=label)
    
    ax.set_xlabel('Log-Moneyness (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Implied Volatility', fontsize=12, fontweight='bold')
    ax.set_title(f'Volatility Surface Evolution: {expiration_str}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='ATM')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def identify_anomalous_curves(
    curves: List[Tuple[int, Optional[VolatilityCurve]]],
    std_threshold: float = 3.0
) -> List[Tuple[int, VolatilityCurve, float]]:
    """
    Identify curves with anomalous ATM slope (skew).
    
    Args:
        curves: List of (timestamp_ms, VolatilityCurve) tuples
        std_threshold: Number of standard deviations from mean to consider anomalous
        
    Returns:
        List of (timestamp_ms, VolatilityCurve, deviation) tuples for anomalous curves
    """
    valid_curves = [(ts, c) for ts, c in curves if c is not None]
    
    if len(valid_curves) < 2:
        return []
    
    # Calculate mean and std of ATM slope
    slopes = [c.slope for _, c in valid_curves]
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    
    if std_slope == 0:
        return []
    
    # Find anomalous curves
    anomalous = []
    for timestamp, curve in valid_curves:
        deviation = abs(curve.slope - mean_slope) / std_slope
        if deviation > std_threshold:
            anomalous.append((timestamp, curve, deviation))
    
    # Sort by deviation (most anomalous first)
    anomalous.sort(key=lambda x: x[2], reverse=True)
    
    return anomalous


def plot_anomalous_volatility_curves(
    curves: List[Tuple[int, Optional[VolatilityCurve]]],
    expiration_str: str,
    std_threshold: float = 3.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot only volatility curves with anomalous ATM slope (skew).
    
    Args:
        curves: List of (timestamp_ms, VolatilityCurve) tuples
        expiration_str: Expiration string for title
        std_threshold: Number of standard deviations from mean to consider anomalous
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Identify anomalous curves
    anomalous_curves = identify_anomalous_curves(curves, std_threshold)
    
    if not anomalous_curves:
        logger.warning(f"No anomalous curves found (threshold: {std_threshold} std dev)")
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(0.5, 0.5, f'No anomalous curves found\n(threshold: {std_threshold} std dev)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Anomalous Volatility Curves: {expiration_str}', fontsize=16, fontweight='bold')
        return fig
    
    logger.info(f"Found {len(anomalous_curves)} anomalous curves out of {len([c for _, c in curves if c is not None])} total")
    
    # Get all valid curves for comparison
    valid_curves = [(ts, c) for ts, c in curves if c is not None]
    all_slopes = [c.slope for _, c in valid_curves]
    mean_slope = np.mean(all_slopes)
    std_slope = np.std(all_slopes)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot all curves in light gray for reference
    colors_all = plt.cm.Greys(np.linspace(0.3, 0.7, len(valid_curves)))
    for idx, (timestamp, curve) in enumerate(valid_curves):
        if not curve.points:
            continue
        
        sorted_points = sorted(curve.points, key=lambda p: p.moneyness)
        moneyness = [p.moneyness for p in sorted_points]
        iv = [p.iv for p in sorted_points]
        
        # Plot in light gray
        ax.plot(moneyness, iv, linestyle='-', linewidth=0.5, 
               color=colors_all[idx], alpha=0.2, zorder=1)
    
    # Plot anomalous curves in bright colors
    colors_anomalous = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(anomalous_curves)))
    
    for idx, (timestamp, curve, deviation) in enumerate(anomalous_curves):
        dt = datetime.fromtimestamp(timestamp / 1000)
        
        if not curve.points:
            continue
        
        sorted_points = sorted(curve.points, key=lambda p: p.moneyness)
        moneyness = [p.moneyness for p in sorted_points]
        iv = [p.iv for p in sorted_points]
        
        # Separate calls and puts
        call_points = [p for p in sorted_points if p.option_type == OptionType.CALL]
        put_points = [p for p in sorted_points if p.option_type == OptionType.PUT]
        
        if call_points and put_points:
            call_moneyness = [p.moneyness for p in call_points]
            call_iv = [p.iv for p in call_points]
            put_moneyness = [p.moneyness for p in put_points]
            put_iv = [p.iv for p in put_points]
            
            label = f'{dt.strftime("%Y-%m-%d %H:%M")} | Slope={curve.slope:.4f} ({deviation:.1f}σ)'
            ax.plot(call_moneyness, call_iv, marker='o', linestyle='-', linewidth=3, 
                   markersize=6, color=colors_anomalous[idx], alpha=0.9, 
                   label=label, zorder=10)
            ax.plot(put_moneyness, put_iv, marker='s', linestyle='--', linewidth=2.5, 
                   markersize=5, color=colors_anomalous[idx], alpha=0.7, zorder=10)
        else:
            label = f'{dt.strftime("%Y-%m-%d %H:%M")} | Slope={curve.slope:.4f} ({deviation:.1f}σ)'
            ax.plot(moneyness, iv, marker='o', linestyle='-', linewidth=3, 
                   markersize=6, color=colors_anomalous[idx], alpha=0.9, 
                   label=label, zorder=10)
    
    ax.set_xlabel('Log-Moneyness (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Implied Volatility', fontsize=12, fontweight='bold')
    ax.set_title(f'Anomalous Volatility Curves (Skew Anomalies): {expiration_str}\n'
                f'Mean Slope: {mean_slope:.4f}, Std: {std_slope:.4f}, Threshold: {std_threshold}σ',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='ATM')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_atm_slope_over_time(
    curves: List[Tuple[int, Optional[VolatilityCurve]]],
    expiration_str: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ATM slope over time to identify anomalies.
    
    Args:
        curves: List of (timestamp_ms, VolatilityCurve) tuples
        expiration_str: Expiration string for title
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    valid_curves = [(ts, c) for ts, c in curves if c is not None]
    
    if not valid_curves:
        logger.warning("No valid curves to plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No valid curves to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    timestamps = [ts / 1000 for ts, _ in valid_curves]
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    atm_ivs = [c.atm_iv for _, c in valid_curves]
    atm_slopes = [c.slope for _, c in valid_curves]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot ATM IV over time
    ax1.plot(dates, atm_ivs, marker='o', linestyle='-', linewidth=2, markersize=4, color='#2c3e50')
    ax1.set_ylabel('ATM Implied Volatility', fontsize=12, fontweight='bold')
    ax1.set_title(f'ATM IV Over Time: {expiration_str}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot ATM Slope over time
    ax2.plot(dates, atm_slopes, marker='s', linestyle='-', linewidth=2, markersize=4, color='#e74c3c')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ATM Slope', fontsize=12, fontweight='bold')
    ax2.set_title(f'ATM Slope Over Time: {expiration_str}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight anomalies (slope > 3 standard deviations from mean)
    if len(atm_slopes) > 1:
        mean_slope = np.mean(atm_slopes)
        std_slope = np.std(atm_slopes)
        threshold = 3 * std_slope
        
        anomalies = [(d, s) for d, s in zip(dates, atm_slopes) if abs(s - mean_slope) > threshold]
        if anomalies:
            anomaly_dates, anomaly_slopes = zip(*anomalies)
            ax2.scatter(anomaly_dates, anomaly_slopes, color='red', s=100, zorder=5, 
                       label=f'Anomalies (>{threshold:.4f} std dev)')
            ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


async def main():
    """Main function to visualize volatility curves"""
    try:
        logger.info("=" * 60)
        logger.info("Volatility Curves Visualization")
        logger.info("=" * 60)
        
        # Configuration
        expiration_str = "26DEC25"  # Change this to your expiration
        days_back = 40  # Number of days to look back
        
        # Calculate timestamps
        end_timestamp = int(datetime.now().timestamp() * 1000)
        start_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        logger.info(f"Expiration: {expiration_str}")
        logger.info(f"Period: {datetime.fromtimestamp(start_timestamp/1000)} to {datetime.fromtimestamp(end_timestamp/1000)}")
        
        # Create backtester instance for DB access
        with SimpleOptionBacktester(options=[]) as backtester:
            # Load volatility curves from database
            logger.info("Loading volatility curves from database...")
            curves = load_volatility_curves_from_db(
                backtester,
                start_timestamp,
                end_timestamp,
                expiration_str
            )
            
            if not curves:
                logger.error("No curves found in database!")
                return
            
            logger.info(f"Loaded {len([c for _, c in curves if c is not None])} valid curves")
            
            # Plot volatility surface evolution (plot all curves)
            logger.info("Generating volatility surface evolution plot...")
            fig1 = plot_volatility_surface_evolution(
                curves,
                expiration_str,
                max_curves=None,  # Plot all curves
                save_path=f"vol_surface_evolution_{expiration_str}.png"
            )
            
            # Plot anomalous volatility curves only
            logger.info("Generating anomalous volatility curves plot...")
            fig2 = plot_anomalous_volatility_curves(
                curves,
                expiration_str,
                std_threshold=3.0,
                save_path=f"anomalous_vol_curves_{expiration_str}.png"
            )
            
            # Plot ATM slope over time
            logger.info("Generating ATM slope over time plot...")
            fig3 = plot_atm_slope_over_time(
                curves,
                expiration_str,
                save_path=f"atm_slope_timeseries_{expiration_str}.png"
            )
            
            logger.info("=" * 60)
            logger.info("Visualization complete!")
            logger.info("=" * 60)
            logger.info(f"Plots saved:")
            logger.info(f"  - vol_surface_evolution_{expiration_str}.png")
            logger.info(f"  - anomalous_vol_curves_{expiration_str}.png")
            logger.info(f"  - atm_slope_timeseries_{expiration_str}.png")
            
            # Show plots
            plt.show()
            
    except Exception as e:
        logger.error(f"Error running visualization: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())


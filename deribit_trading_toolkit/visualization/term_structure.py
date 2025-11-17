"""
Volatility Term Structure Visualization

Provides visualization functions for ATM volatility term structure.
"""

import math
import logging
import numpy as np
import plotly.graph_objs as go
from typing import Dict, List, Optional
from datetime import datetime

from deribit_trading_toolkit import VolatilityCurve

logger = logging.getLogger(__name__)


def build_term_structure_figure(curves: Dict[int, VolatilityCurve]) -> go.Figure:
    """
    Build ATM volatility term structure figure from volatility curves.
    
    Args:
        curves: Dictionary mapping expiration timestamp to VolatilityCurve
        
    Returns:
        Plotly figure showing term structure
    """
    fig = go.Figure()
    
    if not curves:
        fig.update_layout(
            title="ATM Volatility Term Structure (waiting for data)",
            xaxis_title="Days to Expiry",
            yaxis_title="ATM Implied Volatility",
        )
        return fig
    
    # Extract term structure data
    days_to_expiry = []
    atm_ivs = []
    labels = []
    
    for exp_ts, curve in sorted(curves.items()):
        if not curve or curve.atm_iv <= 0:
            continue
        
        # Calculate days to expiry
        current_time = datetime.now().timestamp()
        days = (exp_ts - current_time) / (24 * 3600)
        
        if days > 0:  # Only future expirations
            days_to_expiry.append(days)
            atm_ivs.append(curve.atm_iv)
            labels.append(datetime.fromtimestamp(exp_ts).strftime('%d%b%y'))
    
    if not days_to_expiry:
        fig.update_layout(
            title="ATM Volatility Term Structure (no data)",
            xaxis_title="Days to Expiry",
            yaxis_title="ATM Implied Volatility",
        )
        return fig
    
    # Create scatter plot with line
    fig.add_trace(go.Scatter(
        x=days_to_expiry,
        y=atm_ivs,
        mode='lines+markers',
        name='ATM IV',
        text=labels,
        hovertemplate='Days: %{x:.1f}<br>ATM IV: %{y:.4f}<br>Exp: %{text}<extra></extra>',
        line=dict(width=2, color='blue'),
        marker=dict(size=8, color='blue')
    ))
    
    fig.update_layout(
        title="ATM Volatility Term Structure (live)",
        xaxis_title="Days to Expiry",
        yaxis_title="ATM Implied Volatility",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode='closest'
    )
    
    return fig


def build_combined_figure_with_term_structure(
    curves: Dict[int, VolatilityCurve],
    ssr_values: Optional[Dict[int, Optional[float]]] = None
) -> go.Figure:
    """
    Build combined figure with both IV curves and term structure in subplots.
    
    Args:
        curves: Dictionary mapping expiration timestamp to VolatilityCurve
        ssr_values: Dictionary mapping expiration timestamp to SSR estimate
        
    Returns:
        Plotly figure with subplots (2 or 3 depending on SSR availability)
    """
    from plotly.subplots import make_subplots
    
    # Determine number of subplots (3 if SSR values available, 2 otherwise)
    # Check if we have valid SSR values
    has_ssr = False
    if ssr_values is not None and len(ssr_values) > 0:
        valid_ssr = [
            v for v in ssr_values.values() 
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v)
        ]
        has_ssr = len(valid_ssr) > 0
    
    try:
        if has_ssr:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Implied Volatility Curves", "ATM Volatility Term Structure", "SSR (Skew Stickiness Ratio)"),
                vertical_spacing=0.12,
                row_heights=[0.4, 0.3, 0.3]
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Implied Volatility Curves", "ATM Volatility Term Structure"),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5]
            )
    except Exception as e:
        print(f"Error creating subplots: {e}")
        # Fallback to single plot
        fig = go.Figure()
        fig.update_layout(title="Error creating visualization")
        return fig
    
    if not curves:
        # Add a message when no curves are available
        fig.add_annotation(
            text="Waiting for volatility curve data...<br>Please ensure the application is connected and receiving market data.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Top subplot: IV Curves
    # Use Plotly's default color cycle
    import plotly.colors as pc
    colors = pc.qualitative.Plotly
    
    for idx, (exp, curve) in enumerate(sorted(curves.items())):
        if not curve or not curve.points:
            continue
        
        # Plot raw data points
        pts = sorted(curve.points, key=lambda p: p.moneyness)
        x = np.array([p.moneyness for p in pts])
        y = np.array([p.iv for p in pts])
        label = f"{datetime.fromtimestamp(curve.expiration).strftime('%d%b%y')} | ATM {curve.atm_iv:.3f}"
        
        # Get color for this expiration (cycle through colors)
        color = colors[idx % len(colors)]
        
        # Plot SVI-fitted surface if available
        if hasattr(curve, 'svi_params') and curve.svi_params is not None:
            try:
                # Try multiple import paths for robustness
                try:
                    from ..analytics.svi import svi_iv_at_moneyness
                except ImportError:
                    try:
                        from deribit_trading_toolkit.analytics.svi import svi_iv_at_moneyness
                    except ImportError:
                        logger.warning("Could not import svi_iv_at_moneyness, skipping SVI visualization")
                        raise ImportError("SVI module not available")
                
                # Calculate time to expiry
                current_time = datetime.now().timestamp()
                T = max((exp - current_time) / (365 * 24 * 3600), 0.001)
                
                # Generate smooth moneyness range for SVI curve
                # Use range from min to max moneyness with some padding
                if len(x) > 0:
                    min_k = float(min(x))
                    max_k = float(max(x))
                    padding = (max_k - min_k) * 0.2 if max_k > min_k else 0.2
                else:
                    min_k = -0.5
                    max_k = 0.5
                    padding = 0.2
                
                k_range = np.linspace(min_k - padding, max_k + padding, 200)
                
                # Calculate IV using SVI
                iv_svi = svi_iv_at_moneyness(k_range, T, curve.svi_params)
                
                # Ensure iv_svi is a numpy array
                if not isinstance(iv_svi, np.ndarray):
                    iv_svi = np.array(iv_svi)
                
                # Normalize IV if needed (SVI returns decimal, but might need to match display format)
                # Check if raw IVs are in percentage format
                if len(y) > 0 and np.any(y > 1.0):
                    # Raw IVs are in percentage, but SVI returns decimal
                    iv_svi_display = iv_svi * 100.0
                else:
                    iv_svi_display = iv_svi
                
                # Ensure no NaN or inf values
                iv_svi_display = np.nan_to_num(iv_svi_display, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Plot SVI-fitted curve (smooth line) with same color
                fig.add_trace(go.Scatter(
                    x=k_range,
                    y=iv_svi_display,
                    mode='lines',
                    name=label,
                    line=dict(width=2.5, color=color),
                    hovertemplate="logM: %{x:.4f}<br>SVI IV: %{y:.4f}<extra></extra>",
                ), row=1, col=1)
                
                # Plot raw points with same color (markers on top)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name=f"{label} (raw)",
                    marker=dict(size=5, color=color, opacity=0.7, line=dict(width=0.5, color='white')),
                    hovertemplate="logM: %{x:.4f}<br>IV: %{y:.4f}<extra></extra>",
                    showlegend=False,  # Hide from legend to reduce clutter
                ), row=1, col=1)
            except Exception as e:
                # If SVI plotting fails, just show raw points
                logger.debug(f"Failed to plot SVI curve for {exp}: {e}", exc_info=False)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(color=color),
                    hovertemplate="logM: %{x:.4f}<br>IV: %{y:.4f}<extra></extra>",
                ), row=1, col=1)
        else:
            # No SVI params or poor fit - use cubic spline interpolation
            try:
                from scipy.interpolate import CubicSpline
                
                # Generate smooth moneyness range for cubic spline
                if len(x) > 0:
                    min_k = float(min(x))
                    max_k = float(max(x))
                    padding = (max_k - min_k) * 0.2 if max_k > min_k else 0.2
                else:
                    min_k = -0.5
                    max_k = 0.5
                    padding = 0.2
                
                k_range = np.linspace(min_k - padding, max_k + padding, 200)
                
                # Create cubic spline interpolation
                if len(x) >= 3:  # Need at least 3 points for cubic spline
                    # Sort by moneyness
                    sorted_indices = np.argsort(x)
                    x_sorted = np.array(x)[sorted_indices]
                    y_sorted = np.array(y)[sorted_indices]
                    
                    # Create cubic spline
                    spline = CubicSpline(x_sorted, y_sorted, extrapolate=True)
                    iv_spline = spline(k_range)
                    
                    # Ensure no NaN or inf values
                    iv_spline = np.nan_to_num(iv_spline, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Plot cubic spline curve (smooth line) with same color
                    fig.add_trace(go.Scatter(
                        x=k_range,
                        y=iv_spline,
                        mode='lines',
                        name=label,
                        line=dict(width=2.5, color=color),
                        hovertemplate="logM: %{x:.4f}<br>Cubic Spline IV: %{y:.4f}<extra></extra>",
                    ), row=1, col=1)
                    
                    # Plot raw points with same color (markers on top)
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        name=f"{label} (raw)",
                        marker=dict(size=5, color=color, opacity=0.7, line=dict(width=0.5, color='white')),
                        hovertemplate="logM: %{x:.4f}<br>IV: %{y:.4f}<extra></extra>",
                        showlegend=False,  # Hide from legend to reduce clutter
                    ), row=1, col=1)
                else:
                    # Not enough points for spline, just show raw points with line
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(color=color),
                        hovertemplate="logM: %{x:.4f}<br>IV: %{y:.4f}<extra></extra>",
                    ), row=1, col=1)
            except Exception as e:
                logger.debug(f"Failed to create cubic spline for {exp}: {e}", exc_info=False)
                # Fallback to simple line
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(color=color),
                    hovertemplate="logM: %{x:.4f}<br>IV: %{y:.4f}<extra></extra>",
                ), row=1, col=1)
    
    # Bottom subplot: Term Structure
    days_to_expiry = []
    atm_ivs = []
    labels = []
    
    for exp_ts, curve in sorted(curves.items()):
        if not curve or curve.atm_iv <= 0:
            continue
        
        current_time = datetime.now().timestamp()
        days = (exp_ts - current_time) / (24 * 3600)
        
        if days > 0:
            days_to_expiry.append(days)
            atm_ivs.append(curve.atm_iv)
            labels.append(datetime.fromtimestamp(exp_ts).strftime('%d%b%y'))
    
    if days_to_expiry:
        fig.add_trace(go.Scatter(
            x=days_to_expiry,
            y=atm_ivs,
            mode='lines+markers',
            name='ATM IV Term Structure',
            text=labels,
            hovertemplate='Days: %{x:.1f}<br>ATM IV: %{y:.4f}<br>Exp: %{text}<extra></extra>',
            line=dict(width=2, color='red'),
            marker=dict(size=8, color='red'),
            showlegend=False
        ), row=2, col=1)
    
    # Add vertical line at ATM for top plot
    fig.add_vline(x=0.0, line_width=1, line_color="gray", opacity=0.5, row=1, col=1)
    
    # Add SSR plot if available
    if has_ssr:
        days_to_expiry_ssr = []
        ssr_list = []
        labels_ssr = []
        
        for exp_ts in sorted(curves.keys()):
            if exp_ts not in ssr_values:
                continue
            ssr_val = ssr_values[exp_ts]
            if ssr_val is None:
                continue
            
            # Check for NaN values
            if isinstance(ssr_val, float) and math.isnan(ssr_val):
                continue
            
            current_time = datetime.now().timestamp()
            days = (exp_ts - current_time) / (24 * 3600)
            
            if days > 0:
                days_to_expiry_ssr.append(days)
                ssr_list.append(ssr_val)
                labels_ssr.append(datetime.fromtimestamp(exp_ts).strftime('%d%b%y'))
        
        if days_to_expiry_ssr and ssr_list:
            fig.add_trace(go.Scatter(
                x=days_to_expiry_ssr,
                y=ssr_list,
                mode='lines+markers',
                name='SSR',
                text=labels_ssr,
                hovertemplate='Days: %{x:.1f}<br>SSR: %{y:.4f}<br>Exp: %{text}<extra></extra>',
                line=dict(width=2, color='green'),
                marker=dict(size=8, color='green'),
                showlegend=False
            ), row=3, col=1)
            
            # Add horizontal line at SSR = 1 (reference level)
            fig.add_hline(y=1.0, line_width=1, line_dash="dash", 
                         line_color="gray", opacity=0.5, row=3, col=1)
        else:
            # Show empty SSR plot with message
            fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='markers',
                name='SSR (waiting for data)',
                showlegend=False
            ), row=3, col=1)
        
        # Update axes for SSR plot
        fig.update_xaxes(title_text="Days to Expiry", row=3, col=1)
        fig.update_yaxes(title_text="SSR", row=3, col=1)
    
    # Update axes
    fig.update_xaxes(title_text="log-moneyness (ln(K/F))", row=1, col=1)
    fig.update_yaxes(title_text="Implied Volatility", row=1, col=1)
    fig.update_xaxes(title_text="Days to Expiry", row=2, col=1)
    fig.update_yaxes(title_text="ATM Implied Volatility", row=2, col=1)
    
    height = 1200 if has_ssr else 900
    fig.update_layout(
        title="Deribit IV Curves, ATM Volatility Term Structure & SSR (live)",
        template="plotly_white",
        height=height,
        showlegend=True,
    )
    
    return fig


#!/usr/bin/env python3
"""
Real-time volatility curve visualization using Plotly Dash.

This module exposes:
- DataStore: a thread-safe store for latest curves
- build_figure_from_curves: render curves into a Plotly figure
- create_app: create a Dash application wired to the datastore
"""

from typing import Dict, Optional
from datetime import datetime
import threading

import plotly.graph_objs as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from deribit_trading_toolkit import VolatilityCurve


class DataStore:
    """Thread-safe in-memory store for latest volatility curves and SSR values."""
    def __init__(self):
        self._lock = threading.Lock()
        self._curves: Dict[int, VolatilityCurve] = {}
        self._ssr_values: Dict[int, Optional[float]] = {}

    def set_curves(self, curves: Dict[int, VolatilityCurve]) -> None:
        with self._lock:
            self._curves = dict(curves)

    def get_curves(self) -> Dict[int, VolatilityCurve]:
        with self._lock:
            return dict(self._curves)
    
    def set_ssr_values(self, ssr_values: Dict[int, Optional[float]]) -> None:
        """Set SSR values for expiries."""
        with self._lock:
            self._ssr_values = dict(ssr_values)
    
    def get_ssr_values(self) -> Dict[int, Optional[float]]:
        """Get SSR values for expiries."""
        with self._lock:
            return dict(self._ssr_values)


def build_combined_figure(curves: Dict[int, VolatilityCurve]) -> go.Figure:
    """Build a figure from volatility curves (single plot)"""
    return build_figure_from_curves(curves)


def build_figure_from_curves(curves: Dict[int, VolatilityCurve]) -> go.Figure:
    fig = go.Figure()

    if not curves:
        fig.update_layout(
            title="Volatility Curves (waiting for data)",
            xaxis_title="log-moneyness (ln(K/F))",
            yaxis_title="Implied Volatility",
        )
        return fig

    for exp, curve in sorted(curves.items()):
        if not curve or not curve.points:
            continue
        pts = sorted(curve.points, key=lambda p: p.moneyness)
        x = np.array([p.moneyness for p in pts])
        y = np.array([p.iv for p in pts])
        label = f"{datetime.fromtimestamp(curve.expiration).strftime('%d%b%y')} | ATM {curve.atm_iv:.3f}"
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=label,
            hovertemplate="logM: %{x:.4f}<br>IV: %{y:.4f}<extra></extra>",
        ))

    fig.add_vline(x=0.0, line_width=1, line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Volatility Curves (live)",
        xaxis_title="log-moneyness (ln(K/F))",
        yaxis_title="Implied Volatility",
        legend_title="Expiration | ATM",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def create_app(datastore: DataStore, refresh_ms: int = 1000) -> dash.Dash:
    # Create a completely fresh app instance with unique name to avoid callback conflicts
    import uuid
    app_name = f"deribit_app_{uuid.uuid4().hex[:8]}"
    app = dash.Dash(app_name, suppress_callback_exceptions=True)
    
    # Clear any existing callbacks
    app.callback_map.clear()
    
    app.layout = html.Div(
        children=[
            html.H3("Deribit IV Curves (Real-time)"),
            dcc.Graph(id="combined-graph"),
            dcc.Interval(id="update-interval", interval=refresh_ms, n_intervals=0),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "10px"}
    )

    @app.callback(
        dash.dependencies.Output("combined-graph", "figure"), 
        [dash.dependencies.Input("update-interval", "n_intervals")]
    )
    def update_graph(n_intervals):
        try:
            curves = datastore.get_curves()
            return build_combined_figure(curves)
        except Exception as e:
            # Return empty figure if there's an error
            print(f"Error in callback: {e}")
            fig = go.Figure()
            fig.update_layout(
                title="Error loading data - check logs",
                xaxis_title="log-moneyness (ln(K/F))",
                yaxis_title="Implied Volatility",
            )
            return fig

    return app
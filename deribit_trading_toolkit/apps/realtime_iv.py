"""
Real-time Implied Volatility Visualization Application

Provides real-time IV curve and ATM volatility term structure visualization
using automatic expiration selection based on trading rules.
"""

import asyncio
import logging
import math
import re
import threading
import subprocess
import os
import signal
from datetime import datetime
from typing import Dict, List, Optional

from ..core.client import DeribitClient, DeribitAuth
from ..utils.config import ConfigManager
from ..analytics.volatility import VolatilityAnalyzer
from ..analytics.ssr import SSRCalculator
from ..analytics.min_var_delta import MinimumVarianceDeltaCalculator
from ..models.market_data import MarketData, VolatilityCurve
from ..visualization.vol_curve import DataStore
from ..utils.expiration_selector import ExpirationSelector
from ..visualization.term_structure import build_combined_figure_with_term_structure

logger = logging.getLogger(__name__)


class InstrumentParser:
    """Utility class for parsing Deribit instrument names"""
    
    # Regex patterns for parsing instrument names
    # Supports single-digit days (e.g., "4NOV25") and double-digit days (e.g., "26DEC25")
    EXP_RE = re.compile(r'(\d{1,2}[A-Z]{3}\d{2})', re.IGNORECASE)
    OPT_RE = re.compile(r'BTC[_-](\d{1,2}[A-Z]{3}\d{2})[_-](\d+)[_-]([CP])', re.IGNORECASE)
    
    @classmethod
    def parse_english_future(cls, s: str) -> Optional[str]:
        """
        Parse English-style future name.
        
        Examples:
            - "26DEC25(futures)" → "BTC-26DEC25"
            - "26DEC25" → "BTC-26DEC25"
        
        Args:
            s: English-style future name
            
        Returns:
            Deribit-style future name or None if parsing fails
        """
        m = cls.EXP_RE.search(s.replace(" ", ""))
        if not m:
            return None
        exp = m.group(1).upper()
        return f"BTC-{exp}"
    
    @classmethod
    def parse_english_option(cls, s: str) -> Optional[str]:
        """
        Parse English-style option name.
        
        Examples:
            - "BTC_26DEC25_105000_C" → "BTC-26DEC25-105000-C"
            - "BTC-4NOV25-114000-C" → "BTC-4NOV25-114000-C"
        
        Args:
            s: English-style option name
            
        Returns:
            Deribit-style option name or None if parsing fails
        """
        m = cls.OPT_RE.search(s.replace(" ", ""))
        if not m:
            return None
        exp, strike, cp = m.groups()
        return f"BTC-{exp.upper()}-{int(strike)}-{cp.upper()}"
    
    @classmethod
    def exp_to_epoch(cls, exp_str: str) -> Optional[int]:
        """
        Convert expiration string to epoch timestamp.
        
        Args:
            exp_str: Expiration string (e.g., "26DEC25", "4NOV25")
            
        Returns:
            Epoch timestamp or None if parsing fails
        """
        try:
            return int(datetime.strptime(exp_str.upper(), "%d%b%y").timestamp())
        except Exception:
            return None
    
    @classmethod
    def inst_exp_epoch(cls, instrument_name: str) -> Optional[int]:
        """
        Extract expiration epoch from instrument name.
        
        Args:
            instrument_name: Deribit instrument name (e.g., "BTC-26DEC25-105000-C")
            
        Returns:
            Epoch timestamp or None if parsing fails
        """
        m = cls.EXP_RE.search(instrument_name)
        if not m:
            return None
        return cls.exp_to_epoch(m.group(1))


class DashServer:
    """Manages Dash server for real-time visualization"""
    
    def __init__(self, datastore: DataStore, refresh_seconds: int = 2):
        """
        Initialize Dash server.
        
        Args:
            datastore: DataStore instance for sharing data
            refresh_seconds: Refresh interval in seconds
        """
        self.datastore = datastore
        self.refresh_seconds = refresh_seconds
        self._dash_thread: Optional[threading.Thread] = None
    
    def start(self, port: int = 8050, host: str = "127.0.0.1"):
        """
        Start Dash server in background thread.
        
        Args:
            port: Port number for the server
            host: Host address for the server
        """
        app = self._create_app()
        
        def _free_port(p: int) -> int:
            """Kill any process listening on the given port. Returns number of PIDs killed."""
            try:
                out = subprocess.check_output(["lsof", "-ti", f":{p}"])
                pids = [int(pid) for pid in out.decode().strip().splitlines() if pid.strip()]
            except subprocess.CalledProcessError:
                pids = []
            
            killed = 0
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except Exception:
                    pass
            return killed
        
        def run():
            try:
                killed = _free_port(port)
                if killed:
                    logger.info(f"Freed port {port} by killing {killed} process(es)")
                
                logger.info(f"Starting Dash server on http://{host}:{port}")
                logger.info("Dash server is starting in background thread...")
                
                # Test that the app is valid before starting
                try:
                    # Verify app was created successfully
                    if app is None:
                        raise RuntimeError("Dash app is None - creation failed")
                    logger.info("Dash app created successfully, starting server...")
                except Exception as test_e:
                    logger.error(f"Dash app validation failed: {test_e}", exc_info=True)
                    return
                
                # Start the server (Dash 2.x uses run())
                # Note: app.run() is blocking, so this thread will run until stopped
                logger.info(f"Calling app.run() on {host}:{port}...")
                app.run(host=host, port=port, debug=False, use_reloader=False)
                logger.info("Dash server stopped")
                    
            except OSError as e:
                error_msg = str(e).lower()
                if "address already in use" in error_msg or "address is already in use" in error_msg:
                    logger.warning(f"Port {port} is already in use, trying alternative port...")
                    # Try alternative port
                    alt_port = port + 1
                    try:
                        _free_port(alt_port)
                        logger.info(f"Starting Dash server on alternative port {alt_port}")
                        app.run(host=host, port=alt_port, debug=False, use_reloader=False)
                    except Exception as e2:
                        logger.error(f"Failed to start Dash server on alternative port {alt_port}: {e2}", exc_info=True)
                else:
                    logger.error(f"Failed to start Dash server (OSError): {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error starting Dash server: {e}", exc_info=True)
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Use non-daemon thread so it doesn't get killed when main thread exits
        # But we'll make it daemon=False only if we want it to keep running
        # Actually, for Dash, we want it to be non-daemon so it keeps running
        self._dash_thread = threading.Thread(target=run, daemon=False, name="DashServer")
        self._dash_thread.start()
        logger.info(f"Dash server thread started (thread ID: {self._dash_thread.ident})")
        
        # Wait and verify server is actually running
        import time
        import socket
        
        def check_port_open(host, port, timeout=10):
            """Check if a port is actually listening"""
            for _ in range(timeout):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        return True
                except Exception:
                    pass
                time.sleep(1)
            return False
        
        # Wait for server to start
        logger.info("Waiting for Dash server to start...")
        
        # Check multiple times with increasing delays
        max_wait = 10
        for i in range(max_wait):
            time.sleep(1)
            
            # Verify the thread is still alive
            if not self._dash_thread.is_alive():
                logger.error(f"Dash server thread died after {i+1} seconds. Check logs for errors.")
                logger.error("This might be due to:")
                logger.error("  1. Port already in use")
                logger.error("  2. Missing dependencies (dash, plotly)")
                logger.error("  3. Error in app creation")
                logger.error("  4. Import error in callback or layout")
                return
            
            # Check if port is actually open
            if check_port_open(host, port, timeout=1):
                url = f"http://{host}:{port}"
                logger.info(f"✓ Dash server is running! Access dashboard at {url}")
                
                # Try to open browser automatically
                try:
                    import webbrowser
                    time.sleep(0.5)  # Brief wait
                    webbrowser.open(url)
                    logger.info(f"Opened browser to {url}")
                except Exception as e:
                    logger.debug(f"Could not open browser automatically: {e}")
                    logger.info(f"Please manually open your browser and navigate to {url}")
                return
        
        # If we get here, server didn't start in time
        logger.warning(f"Dash server thread is alive but port {port} did not become available after {max_wait} seconds")
        logger.warning(f"The server may still be starting. Try accessing http://{host}:{port} in a few more seconds")
        logger.warning("If the problem persists:")
        logger.warning("  1. Check if dash and plotly are installed: pip install dash plotly")
        logger.warning("  2. Check for errors in the logs above")
        logger.warning("  3. Try manually accessing http://{}:{}".format(host, port))
    
    def _create_app(self):
        """Create Dash application with IV curves and term structure"""
        try:
            import uuid
            import dash
            from dash import dcc, html
            import plotly.graph_objs as go
        except ImportError as e:
            logger.error(f"Missing required dependencies: {e}")
            logger.error("Please install: pip install dash plotly")
            raise
        
        try:
            app_name = f"deribit_app_{uuid.uuid4().hex[:8]}"
            app = dash.Dash(app_name, suppress_callback_exceptions=True)
            
            # Clear any existing callbacks
            app.callback_map.clear()
        except Exception as e:
            logger.error(f"Failed to create Dash app: {e}", exc_info=True)
            raise
        
        try:
            app.layout = html.Div(
                children=[
                    html.H3("Deribit IV Curves & ATM Volatility Term Structure (Real-time)"),
                    dcc.Graph(id="combined-graph"),
                    dcc.Interval(
                        id="update-interval",
                        interval=int(self.refresh_seconds * 1000),
                        n_intervals=0
                    ),
                ],
                style={"maxWidth": "1400px", "margin": "0 auto", "padding": "10px"}
            )
            logger.debug("Dash app layout created successfully")
        except Exception as e:
            logger.error(f"Failed to create Dash app layout: {e}", exc_info=True)
            raise
        
        try:
            @app.callback(
                dash.dependencies.Output("combined-graph", "figure"),
                [dash.dependencies.Input("update-interval", "n_intervals")]
            )
            def update_graph(n_intervals):
                try:
                    curves = self.datastore.get_curves()
                    ssr_values = self.datastore.get_ssr_values()
                    
                    # Debug: Log what we're getting
                    if n_intervals % 10 == 0:  # Log every 10 updates
                        valid_ssr = {k: v for k, v in (ssr_values.items() if ssr_values else {}) if v is not None}
                        logger.debug(f"Dashboard update #{n_intervals}: {len(curves)} curves, {len(ssr_values or {})} SSR values ({len(valid_ssr)} valid)")
                    
                    # Build figure with error handling
                    try:
                        return build_combined_figure_with_term_structure(curves, ssr_values)
                    except Exception as build_error:
                        logger.error(f"Error building figure: {build_error}", exc_info=True)
                        # Return a simple error figure
                        fig = go.Figure()
                        fig.add_annotation(
                            text=f"Error building visualization: {str(build_error)}<br>Check logs for details.",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=14)
                        )
                        fig.update_layout(
                            title="Error loading data - check logs",
                            xaxis_title="log-moneyness (ln(K/F))",
                            yaxis_title="Implied Volatility",
                        )
                        return fig
                except Exception as e:
                    logger.error(f"Error in callback: {e}", exc_info=True)
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Error: {str(e)}<br>Check logs for details.",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=14)
                    )
                    fig.update_layout(
                        title="Error loading data - check logs",
                        xaxis_title="log-moneyness (ln(K/F))",
                        yaxis_title="Implied Volatility",
                    )
                    return fig
            
            logger.debug("Dash app callback registered successfully")
        except Exception as e:
            logger.error(f"Failed to register Dash app callback: {e}", exc_info=True)
            raise
        
        logger.info("Dash app created and configured successfully")
        return app


class RealTimeIVApp:
    """
    Real-time Implied Volatility Visualization Application
    
    Features:
    - Automatic expiration selection based on trading rules
    - Real-time IV curve visualization
    - ATM volatility term structure
    - Web-based dashboard using Dash
    - Support for daily, weekly, monthly, and quarterly options
    """
    
    def __init__(
        self,
        refresh_seconds: int = 2,
        futures_refresh_seconds: int = 15,
        use_auto_expirations: bool = True,
        reference_date: Optional[datetime] = None
    ):
        """
        Initialize real-time IV app with automatic expiration selection.
        
        Args:
            refresh_seconds: Refresh interval for IV curves (seconds)
            futures_refresh_seconds: Refresh interval for futures prices (seconds)
            use_auto_expirations: If True, automatically select expirations based on rules
            reference_date: Reference date for expiration selection (defaults to now)
        """
        self.refresh_seconds = refresh_seconds
        self.futures_refresh_seconds = futures_refresh_seconds
        self.parser = InstrumentParser()
        
        # Initialize expiration selection
        if use_auto_expirations:
            selector = ExpirationSelector(reference_date or datetime.now())
            summary = selector.get_summary()
            
            target_expirations_english = summary['all_expirations']
            futures_map_english = selector.get_futures_map()
            daily_expirations_english = selector.get_daily_expirations_english()
            
            logger.info(f"Auto-selected {summary['total_count']} expirations:")
            logger.info(f"  Daily: {summary['daily_count']} (using perpetual: {daily_expirations_english})")
            logger.info(f"  Weekly: {summary['weekly_count']}")
            logger.info(f"  Monthly: {summary['monthly_count']}")
            logger.info(f"  Quarterly: {summary['quarterly_count']}")
            logger.info(f"  Expirations: {', '.join(target_expirations_english)}")
            
            # Log epoch timestamps for debugging
            for exp_str in target_expirations_english[:5]:
                exp_epoch = self.parser.exp_to_epoch(exp_str)
                if exp_epoch:
                    logger.info(f"    {exp_str} -> epoch: {exp_epoch}")
        else:
            target_expirations_english = []
            futures_map_english = {}
            daily_expirations_english = []
        
        # Map expirations to epoch and Deribit future names (or None for perpetual)
        self.target_exp_epochs: List[int] = []
        daily_exp_epochs: List[int] = []
        
        for e in target_expirations_english:
            ts = self.parser.exp_to_epoch(e)
            if ts:
                self.target_exp_epochs.append(ts)
                if e in daily_expirations_english:
                    daily_exp_epochs.append(ts)
        
        self.futures_by_exp: Dict[int, Optional[str]] = {}
        for e, f in futures_map_english.items():
            ts = self.parser.exp_to_epoch(e)
            if not ts:
                continue
            fut_name = self.parser.parse_english_future(f)
            self.futures_by_exp[ts] = fut_name
        
        # Daily expirations explicitly don't have futures (will use perpetual)
        self.daily_exp_epochs = set(daily_exp_epochs)
        
        # Initialize core components
        self.client: Optional[DeribitClient] = None
        self.analyzer = VolatilityAnalyzer()
        self.ssr_calculator = SSRCalculator()
        self.min_var_delta_calc = MinimumVarianceDeltaCalculator()
        self.option_ticks: Dict[str, List[MarketData]] = {}
        self.under_by_exp: Dict[int, float] = {}
        self.current_F: float = 0.0  # Current forward/spot price for SSR
        self._listen_task: Optional[asyncio.Task] = None
        
        # Visualization components
        self.datastore = DataStore()
        self.dash_server: Optional[DashServer] = None
    
    async def start(self):
        """Start the real-time IV application"""
        cfg = ConfigManager().get_config()
        auth = DeribitAuth(cfg.deribit.client_id, cfg.deribit.private_key_path)
        self.client = DeribitClient(cfg.deribit, auth)
        await self.client.connect()
        
        # Initialize underlying prices - set daily options to perpetual immediately
        await self._initialize_underlying_prices()
        
        # Subscribe to options mark price stream
        await self.client.subscribe_to_mark_price()
        
        # Register handler
        self.client.register_message_handler(
            r"markprice\.options\.btc_usd",
            self._on_options_markprice
        )
        self._listen_task = asyncio.create_task(self.client.listen_for_messages())
        
        # Kick off futures refresh loop (REST)
        asyncio.create_task(self._refresh_futures_loop())
        
        # Start Dash server in background
        self.dash_server = DashServer(self.datastore, self.refresh_seconds)
        self.dash_server.start()
        
        # Main loop: analyze and visualize
        await self._run_loop()
    
    async def stop(self):
        """Stop the real-time IV application"""
        if self._listen_task:
            self._listen_task.cancel()
        if self.client:
            await self.client.disconnect()
    
    async def _initialize_underlying_prices(self):
        """Initialize underlying prices, especially for daily options"""
        try:
            # Get perpetual price for daily options
            perp_md = await self.client.get_ticker("BTC-PERPETUAL")
            perp_price = perp_md.mark_price or perp_md.mid_price or None
            
            if perp_price:
                # Set perpetual price for all daily expirations immediately
                for exp in self.daily_exp_epochs:
                    self.under_by_exp[exp] = perp_price
                    logger.info(f"Initialized daily option {exp} with perpetual price: {perp_price}")
                
                # Initialize current_F for SSR (use perpetual as proxy for forward)
                self.current_F = perp_price
        except Exception as e:
            logger.warning(f"Error initializing underlying prices: {e}")
    
    async def _refresh_futures_loop(self):
        """
        Periodically refresh futures mark price via REST.
        
        Daily options use BTC-PERPETUAL (no futures exist).
        Other options try to use matched futures, fallback to BTC-PERPETUAL.
        """
        assert self.client
        while True:
            try:
                # Get perpetual price once
                perp_md = await self.client.get_ticker("BTC-PERPETUAL")
                perp_price = perp_md.mark_price or perp_md.mid_price or None
                
                # Update current_F for SSR calculation
                if perp_price:
                    self.current_F = perp_price
                
                # Refresh each target expiration's underlying
                for exp in self.target_exp_epochs:
                    # Daily options always use perpetual (no futures exist)
                    if exp in self.daily_exp_epochs:
                        if perp_price:
                            self.under_by_exp[exp] = perp_price
                        continue
                    
                    # For non-daily options, try to get future price
                    fut_name = self.futures_by_exp.get(exp)
                    if fut_name:
                        try:
                            md = await self.client.get_ticker(fut_name)
                            price = md.mark_price
                            if price:
                                self.under_by_exp[exp] = price
                                logger.debug(
                                    f"exp: {exp} using future {fut_name} price: "
                                    f"{self.under_by_exp[exp]}"
                                )
                                continue
                        except Exception as e:
                            logger.debug(
                                f"Failed to get future {fut_name} for exp {exp}: {e}"
                            )
                    
                    # Fallback to perpetual for non-daily options if future not available
                    if perp_price:
                        self.under_by_exp[exp] = perp_price
                        logger.debug(f"exp: {exp} using perpetual price: {perp_price}")
            
            except Exception as e:
                logger.warning(f"Futures refresh error: {e}")
            
            await asyncio.sleep(self.futures_refresh_seconds)
    
    async def _on_options_markprice(self, channel: str, msg: dict):
        """
        Handle options mark price updates.
        
        Args:
            channel: WebSocket channel name
            msg: Message containing mark price data
        """
        try:
            arr = msg.get('params', {}).get('data', [])
            for row in arr:
                name = row.get('instrument_name')
                if not name:
                    continue
                
                exp = self.parser.inst_exp_epoch(name)
                if exp is None:
                    logger.debug(f"Could not parse expiration from instrument: {name}")
                    continue
                
                if exp not in self.target_exp_epochs:
                    logger.debug(
                        f"Expiration {exp} not in target expirations for instrument: {name}"
                    )
                    continue
                
                md = MarketData(
                    instrument_name=name,
                    timestamp=row.get('timestamp', 0),
                    mark_price=row.get('mark_price'),
                    mark_iv=row.get('iv') or row.get('mark_iv'),  # Try both fields
                )
                
                self.option_ticks.setdefault(name, []).append(md)
        except Exception as e:
            logger.debug(f"parse markprice error: {e}")
    
    def _build_curves(self) -> Dict[int, VolatilityCurve]:
        """
        Build volatility curves for selected expirations.
        
        Returns:
            Dictionary mapping expiration epochs to VolatilityCurve instances
        """
        return self.analyzer.build_curves_for_expirations(
            options_data_by_instrument=self.option_ticks,
            expirations=self.target_exp_epochs,
            underlying_by_expiration=self.under_by_exp
        )
    
    async def _run_loop(self):
        """Main analysis and visualization loop"""
        while True:
            try:
                curves = self._build_curves()
                
                # Debug: log missing underlying prices
                for exp in self.target_exp_epochs:
                    if exp not in self.under_by_exp or self.under_by_exp.get(exp, 0) <= 0:
                        logger.debug(
                            f"Missing underlying price for expiration: {exp} "
                            f"(in target: {exp in self.target_exp_epochs})"
                        )
                
                # Debug: log options data collected
                if self.option_ticks:
                    total_ticks = sum(len(ticks) for ticks in self.option_ticks.values())
                    logger.debug(
                        f"Collected {len(self.option_ticks)} instruments with "
                        f"{total_ticks} total ticks"
                    )
                
                if curves and self.current_F > 0:
                    # Publish to datastore for Dash
                    self.datastore.set_curves(curves)
                    
                    # Build term structure for SSR calculation
                    term_structure = []
                    current_time = datetime.now().timestamp()
                    current_time_ms = current_time * 1000.0
                    
                    for exp, curve in curves.items():
                        if curve and curve.atm_iv > 0:
                            # Calculate time to expiry
                            tau = self.ssr_calculator.tau_years(int(exp * 1000), current_time_ms)
                            if tau <= 0:
                                continue
                            
                            # Calculate ATM slope in IV space
                            slope_iv = self.analyzer.calculate_atm_slope(curve)
                            
                            # Convert IV slope to √w slope: ∂√w/∂k = √τ * ∂IV/∂k
                            slope_sqrtw = slope_iv * math.sqrt(tau)
                            
                            term_structure.append({
                                'expiry_ms': int(exp * 1000),  # Convert to milliseconds for SSR calculator
                                'atm_iv1y': curve.atm_iv,
                                'slope_atm': slope_sqrtw
                            })
                    
                    # Process SSR tick and get updated SSR estimates
                    if term_structure:
                        self.ssr_calculator.on_tick(
                            t=current_time,
                            F_t=self.current_F,
                            term_structure=term_structure
                        )
                        
                        # Also add data points for minimum variance delta calculation
                        # This collects intraday series for SSR(τ) calculation
                        for item in term_structure:
                            exp_ms = item['expiry_ms']
                            tau = self.ssr_calculator.tau_years(exp_ms, current_time_ms)
                            if tau <= 0:
                                continue
                            
                            # Get ATM √w and slope
                            atm_iv1y = item.get('atm_iv1y', 0.0)
                            slope_atm = item.get('slope_atm', 0.0)
                            sqrtw_atm = self.ssr_calculator.sqrtw_from_iv(atm_iv1y, tau)
                            
                            # Add intraday point for minimum variance delta calculation
                            self.min_var_delta_calc.add_intraday_point(
                                expiry_ms=exp_ms,
                                timestamp=current_time,
                                F_t=self.current_F,
                                sqrtw_atm=sqrtw_atm,
                                slope_atm=slope_atm
                            )
                    
                    # Always get current SSR estimates and store (even if empty/initializing)
                    ssr_values_ms = self.ssr_calculator.get_current_SSR()
                    
                    # Convert SSR keys from milliseconds to seconds to match curves keys
                    ssr_values = {exp_ms // 1000: ssr for exp_ms, ssr in ssr_values_ms.items()}
                    
                    # Store SSR values in datastore for dashboard (always, even if empty)
                    self.datastore.set_ssr_values(ssr_values)
                    
                    # Log SSR values periodically
                    if ssr_values:
                        valid_count = sum(1 for v in ssr_values.values() if v is not None)
                        logger.info(f"SSR values: {valid_count}/{len(ssr_values)} valid for {len(ssr_values)} expiries")
                        for exp_ms, ssr in ssr_values.items():
                            if ssr is not None:
                                logger.info(f"SSR for exp {exp_ms}: {ssr:.4f}")
                    else:
                        logger.debug("SSR values empty - waiting for initialization")
                    
                    # Quick stats
                    for exp, curve in curves.items():
                        slope = self.analyzer.calculate_atm_slope(curve)
                        logger.debug(
                            f"exp={exp} ATM_IV={curve.atm_iv:.4f} "
                            f"slope={slope:.4f} points={len(curve.points)}"
                        )
                else:
                    # Even if curves aren't ready, try to get existing SSR values
                    ssr_values_ms = self.ssr_calculator.get_current_SSR()
                    ssr_values = {exp_ms // 1000: ssr for exp_ms, ssr in ssr_values_ms.items()}
                    if ssr_values:
                        self.datastore.set_ssr_values(ssr_values)
                    
                    logger.debug(
                        f"No curves built. Underlying prices: {len(self.under_by_exp)}, "
                        f"Option ticks: {len(self.option_ticks)}, "
                        f"Current_F: {self.current_F}"
                    )
            
            except Exception as e:
                logger.error(f"Analysis error: {e}", exc_info=True)
            
            await asyncio.sleep(self.refresh_seconds)


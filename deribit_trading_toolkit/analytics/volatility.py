"""
Volatility Analytics Module

Comprehensive volatility analysis tools including:
- Implied volatility curve construction
- Volatility surface analysis
- Term structure analysis
- Skewness calculations
"""

import numpy as np
import pandas as pd
import asyncio
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar
from typing import List, Dict, Optional, Tuple, Union, Iterable
from dataclasses import dataclass
import logging
import re
from datetime import datetime

from ..core.client import DeribitClient
from ..models.market_data import MarketData, VolatilityCurve, VolatilityPoint, OptionType
from .svi import (
    calibrate_svi_from_iv, svi_iv_at_moneyness, svi_slope_at_moneyness,
    svi_curvature_at_moneyness, SVIParams
)

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """
    Comprehensive volatility analysis toolkit
    
    Features:
    - IV curve construction and smoothing
    - ATM slope calculation (skewness proxy)
    - Volatility surface interpolation
    - Term structure analysis
    - Skewness analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.05, use_svi: bool = True):
        self.risk_free_rate = risk_free_rate
        self.interpolation_method = 'svi' if use_svi else 'cubic'
        self.use_svi = use_svi
        self.collected_option_mark = {}

    async def collect_option_mark(self, client: DeribitClient, duration: int = 30) -> Dict[str, List[MarketData]]:
        """Collect mark price data for volatility analysis"""
        logger.info(f"📊 Collecting mark price data for {duration} seconds...")
        
        # Subscribe to mark price updates
        await client.subscribe_to_mark_price()
        
        # Register handler and listen
        client.register_message_handler(r"markprice\.options\.btc_usd", self._price_handler)
        listen_task = asyncio.create_task(client.listen_for_messages())
        await asyncio.sleep(duration)
        listen_task.cancel()
        
        logger.info(f"✅ Collected mark price data for {len(self.collected_option_mark)} instruments")

    def build_volatility_curve(self, options_data: List[MarketData], 
                             underlying_price: float,
                             expiration: int) -> Optional[VolatilityCurve]:
        """
        Build volatility curve from options data
        
        Args:
            options_data: List of market data for options
            underlying_price: Current underlying price
            expiration: Expiration timestamp
            
        Returns:
            VolatilityCurve object or None if insufficient data
        """
        try:
            # Filter and process data
            valid_options = self._filter_valid_options(options_data, expiration)
            if len(valid_options) < 2:
                logger.warning(f"Insufficient data for expiration {expiration}: {len(valid_options)} options (need at least 2)")
                return None
            
            # Deduplicate by strike + option_type using the latest tick
            latest_by_key = {}  # key: (strike, option_type) -> MarketData
            for opt in valid_options:
                strike = self._extract_strike(opt.instrument_name)
                if strike is None:
                    continue
                opt_type = self._extract_option_type(opt.instrument_name)
                key = (strike, opt_type)
                prev = latest_by_key.get(key)
                if (prev is None) or (opt.timestamp and opt.timestamp > (prev.timestamp or 0)):
                    latest_by_key[key] = opt

            # Build points from latest-only data
            points = []
            for opt in latest_by_key.values():
                strike = self._extract_strike(opt.instrument_name)
                if strike is None:
                    continue
                moneyness = float(np.log(max(1e-12, strike) / max(1e-12, underlying_price)))
                iv = opt.mark_iv or 0.0
                if iv > 0:
                    points.append(VolatilityPoint(
                        strike=strike,
                        iv=float(iv),
                        moneyness=moneyness,
                        option_type=self._extract_option_type(opt.instrument_name),
                        expiration=expiration,
                        timestamp=opt.timestamp
                    ))

            if len(points) < 2:
                logger.debug(f"Insufficient points for expiration {expiration}: {len(points)} points (need at least 2)")
                return None
            
            # Calculate time to expiry
            T = self._calculate_time_to_expiry(expiration)
            if T <= 0:
                logger.debug(f"Expiration {expiration} is in the past")
                return None
            
            # Calculate ATM metrics using SVI or cubic spline
            atm_strike = underlying_price
            svi_params = None
            
            if self.use_svi:
                # Use SVI parameterization
                try:
                    # Extract strikes and IVs
                    strikes = np.array([p.strike for p in points])
                    ivs = np.array([p.iv for p in points])
                    
                    # Normalize IVs if stored as percentage
                    if np.any(ivs > 1.0):
                        ivs = ivs / 100.0
                    
                    # Calculate weights: emphasize ATM options and reduce weight for outliers
                    k_values = np.log(strikes / underlying_price)
                    # Weight by inverse distance from ATM (k=0), with minimum weight
                    weights = 1.0 / (1.0 + np.abs(k_values) * 2.0)
                    # Also weight by IV quality (lower weight for very high/low IVs)
                    iv_median = np.median(ivs)
                    iv_std = np.std(ivs)
                    iv_weights = np.exp(-0.5 * ((ivs - iv_median) / (iv_std + 0.01)) ** 2)
                    weights = weights * iv_weights
                    weights = weights / np.max(weights)  # Normalize to max 1.0
                    
                    # Fit SVI parameters with weights
                    svi_params = calibrate_svi_from_iv(strikes, ivs, T, underlying_price, weights=weights)
                    
                    # Evaluate SVI fit quality
                    fit_quality = self._evaluate_svi_fit_quality(strikes, ivs, T, underlying_price, svi_params)
                    
                    # If fit quality is poor, fall back to cubic spline
                    if fit_quality['is_good']:
                        # Calculate ATM metrics using SVI
                        atm_moneyness = 0.0  # ATM
                        atm_iv = float(svi_iv_at_moneyness(np.array([atm_moneyness]), T, svi_params)[0])
                        slope = float(svi_slope_at_moneyness(atm_moneyness, T, svi_params))
                        curvature = float(svi_curvature_at_moneyness(atm_moneyness, T, svi_params))
                    else:
                        # Poor fit quality - use cubic spline instead
                        logger.debug(f"SVI fit quality poor (R²={fit_quality['r_squared']:.3f}, RMSE={fit_quality['rmse']:.4f}), using cubic spline")
                        svi_params = None  # Don't store poor SVI params
                        atm_iv = self._interpolate_atm_iv(points, atm_strike)
                        slope = self._calculate_atm_slope(points)
                        curvature = self._calculate_curvature(points)
                    
                except Exception as e:
                    logger.warning(f"SVI fitting failed, falling back to cubic spline: {e}")
                    svi_params = None
                    # Fallback to cubic spline
                    atm_iv = self._interpolate_atm_iv(points, atm_strike)
                    slope = self._calculate_atm_slope(points)
                    curvature = self._calculate_curvature(points)
            else:
                # Use cubic spline (original method)
                atm_iv = self._interpolate_atm_iv(points, atm_strike)
                slope = self._calculate_atm_slope(points)
                curvature = self._calculate_curvature(points)
            
            curve = VolatilityCurve(
                expiration=expiration,
                points=points,
                atm_strike=atm_strike,
                atm_iv=atm_iv,
                slope=slope,
                curvature=curvature,
                timestamp=max(p.timestamp for p in points)
            )
            
            # Store SVI params as attribute (not in dataclass, but accessible)
            if svi_params:
                curve.svi_params = svi_params
            
            return curve
            
        except Exception as e:
            logger.error(f"Error building volatility curve: {e}")
            return None

    def build_curves_for_expirations(
        self,
        options_data_by_instrument: Dict[str, List[MarketData]],
        expirations: List[int],
        underlying_by_expiration: Dict[int, float]) -> Dict[int, VolatilityCurve]:
        """
        Build curves only for provided expirations using per-expiration underlying.
        - options_data_by_instrument: instrument -> List[MarketData]
        - expirations: target expirations (epoch seconds)
        - underlying_by_expiration: expiration -> futures mark price (matched expiry)
        """
        # Flatten data per expiration
        grouped: Dict[int, List[MarketData]] = {}
        for instrument, series in options_data_by_instrument.items():
            if not series:
                continue
            exp = self._extract_expiration(instrument)
            if exp in expirations:
                grouped.setdefault(exp, []).extend(series)

        curves: Dict[int, VolatilityCurve] = {}
        for exp in expirations:
            data_list = grouped.get(exp, [])
            if not data_list:
                logger.debug(f"No data for expiration {exp}")
                continue
            under = underlying_by_expiration.get(exp)
            if not under or under <= 0:
                # Skip if no matched futures mark price
                logger.debug(f"No underlying price for expiration {exp} (has {len(data_list)} data points)")
                continue
            curve = self.build_volatility_curve(data_list, under, exp)
            if curve:
                curves[exp] = curve
            else:
                logger.debug(f"Failed to build curve for expiration {exp} (had {len(data_list)} data points, underlying: {under})")
        return curves


    def calculate_atm_slope(self, curve: VolatilityCurve) -> float:
        """
        Calculate ATM slope (skewness proxy)
        
        This is the derivative of the volatility curve at ATM (moneyness = 0)
        """
        if not curve.points:
            return 0.0
        
        try:
            # Sort points by moneyness
            sorted_points = sorted(curve.points, key=lambda p: p.moneyness)
            
            # Extract moneyness and IV arrays
            moneyness = np.array([p.moneyness for p in sorted_points])
            iv = np.array([p.iv for p in sorted_points])
            
            # Remove duplicates and sort
            unique_indices = np.unique(moneyness, return_index=True)[1]
            moneyness = moneyness[unique_indices]
            iv = iv[unique_indices]
            
            # For 2 points, use linear interpolation for slope
            if len(moneyness) < 2:
                return 0.0
            elif len(moneyness) == 2:
                # Simple linear slope between 2 points
                return float((iv[1] - iv[0]) / (moneyness[1] - moneyness[0])) if (moneyness[1] - moneyness[0]) != 0 else 0.0
            
            # For 3+ points, use cubic spline or linear interpolation
            if self.interpolation_method == 'cubic' and len(moneyness) >= 3:
                spline = CubicSpline(moneyness, iv, extrapolate=True)
                slope = spline.derivative()(0.0)
            else:
                # Linear interpolation
                interp = interp1d(moneyness, iv, kind='linear', 
                                fill_value='extrapolate')
                # Calculate slope using finite differences
                h = 0.01
                slope = (interp(h) - interp(-h)) / (2 * h)
            
            return float(slope)
            
        except Exception as e:
            logger.error(f"Error calculating ATM slope: {e}")
            return 0.0
    
    def build_volatility_surface(self, curves: List[VolatilityCurve]) -> Dict:
        """
        Build volatility surface from multiple curves
        
        Returns:
            Dictionary containing surface data and interpolation functions
        """
        if not curves:
            return {}
        
        try:
            # Extract data points
            all_points = []
            for curve in curves:
                for point in curve.points:
                    all_points.append({
                        'moneyness': point.moneyness,
                        'time_to_expiry': curve.time_to_expiry,
                        'iv': point.iv
                    })
            
            df = pd.DataFrame(all_points)
            
            # Create interpolation grid
            moneyness_range = np.linspace(df['moneyness'].min(), 
                                        df['moneyness'].max(), 50)
            time_range = np.linspace(df['time_to_expiry'].min(), 
                                  df['time_to_expiry'].max(), 20)
            
            # Interpolate surface
            surface = self._interpolate_surface(df, moneyness_range, time_range)
            
            return {
                'surface': surface,
                'moneyness_range': moneyness_range,
                'time_range': time_range,
                'raw_data': df
            }
            
        except Exception as e:
            logger.error(f"Error building volatility surface: {e}")
            return {}
    
    def calculate_term_structure(self, curves: List[VolatilityCurve]) -> Dict:
        """
        Calculate volatility term structure
        
        Returns:
            Dictionary with term structure metrics
        """
        if not curves:
            return {}
        
        try:
            # Sort curves by expiration
            sorted_curves = sorted(curves, key=lambda c: c.expiration)
            
            expirations = [c.expiration for c in sorted_curves]
            atm_ivs = [c.atm_iv for c in sorted_curves]
            slopes = [c.slope for c in sorted_curves]
            
            # Calculate time to expiry
            current_time = int(pd.Timestamp.now().timestamp())
            time_to_expiry = [(exp - current_time) / (365 * 24 * 3600) 
                            for exp in expirations]
            
            return {
                'expirations': expirations,
                'time_to_expiry': time_to_expiry,
                'atm_iv': atm_ivs,
                'slopes': slopes,
                'term_structure_slope': self._calculate_term_structure_slope(
                    time_to_expiry, atm_ivs
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating term structure: {e}")
            return {}
    
    def calculate_skewness_metrics(self, curve: VolatilityCurve) -> Dict:
        """
        Calculate comprehensive skewness metrics
        
        Returns:
            Dictionary with skewness metrics
        """
        if not curve.points:
            return {}
        
        try:
            # Sort points by moneyness
            sorted_points = sorted(curve.points, key=lambda p: p.moneyness)
            
            # Separate calls and puts
            call_points = [p for p in sorted_points if p.option_type == OptionType.CALL]
            put_points = [p for p in sorted_points if p.option_type == OptionType.PUT]
            
            if not call_points or not put_points:
                return {}
            
            # Calculate call-put skew
            call_ivs = [p.iv for p in call_points]
            put_ivs = [p.iv for p in put_points]
            
            avg_call_iv = np.mean(call_ivs)
            avg_put_iv = np.mean(put_ivs)
            call_put_skew = avg_call_iv - avg_put_iv
            
            # Calculate risk reversal (25-delta)
            rr_25 = self._calculate_risk_reversal(curve, 0.25)
            
            # Calculate butterfly spread (25-delta)
            butterfly_25 = self._calculate_butterfly_spread(curve, 0.25)
            
            # Calculate skew slope at different moneyness levels
            skew_slopes = {}
            for moneyness in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                slope = self._calculate_slope_at_moneyness(curve, moneyness)
                skew_slopes[f'slope_{moneyness}'] = slope
            
            return {
                'atm_slope': curve.slope,
                'call_put_skew': call_put_skew,
                'risk_reversal_25': rr_25,
                'butterfly_25': butterfly_25,
                'skew_slopes': skew_slopes,
                'avg_call_iv': avg_call_iv,
                'avg_put_iv': avg_put_iv,
                'skew_strength': abs(curve.slope) / curve.atm_iv if curve.atm_iv > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating skewness metrics: {e}")
            return {}
    
    def forecast_volatility(self, historical_data: List[VolatilityCurve],
                          forecast_horizon: int = 7) -> Dict:
        """
        Simple volatility forecasting using historical patterns
        
        Args:
            historical_data: List of historical volatility curves
            forecast_horizon: Forecast horizon in days
            
        Returns:
            Dictionary with forecast results
        """
        if len(historical_data) < 10:
            logger.warning("Insufficient historical data for forecasting")
            return {}
        
        try:
            # Extract time series
            timestamps = [c.timestamp for c in historical_data]
            atm_ivs = [c.atm_iv for c in historical_data]
            slopes = [c.slope for c in historical_data]
            
            # Simple moving average forecast
            recent_iv = np.mean(atm_ivs[-5:])  # Last 5 observations
            recent_slope = np.mean(slopes[-5:])
            
            # Add some trend component
            if len(atm_ivs) >= 10:
                trend = np.polyfit(range(len(atm_ivs[-10:])), atm_ivs[-10:], 1)[0]
                forecast_iv = recent_iv + trend * forecast_horizon
            else:
                forecast_iv = recent_iv
            
            return {
                'forecast_iv': max(0.01, forecast_iv),  # Ensure positive
                'forecast_slope': recent_slope,
                'confidence': self._calculate_forecast_confidence(historical_data),
                'forecast_horizon': forecast_horizon,
                'trend': trend if len(atm_ivs) >= 10 else 0
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return {}
    
   
    # Helper methods
    def _filter_valid_options(self, options_data: List[MarketData], 
                            expiration: int) -> List[MarketData]:
        """Filter valid options for given expiration"""
        return [opt for opt in options_data 
                if self._extract_expiration(opt.instrument_name) == expiration
                and opt.mark_iv > 0]
    
    def _extract_strike(self, instrument_name: str) -> Optional[float]:
        """Extract strike price from instrument name"""
        match = re.search(r'-(\d+)-[CP]', instrument_name)
        return float(match.group(1)) if match else None
    
    def _extract_option_type(self, instrument_name: str) -> OptionType:
        """Extract option type from instrument name"""
        return OptionType.CALL if '-C' in instrument_name else OptionType.PUT
    
    def _extract_expiration(self, instrument_name: str) -> Optional[int]:
        """Extract expiration timestamp from instrument name"""
        match = re.search(r'-(\d{1,2}[A-Z]{3}\d{2})-', instrument_name)
        if match:
            try:
                date_str = match.group(1)
                expiration_date = datetime.strptime(date_str, "%d%b%y")
                return int(expiration_date.timestamp())

            except ValueError:
                return None
        return None
    
    def _group_by_expiration(self,options_data: Dict[str, List[MarketData]]) -> Dict[int, List[MarketData]]:    
        """Group options data by expiration"""
        
        expiration_groups = {}
        
        for instrument, data_list in options_data.items():
            if not data_list:
                continue
                
            # Extract expiration from instrument name
            expiration = self._extract_expiration(instrument)
                    
            if expiration not in expiration_groups:
                expiration_groups[expiration] = []
            expiration_groups[expiration].extend(data_list)
           
    
        return expiration_groups

    async def _price_handler(self, channel: str, data: dict):

        if 'params' in data and 'data' in data['params']:
            mark_data = data['params']['data']
        
            # Process each instrument in mark price data
            for instrument_data in mark_data:
                instrument_name = instrument_data.get('instrument_name', 'unknown')
            
                if instrument_name not in self.collected_option_mark:
                    self.collected_option_mark[instrument_name] = []
            
                # Create MarketData object from mark price data
                market_data = MarketData(
                    instrument_name=instrument_name,
                    timestamp=instrument_data.get('timestamp', 0),
                    mark_price=instrument_data.get('mark_price'),
                    mark_iv=instrument_data.get('iv'),
                    # Mark price data typically doesn't have bid/ask
                    best_bid_price=None,
                    best_ask_price=None
                )
            
                self.collected_option_mark[instrument_name].append(market_data)
            
                # Log key instruments
                if 'BTC-' in instrument_name and ('-C' in instrument_name or '-P' in instrument_name):
                    logger.info(f"📈 {instrument_name}: Price={market_data.mark_price:.2f}, IV={market_data.mark_iv:.4f}")
            
    def _interpolate_atm_iv(self, points: List[VolatilityPoint], 
                           atm_strike: float) -> float:
        """Interpolate IV at ATM strike"""
        try:
            strikes = [p.strike for p in points]
            ivs = [p.iv for p in points]
            
            if atm_strike in strikes:
                return ivs[strikes.index(atm_strike)]
            
            # Interpolate
            interp = interp1d(strikes, ivs, kind='linear', 
                            fill_value='extrapolate')
            return float(interp(atm_strike))
        except:
            return np.mean([p.iv for p in points])
    
    def _calculate_atm_slope(self, points: List[VolatilityPoint]) -> float:
        """Calculate ATM slope using the points"""
        curve = VolatilityCurve(
            expiration=points[0].expiration,
            points=points,
            atm_strike=0,
            atm_iv=0,
            slope=0,
            curvature=0,
            timestamp=points[0].timestamp
        )
        return self.calculate_atm_slope(curve)
    
    def _calculate_curvature(self, points: List[VolatilityPoint]) -> float:
        """Calculate curvature of volatility curve"""
        try:
            sorted_points = sorted(points, key=lambda p: p.moneyness)
            moneyness = np.array([p.moneyness for p in sorted_points])
            iv = np.array([p.iv for p in sorted_points])
            
            if len(moneyness) < 3:
                return 0.0
            
            # Calculate second derivative at ATM
            spline = CubicSpline(moneyness, iv, extrapolate=True)
            curvature = spline.derivative(2)(0.0)
            return float(curvature)
        except:
            return 0.0
    
    def _evaluate_svi_fit_quality(self, strikes: np.ndarray, ivs: np.ndarray, 
                                  T: float, forward: float, 
                                  svi_params) -> Dict[str, float]:
        """
        Evaluate the quality of SVI fit.
        
        Returns:
            Dictionary with 'r_squared', 'rmse', 'max_error', and 'is_good' (bool)
        """
        try:
            from .svi import svi_iv_at_moneyness
            
            # Normalize IVs if needed
            ivs_normalized = ivs.copy()
            if np.any(ivs_normalized > 1.0):
                ivs_normalized = ivs_normalized / 100.0
            
            # Calculate log-moneyness
            k_values = np.log(strikes / forward)
            
            # Get fitted IVs from SVI
            iv_fitted = svi_iv_at_moneyness(k_values, T, svi_params)
            
            # Ensure same format (convert to numpy array if needed)
            if not isinstance(iv_fitted, np.ndarray):
                iv_fitted = np.array(iv_fitted)
            
            # Calculate residuals
            residuals = iv_fitted - ivs_normalized
            
            # Calculate metrics
            ss_res = np.sum(residuals ** 2)  # Sum of squares of residuals
            ss_tot = np.sum((ivs_normalized - np.mean(ivs_normalized)) ** 2)  # Total sum of squares
            
            # R² (coefficient of determination)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            # Max absolute error
            max_error = np.max(np.abs(residuals))
            
            # Mean absolute error
            mae = np.mean(np.abs(residuals))
            
            # Determine if fit is good
            # Criteria: R² > 0.8, RMSE < 0.1 (10% of typical IV), max error < 0.2 (20%)
            is_good = (r_squared > 0.8 and rmse < 0.1 and max_error < 0.2)
            
            return {
                'r_squared': float(r_squared),
                'rmse': float(rmse),
                'max_error': float(max_error),
                'mae': float(mae),
                'is_good': is_good
            }
        except Exception as e:
            logger.debug(f"Error evaluating SVI fit quality: {e}")
            # If evaluation fails, assume poor fit
            return {
                'r_squared': 0.0,
                'rmse': float('inf'),
                'max_error': float('inf'),
                'mae': float('inf'),
                'is_good': False
            }
    
    def _calculate_time_to_expiry(self, expiration: int) -> float:
        """Calculate time to expiry in years"""
        current_time = int(pd.Timestamp.now().timestamp())
        return (expiration - current_time) / (365 * 24 * 3600)
    
    def _interpolate_surface(self, df: pd.DataFrame, 
                           moneyness_range: np.ndarray,
                           time_range: np.ndarray) -> np.ndarray:
        """Interpolate volatility surface"""
        from scipy.interpolate import griddata
        
        points = df[['moneyness', 'time_to_expiry']].values
        values = df['iv'].values
        
        # Create grid
        moneyness_grid, time_grid = np.meshgrid(moneyness_range, time_range)
        grid_points = np.column_stack([moneyness_grid.ravel(), time_grid.ravel()])
        
        # Interpolate
        surface = griddata(points, values, grid_points, method='cubic')
        return surface.reshape(moneyness_grid.shape)
    
    def _calculate_term_structure_slope(self, time_to_expiry: List[float],
                                      atm_iv: List[float]) -> float:
        """Calculate slope of term structure"""
        if len(time_to_expiry) < 2:
            return 0.0
        
        try:
            slope = np.polyfit(time_to_expiry, atm_iv, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    def _calculate_risk_reversal(self, curve: VolatilityCurve, delta: float) -> float:
        """Calculate risk reversal for given delta"""
        try:
            # Find call and put with target delta
            call_iv = self._find_iv_at_delta(curve, delta, OptionType.CALL)
            put_iv = self._find_iv_at_delta(curve, delta, OptionType.PUT)
            
            return call_iv - put_iv
        except:
            return 0.0
    
    def _calculate_butterfly_spread(self, curve: VolatilityCurve, delta: float) -> float:
        """Calculate butterfly spread for given delta"""
        try:
            # Find IVs at target delta and ATM
            wing_iv = self._find_iv_at_delta(curve, delta, OptionType.CALL)
            atm_iv = curve.atm_iv
            
            return wing_iv - atm_iv
        except:
            return 0.0
    
    def _find_iv_at_delta(self, curve: VolatilityCurve, target_delta: float, 
                         option_type: OptionType) -> float:
        """Find IV at target delta (simplified)"""
        # This is a simplified implementation
        # In practice, you'd need to solve the Black-Scholes equation
        filtered_points = [p for p in curve.points if p.option_type == option_type]
        if not filtered_points:
            return curve.atm_iv
        
        # Use closest moneyness as proxy
        target_moneyness = np.log(1 + target_delta)  # Simplified
        closest_point = min(filtered_points, 
                          key=lambda p: abs(p.moneyness - target_moneyness))
        return closest_point.iv
    
    def _calculate_slope_at_moneyness(self, curve: VolatilityCurve, 
                                    moneyness: float) -> float:
        """Calculate slope at specific moneyness level"""
        try:
            sorted_points = sorted(curve.points, key=lambda p: p.moneyness)
            moneyness_array = np.array([p.moneyness for p in sorted_points])
            iv_array = np.array([p.iv for p in sorted_points])
            
            if len(moneyness_array) < 3:
                return 0.0
            
            spline = CubicSpline(moneyness_array, iv_array, extrapolate=True)
            slope = spline.derivative()(moneyness)
            return float(slope)
        except:
            return 0.0
    
    def _calculate_forecast_confidence(self, historical_data: List[VolatilityCurve]) -> float:
        """Calculate confidence in forecast based on historical stability"""
        if len(historical_data) < 5:
            return 0.5
        
        # Calculate coefficient of variation
        atm_ivs = [c.atm_iv for c in historical_data[-10:]]
        cv = np.std(atm_ivs) / np.mean(atm_ivs)
        
        # Convert to confidence (lower CV = higher confidence)
        confidence = max(0.1, min(0.9, 1 - cv))
        return confidence
    

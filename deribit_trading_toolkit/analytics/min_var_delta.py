"""
Minimum Variance Delta Hedging Module

Calculates minimum variance delta using SSR(τ):
    Δ_min = ∂V/∂F + e^{-rτ}φ(d₁)(β̂_dyn(k_t,τ) - ∂ₖ√w(k_t,τ))

where:
    β̂_dyn(k_t,τ) ≈ SSR(τ) · ∂ₖ√w(k_t,τ)
    SSR(τ) = β̂_dyn(ATM,τ) / ∂ₖ√w|₀

References:
- SSR(τ) needs intraday series of ATM √w and log-spot for fixed expiry τ
- Delta_min calculated once per day using end-of-day SSR estimate
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class IntradayDataPoint:
    """Single intraday data point for SSR calculation"""
    timestamp: float
    F_t: float  # Forward/spot price
    sqrtw_atm: float  # ATM √w
    logF: float  # log(F_t)
    dlogF: Optional[float] = None  # ΔlogF from previous point
    dsqrtw: Optional[float] = None  # Δ√w from previous point


@dataclass
class ExpiryIntradayData:
    """Intraday data series for a single expiry"""
    expiry_ms: int
    data_points: List[IntradayDataPoint] = field(default_factory=list)
    
    # Daily accumulators for SSR calculation
    C_xy_day: float = 0.0  # ∑ Δ√w * ΔlogF
    C_xx_day: float = 0.0  # ∑ (ΔlogF)²
    slope_atm_sum: float = 0.0  # ∑ ∂ₖ√w|₀ (ATM slope)
    slope_atm_count: int = 0
    
    # End-of-day SSR estimate
    SSR_tau: Optional[float] = None
    
    def add_point(self, timestamp: float, F_t: float, sqrtw_atm: float, slope_atm: float):
        """Add a new intraday data point"""
        logF = math.log(F_t) if F_t > 0 else 0.0
        
        # Calculate increments from previous point
        dlogF = None
        dsqrtw = None
        
        if self.data_points:
            prev = self.data_points[-1]
            dlogF = logF - prev.logF
            dsqrtw = sqrtw_atm - prev.sqrtw_atm
            
            # Update daily accumulators
            if dlogF is not None and dsqrtw is not None:
                self.C_xy_day += dsqrtw * dlogF
                self.C_xx_day += dlogF * dlogF
        
        # Add new point
        point = IntradayDataPoint(
            timestamp=timestamp,
            F_t=F_t,
            sqrtw_atm=sqrtw_atm,
            logF=logF,
            dlogF=dlogF,
            dsqrtw=dsqrtw
        )
        self.data_points.append(point)
        
        # Update slope accumulator
        self.slope_atm_sum += slope_atm
        self.slope_atm_count += 1
    
    def calculate_daily_SSR(self) -> Optional[float]:
        """
        Calculate end-of-day SSR(τ) estimate.
        
        SSR(τ) = β̂_dyn(ATM,τ) / ∂ₖ√w|₀
        where β̂_dyn(ATM,τ) = C_xy / C_xx
        """
        if self.C_xx_day > 1e-16 and self.slope_atm_count > 0:
            beta_dyn = self.C_xy_day / self.C_xx_day
            slope_mean = self.slope_atm_sum / self.slope_atm_count
            
            if abs(slope_mean) > 1e-16:
                self.SSR_tau = beta_dyn / slope_mean
            else:
                self.SSR_tau = None
        else:
            self.SSR_tau = None
        
        return self.SSR_tau
    
    def reset_day(self):
        """Reset for new trading day"""
        self.data_points.clear()
        self.C_xy_day = 0.0
        self.C_xx_day = 0.0
        self.slope_atm_sum = 0.0
        self.slope_atm_count = 0
        self.SSR_tau = None


@dataclass
class MinimumVarianceDeltaResult:
    """Result of minimum variance delta calculation"""
    delta_min: float
    delta_bs: float  # Black-Scholes delta
    adjustment_term: float  # e^{-rτ}φ(d₁)(β̂_dyn - ∂ₖ√w)
    beta_dyn: float  # β̂_dyn(k_t,τ)
    slope_atm: float  # ∂ₖ√w(k_t,τ)
    SSR_tau: Optional[float]  # SSR(τ) used
    diagnostics: Dict[str, float] = field(default_factory=dict)


class MinimumVarianceDeltaCalculator:
    """
    Minimum Variance Delta Calculator
    
    Calculates Δ_min = ∂V/∂F + e^{-rτ}φ(d₁)(β̂_dyn(k_t,τ) - ∂ₖ√w(k_t,τ))
    
    where:
    - β̂_dyn(k_t,τ) ≈ SSR(τ) · ∂ₖ√w(k_t,τ)
    - SSR(τ) is calculated from intraday series of ATM √w and log-spot
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize minimum variance delta calculator.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.05 = 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.intraday_data: Dict[int, ExpiryIntradayData] = {}
    
    def add_intraday_point(
        self,
        expiry_ms: int,
        timestamp: float,
        F_t: float,
        sqrtw_atm: float,
        slope_atm: float
    ) -> None:
        """
        Add an intraday data point for SSR(τ) calculation.
        
        Args:
            expiry_ms: Expiration timestamp in milliseconds
            timestamp: Current timestamp in seconds
            F_t: Current forward/spot price
            sqrtw_atm: ATM √w (square root of total variance)
            slope_atm: ∂ₖ√w|₀ (ATM slope)
        """
        if expiry_ms not in self.intraday_data:
            self.intraday_data[expiry_ms] = ExpiryIntradayData(expiry_ms=expiry_ms)
        
        self.intraday_data[expiry_ms].add_point(
            timestamp=timestamp,
            F_t=F_t,
            sqrtw_atm=sqrtw_atm,
            slope_atm=slope_atm
        )
    
    def calculate_daily_SSR(self, expiry_ms: int) -> Optional[float]:
        """
        Calculate end-of-day SSR(τ) for a specific expiry.
        
        Args:
            expiry_ms: Expiration timestamp in milliseconds
            
        Returns:
            SSR(τ) estimate or None if insufficient data
        """
        if expiry_ms not in self.intraday_data:
            return None
        
        data = self.intraday_data[expiry_ms]
        return data.calculate_daily_SSR()
    
    def get_SSR_tau(self, expiry_ms: int) -> Optional[float]:
        """
        Get current SSR(τ) estimate for an expiry.
        
        Args:
            expiry_ms: Expiration timestamp in milliseconds
            
        Returns:
            SSR(τ) estimate or None
        """
        if expiry_ms not in self.intraday_data:
            return None
        
        return self.intraday_data[expiry_ms].SSR_tau
    
    def calculate_delta_min(
        self,
        F_t: float,
        K: float,
        tau_years: float,
        sqrtw: float,
        d_sqrtw_dk: float,
        expiry_ms: Optional[int] = None,
        SSR_tau: Optional[float] = None,
        option_type: str = 'call'  # 'call' or 'put'
    ) -> MinimumVarianceDeltaResult:
        """
        Calculate minimum variance delta.
        
        Δ_min = ∂V/∂F + e^{-rτ}φ(d₁)(β̂_dyn(k_t,τ) - ∂ₖ√w(k_t,τ))
        
        where:
        - β̂_dyn(k_t,τ) ≈ SSR(τ) · ∂ₖ√w(k_t,τ)
        - ∂V/∂F is the Black-Scholes delta
        
        Args:
            F_t: Current forward/spot price
            K: Strike price
            tau_years: Time to expiry in years
            sqrtw: √w at current moneyness
            d_sqrtw_dk: ∂ₖ√w at current moneyness
            expiry_ms: Expiration timestamp (optional, for SSR lookup)
            SSR_tau: SSR(τ) estimate (optional, if not provided will lookup or use approximation)
            option_type: 'call' or 'put' (default: 'call')
            
        Returns:
            MinimumVarianceDeltaResult with delta_min and diagnostics
        """
        # Calculate k_t (log-moneyness)
        k_t = math.log(K / F_t) if F_t > 0 else 0.0
        
        # Calculate d1 and d2
        if sqrtw <= 0:
            logger.warning("sqrtw <= 0, returning zero delta_min")
            return MinimumVarianceDeltaResult(
                delta_min=0.0,
                delta_bs=0.0,
                adjustment_term=0.0,
                beta_dyn=0.0,
                slope_atm=d_sqrtw_dk,
                SSR_tau=SSR_tau,
                diagnostics={
                    "k_t": k_t,
                    "sqrtw": sqrtw,
                    "tau_years": tau_years,
                    "note": "Degenerate case: sqrtw <= 0"
                }
            )
        
        d1 = -k_t / sqrtw + 0.5 * sqrtw
        d2 = d1 - sqrtw
        
        # Discount factor
        disc = math.exp(-self.risk_free_rate * tau_years)
        
        # Black-Scholes delta: ∂V/∂F
        # For calls: delta = e^{-rτ}Φ(d₁)
        # For puts: delta = -e^{-rτ}Φ(-d₁)
        if option_type.lower() == 'put':
            delta_bs = -disc * norm.cdf(-d1)
        else:  # call (default)
            delta_bs = disc * norm.cdf(d1)
        
        # Get SSR(τ) if not provided
        if SSR_tau is None and expiry_ms is not None:
            SSR_tau = self.get_SSR_tau(expiry_ms)
        
        # Calculate β̂_dyn(k_t,τ) ≈ SSR(τ) · ∂ₖ√w(k_t,τ)
        if SSR_tau is not None:
            beta_dyn = SSR_tau * d_sqrtw_dk
        else:
            logger.warning("SSR(τ) not available, using SSR=1 approximation")
            # Fallback: use d_sqrtw_dk directly if SSR not available
            # This approximates SSR = 1 (sticky strike regime)
            beta_dyn = d_sqrtw_dk
        
        # Adjustment term: e^{-rτ}φ(d₁)(β̂_dyn - ∂ₖ√w)
        phi_d1 = norm.pdf(d1)
        adjustment_term = disc * phi_d1 * (beta_dyn - d_sqrtw_dk)
        
        # Minimum variance delta
        delta_min = delta_bs + adjustment_term
      


        
        return MinimumVarianceDeltaResult(
            delta_min=delta_min,
            delta_bs=delta_bs,
            adjustment_term=adjustment_term,
            beta_dyn=beta_dyn,
            slope_atm=d_sqrtw_dk,
            SSR_tau=SSR_tau,
            diagnostics={
                "k_t": k_t,
                "d1": d1,
                "d2": d2,
                "sqrtw": sqrtw,
                "tau_years": tau_years,
                "disc": disc,
                "phi_d1": phi_d1
            }
        )
    
    def reset_day(self, expiry_ms: Optional[int] = None):
        """
        Reset intraday data for new trading day.
        
        Args:
            expiry_ms: Specific expiry to reset (None = reset all)
        """
        if expiry_ms is not None:
            if expiry_ms in self.intraday_data:
                self.intraday_data[expiry_ms].reset_day()
        else:
            for data in self.intraday_data.values():
                data.reset_day()
            logger.info("Reset all intraday data for new trading day")
    
    def get_all_SSR_tau(self) -> Dict[int, Optional[float]]:
        """
        Get SSR(τ) estimates for all expiries.
        
        Returns:
            Dictionary mapping expiry_ms to SSR(τ) estimate
        """
        result = {}
        for expiry_ms, data in self.intraday_data.items():
            result[expiry_ms] = data.SSR_tau
        return result


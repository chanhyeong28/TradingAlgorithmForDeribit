"""
Skew Stickiness Ratio (SSR) Analytics Module

Calculates SSR dynamically using Kalman filter:
    SSR = β / slope_atm
where:
    β = Cov(Δ√w, ΔlogF) / Var(ΔlogF) = ∫(d√w * dlogF) / ∫(dlogF)²
    slope_atm = ∂√w/∂k|_ATM

References:
- SSR = 1 for sticky strike regime
- SSR = 0 for sticky moneyness regime
- Empirically SSR is typically between 0.9 and 1.7 for S&P 500 options
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Kalman filter state for SSR estimation"""
    # State estimate
    ssr_estimate: float = 1.0  # Initial estimate (neutral between regimes)
    
    # Uncertainty (covariance)
    P: float = 1.0  # Initial uncertainty
    
    # Process noise and measurement noise
    Q: float = 0.01  # Process noise covariance (how much SSR can change)
    R: float = 0.1   # Measurement noise covariance (uncertainty in observed SSR)


@dataclass
class ExpiryState:
    """State per expiry for SSR calculation with Kalman filtering"""
    # Last tick values
    last_F: float = 0.0
    last_sqrtw_atm: float = 0.0
    last_ts: float = 0.0
    initialized: bool = False
    
    # Running statistics for beta estimation (exponentially filtered)
    C_xy_filtered: float = 0.0  # Filtered covariance: E[Δ√w * ΔlogF]
    C_xx_filtered: float = 0.0  # Filtered variance: E[(ΔlogF)^2]
    slope_filtered: float = 0.0  # Filtered ATM slope
    
    # Kalman filter for SSR
    kalman: Optional[KalmanState] = None
    
    # Current SSR estimate from Kalman filter
    SSR_estimate: Optional[float] = None
    SSR_uncertainty: Optional[float] = None


class SSRCalculator:
    """
    Skew Stickiness Ratio Calculator with Kalman Filter
    
    Uses a Kalman filter to estimate SSR from noisy observations:
    - State: SSR (the true SSR we want to estimate)
    - Observation: SSR_observed = beta / slope_atm (noisy measurement)
    - Process model: SSR_t = SSR_{t-1} + w_t (random walk with noise)
    - Observation model: SSR_observed = SSR_true + v_t (with measurement noise)
    
    The Kalman filter provides:
    - Optimal estimate of SSR given noisy observations
    - Uncertainty quantification (P)
    - Adaptive filtering that responds to changing market conditions
    """
    
    MIN_SAMPLE_SPACING = 10.0  # seconds - downsample to ~10s effective bars
    WINSOR_DLOGF = 0.05  # clip ±5% per bar
    WINSOR_DSW = 0.05  # clip ±5% per bar
    
    # SSR bounds (empirical range for S&P 500: 0.9-1.7, we use wider bounds)
    SSR_MIN = 0.0   # Sticky moneyness regime
    SSR_MAX = 2.0   # Conservative upper bound
    
    def __init__(
        self,
        min_sample_spacing: float = 10.0,
        winsor_dlogf: float = 0.05,
        winsor_dsw: float = 0.05,
        alpha: float = 0.1,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_ssr: float = 1.0
    ):
        """
        Initialize SSR calculator with Kalman filter.
        
        Args:
            min_sample_spacing: Minimum time between samples (seconds)
            winsor_dlogf: Winsorization limit for ΔlogF
            winsor_dsw: Winsorization limit for Δ√w
            alpha: Exponential smoothing factor for beta/slope estimation (0 < alpha <= 1)
            process_noise: Process noise covariance Q (how much SSR can change per step)
            measurement_noise: Measurement noise covariance R (uncertainty in observed SSR)
            initial_ssr: Initial SSR estimate (default 1.0 = neutral)
        """
        self.min_sample_spacing = min_sample_spacing
        self.winsor_dlogf = winsor_dlogf
        self.winsor_dsw = winsor_dsw
        self.alpha = alpha
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_ssr = initial_ssr
        self.expiry_states: Dict[int, ExpiryState] = {}
    
    @staticmethod
    def tau_years(expiration_ms: int, now_ms: float) -> float:
        """
        Calculate time to expiry in years using ACT/365 convention.
        
        Args:
            expiration_ms: Expiration timestamp in milliseconds
            now_ms: Current timestamp in milliseconds
            
        Returns:
            Time to expiry in years (max 0)
        """
        return max((expiration_ms - now_ms) / (1000.0 * 60.0 * 60.0 * 24.0 * 365.0), 0.0)
    
    @staticmethod
    def sqrtw_from_iv(iv1y: float, tau: float) -> float:
        """
        Convert annualized IV to √w (square root of total variance).
        
        Uses ACT/365 convention: √w = IV * √τ
        
        Args:
            iv1y: Annualized implied volatility
            tau: Time to expiry in years
            
        Returns:
            √w (square root of total variance)
        """
        if tau <= 0:
            return 0.0
        return iv1y * math.sqrt(tau)
    
    @staticmethod
    def winsorize_scalar(x: float, lo: float, hi: float) -> float:
        """
        Winsorize a scalar value to bounds.
        
        Args:
            x: Value to winsorize
            lo: Lower bound
            hi: Upper bound
            
        Returns:
            Winsorized value
        """
        return min(max(x, lo), hi)
    
    def _kalman_predict(self, kalman: KalmanState) -> None:
        """
        Prediction step of Kalman filter.
        
        x_t|t-1 = x_t-1|t-1  (state prediction, random walk)
        P_t|t-1 = P_t-1|t-1 + Q  (uncertainty increases due to process noise)
        
        Args:
            kalman: KalmanState to update
        """
        # State prediction (random walk: SSR_t = SSR_{t-1})
        # No change to estimate, uncertainty increases
        kalman.P = kalman.P + kalman.Q
    
    def _kalman_update(self, kalman: KalmanState, z_observed: float) -> None:
        """
        Update step of Kalman filter.
        
        K = P_t|t-1 / (P_t|t-1 + R)  (Kalman gain)
        x_t|t = x_t|t-1 + K * (z_t - x_t|t-1)  (state update)
        P_t|t = (1 - K) * P_t|t-1  (uncertainty update)
        
        Args:
            kalman: KalmanState to update
            z_observed: Observed SSR value (beta / slope_atm)
        """
        # Kalman gain
        K = kalman.P / (kalman.P + kalman.R)
        
        # State update (innovation = observed - predicted)
        innovation = z_observed - kalman.ssr_estimate
        kalman.ssr_estimate = kalman.ssr_estimate + K * innovation
        
        # Bound SSR to reasonable range
        kalman.ssr_estimate = self.winsorize_scalar(
            kalman.ssr_estimate, self.SSR_MIN, self.SSR_MAX
        )
        
        # Uncertainty update
        kalman.P = (1.0 - K) * kalman.P
        
        # Prevent uncertainty from collapsing to zero
        kalman.P = max(kalman.P, 1e-6)
    
    def on_tick(
        self,
        t: float,
        F_t: float,
        term_structure: List[Dict]
    ) -> Dict[int, Optional[float]]:
        """
        Process a tick with term structure data and update SSR estimates using Kalman filter.
        
        Args:
            t: Current timestamp in seconds
            F_t: Current forward/spot price
            term_structure: List of term structure items, each with:
                - expiry_ms: Expiration timestamp in milliseconds
                - atm_iv1y: Annualized ATM IV
                - slope_atm: ∂√w/∂k|_ATM (slope of √w curve at ATM in moneyness space)
        
        Returns:
            Dictionary mapping expiry_ms to current SSR estimate (or None)
        """
        t_ms = t * 1000.0
        ssr_updates = {}
        
        for item in term_structure:
            exp_ms = item['expiry_ms']
            tau = self.tau_years(int(exp_ms), t_ms)
            
            if tau <= 0:
                continue
            
            # Extract term structure data
            atm_iv1y = item.get('atm_iv1y', 0.0)
            slope_atm_t = item.get('slope_atm', 0.0)
            
            # Convert IV to √w
            sqrtw_atm_t = self.sqrtw_from_iv(atm_iv1y, tau)
            
            # Get or create expiry state
            S = self.expiry_states.get(exp_ms)
            if S is None:
                S = ExpiryState()
                S.initialized = False
                # Initialize Kalman filter
                S.kalman = KalmanState(
                    ssr_estimate=self.initial_ssr,
                    P=1.0,
                    Q=self.process_noise,
                    R=self.measurement_noise
                )
                self.expiry_states[exp_ms] = S
            
            # Enforce minimal spacing
            if S.initialized and (t - S.last_ts) < self.min_sample_spacing:
                # Still update slope filter with recent value
                if abs(slope_atm_t) > 1e-16:
                    if S.slope_filtered == 0.0:
                        S.slope_filtered = slope_atm_t
                    else:
                        S.slope_filtered = (1.0 - self.alpha) * S.slope_filtered + self.alpha * slope_atm_t
                
                # Perform Kalman prediction step (even if skipping update)
                self._kalman_predict(S.kalman)
                S.SSR_estimate = S.kalman.ssr_estimate
                S.SSR_uncertainty = S.kalman.P
                
                continue
            
            # Initialize if first sample
            if not S.initialized:
                S.last_F = F_t
                S.last_sqrtw_atm = sqrtw_atm_t
                S.last_ts = t
                S.initialized = True
                
                # Initialize filters
                S.C_xy_filtered = 0.0
                S.C_xx_filtered = 0.0
                S.slope_filtered = slope_atm_t if abs(slope_atm_t) > 1e-16 else 0.0
                # Set initial SSR estimate (from Kalman filter initial state)
                S.SSR_estimate = S.kalman.ssr_estimate
                S.SSR_uncertainty = S.kalman.P
                # Return initial estimate so it appears on dashboard
                ssr_updates[int(exp_ms)] = S.SSR_estimate
                continue
            
            # Calculate increments
            dlogF = math.log(F_t / S.last_F) if S.last_F > 0 else 0.0
            dsw = sqrtw_atm_t - S.last_sqrtw_atm
            
            # Winsorization
            dlogF = self.winsorize_scalar(dlogF, -self.WINSOR_DLOGF, self.WINSOR_DLOGF)
            dsw = self.winsorize_scalar(dsw, -self.WINSOR_DSW, self.WINSOR_DSW)
            
            # Update exponential filters for beta estimation
            # C_xy = E[Δ√w * ΔlogF]
            xy_observation = dsw * dlogF
            if S.C_xy_filtered == 0.0:
                S.C_xy_filtered = xy_observation
            else:
                S.C_xy_filtered = (1.0 - self.alpha) * S.C_xy_filtered + self.alpha * xy_observation
            
            # C_xx = E[(ΔlogF)^2]
            xx_observation = dlogF * dlogF
            if S.C_xx_filtered == 0.0:
                S.C_xx_filtered = xx_observation
            else:
                S.C_xx_filtered = (1.0 - self.alpha) * S.C_xx_filtered + self.alpha * xx_observation
            
            # Update slope filter
            if abs(slope_atm_t) > 1e-16:
                if S.slope_filtered == 0.0:
                    S.slope_filtered = slope_atm_t
                else:
                    S.slope_filtered = (1.0 - self.alpha) * S.slope_filtered + self.alpha * slope_atm_t
            
            # Calculate observed SSR (noisy measurement)
            ssr_observed = None
            if S.C_xx_filtered > 1e-16 and abs(S.slope_filtered) > 1e-16:
                beta = S.C_xy_filtered / S.C_xx_filtered
                ssr_observed = beta / S.slope_filtered
                
                # Bound observed SSR
                ssr_observed = self.winsorize_scalar(
                    ssr_observed, self.SSR_MIN, self.SSR_MAX
                )
            
            # Kalman filter update
            if ssr_observed is not None:
                # Prediction step
                self._kalman_predict(S.kalman)
                
                # Update step with observed SSR
                self._kalman_update(S.kalman, ssr_observed)
                
                S.SSR_estimate = S.kalman.ssr_estimate
                S.SSR_uncertainty = S.kalman.P
            else:
                # No valid observation, just prediction step
                self._kalman_predict(S.kalman)
                S.SSR_estimate = S.kalman.ssr_estimate
                S.SSR_uncertainty = S.kalman.P
            
            # Update last values
            S.last_F = F_t
            S.last_sqrtw_atm = sqrtw_atm_t
            S.last_ts = t
            
            ssr_updates[int(exp_ms)] = S.SSR_estimate
        
        return ssr_updates
    
    def get_current_SSR(self) -> Dict[int, Optional[float]]:
        """
        Get current SSR estimates for all expiries.
        
        Returns:
            Dictionary mapping expiry_ms to SSR estimate (or None if not available)
        """
        result = {}
        for exp_ms, S in self.expiry_states.items():
            result[int(exp_ms)] = S.SSR_estimate
        return result
    
    def get_current_SSR_with_uncertainty(self) -> Dict[int, tuple]:
        """
        Get current SSR estimates with uncertainty for all expiries.
        
        Returns:
            Dictionary mapping expiry_ms to (SSR_estimate, uncertainty) tuple
        """
        result = {}
        for exp_ms, S in self.expiry_states.items():
            if S.SSR_estimate is not None:
                result[int(exp_ms)] = (S.SSR_estimate, S.SSR_uncertainty or 0.0)
            else:
                result[int(exp_ms)] = (None, None)
        return result
    
    def reset_day(self) -> None:
        """Reset all expiry states for a new trading day"""
        self.expiry_states.clear()
        logger.info("SSR calculator reset for new trading day")

"""
PnL Decomposition Module

Model-free P&L decomposition using implied volatility surface analysis.
Decomposes option P&L into buckets: funding, IR theta, delta, gamma, vol block, and vanna block.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from math import log, exp, sqrt
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class PnLDecompositionResult:
    """Result of P&L decomposition calculation"""
    funding: float
    ir_theta: float
    delta: float
    gamma: float
    vol_block: float
    vanna_block: float
    total: float
    diagnostics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "funding": self.funding,
            "ir_theta": self.ir_theta,
            "delta": self.delta,
            "gamma": self.gamma,
            "vol_block": self.vol_block,
            "vanna_block": self.vanna_block,
            "total": self.total,
            "_diagnostics": self.diagnostics
        }


class PnLDecomposer:
    """
    Model-free P&L decomposition using implied volatility surface.
    
    Features:
    - ACT/365 day-count convention support
    - Model-free delta and gamma calculation
    - Volatility surface move decomposition
    - Multiple input formats (√w space or annualized IV space)
    
    All calculations use ACT/365 convention where 1 year = 365 days.
    """
    
    def __init__(self, risk_free_rate: float = 0.05, basis_days: float = 365.0):
        """
        Initialize PnL decomposer
        
        Args:
            risk_free_rate: Continuous risk-free rate (default: 0.05 = 5%)
            basis_days: Days basis for ACT/365 convention (default: 365.0)
        """
        self.risk_free_rate = risk_free_rate
        self.basis_days = basis_days
    
    def tau_years_from_days(self, days_to_expiry: float) -> float:
        """
        Convert days to year-fraction under ACT/365 convention.
        
        Args:
            days_to_expiry: Days until expiration
            
        Returns:
            Year fraction (tau)
        """
        return max(days_to_expiry / self.basis_days, 0.0)
    
    def sqrtw_from_iv_1y(self, iv_1y: float, tau_years: float) -> float:
        """
        Calculate √w from an annualized IV (1y=365d).
        
        Args:
            iv_1y: Annualized implied volatility
            tau_years: Time to expiry in years
            
        Returns:
            √w (square root of total variance)
        """
        return max(iv_1y, 0.0) * sqrt(max(tau_years, 0.0))
    
    def d_sqrtw_dk_from_dsigma_dk(self, dsigma1y_dk: float, tau_years: float) -> float:
        """
        Calculate ∂√w/∂k from ∂σ_1y/∂k under 1y=365d quoting.
        
        Args:
            dsigma1y_dk: First derivative of annualized IV w.r.t. log-strike
            tau_years: Time to expiry in years
            
        Returns:
            ∂√w/∂k
        """
        return dsigma1y_dk * sqrt(max(tau_years, 0.0))
    
    def d2_sqrtw_dk2_from_d2sigma_dk2(self, d2sigma1y_dk2: float, tau_years: float) -> float:
        """
        Calculate ∂²√w/∂k² from ∂²σ_1y/∂k² under 1y=365d quoting.
        
        Args:
            d2sigma1y_dk2: Second derivative of annualized IV w.r.t. log-strike
            tau_years: Time to expiry in years
            
        Returns:
            ∂²√w/∂k²
        """
        return d2sigma1y_dk2 * sqrt(max(tau_years, 0.0))
    
    def decompose(
        self,
        *,
        # Spot/strike/time/rate
        F_t: float,
        F_next: float,
        K: float,
        tau_years: float,
        r: Optional[float] = None,
        dt: float = 0.0,
        V_t: float = 0.0,
        Pi_t: float = 0.0,
        option_type: str = 'call',  # 'call' or 'put'
        
        # EITHER pass √w and its k-derivatives at (t, k_t) ...
        sqrtw_t_kt: Optional[float] = None,
        d_sqrtw_dk_t_kt: Optional[float] = None,
        d2_sqrtw_dk2_t_kt: Optional[float] = None,
        
        # ... OR pass annualized IV (1y=365d) and its k-slopes at (t, k_t)
        iv1y_t_kt: Optional[float] = None,
        d_iv1y_dk_t_kt: Optional[float] = None,
        d2_iv1y_dk2_t_kt: Optional[float] = None,
        
        # Observed IV-surface moves measured at a fixed moneyness k0 (e.g., k0=0, ATM):
        # You can pass directly in √w-space ...
        delta_sqrtw_k0: Optional[float] = None,
        delta_d_sqrtw_dk_k0: Optional[float] = None,
        
        # ... or pass moves in annualized IV-space and the function will convert:
        delta_iv1y_k0: Optional[float] = None,
        delta_d_iv1y_dk_k0: Optional[float] = None,
    ) -> PnLDecompositionResult:
        """
        Discrete-time model-free option P&L decomposition using implied-vol surface.
        
        All terms are expressed with observed spot return, IV moves at fixed moneyness (k0),
        and model-free Δ/Γ identities built from √w(t, k_t) and its k-derivatives.
        
        Args:
            F_t: Forward/spot price at time t
            F_next: Forward/spot price at time t+dt
            K: Strike price
            tau_years: ACT/365 year-fraction to expiry at time t
            r: Continuous risk-free rate (defaults to instance risk_free_rate)
            dt: Bar length (years) for carry buckets
            V_t: Option MtM at t (for IR-theta carry)
            Pi_t: Portfolio MtM at t (for funding carry)
            
            # Either provide √w inputs at (t, k_t):
            sqrtw_t_kt: √w at (t, k_t)
            d_sqrtw_dk_t_kt: First derivative of √w w.r.t. k at (t, k_t)
            d2_sqrtw_dk2_t_kt: Second derivative of √w w.r.t. k at (t, k_t)
            
            # Or provide annualized IV inputs at (t, k_t):
            iv1y_t_kt: Annualized IV (1y=365d) at (t, k_t)
            d_iv1y_dk_t_kt: First derivative of IV w.r.t. k at (t, k_t)
            d2_iv1y_dk2_t_kt: Second derivative of IV w.r.t. k at (t, k_t)
            
            # Either provide √w-space moves at fixed moneyness k0:
            delta_sqrtw_k0: Change in √w at fixed moneyness k0
            delta_d_sqrtw_dk_k0: Change in ∂√w/∂k at fixed moneyness k0
            
            # Or provide annualized IV-space moves at fixed moneyness k0:
            delta_iv1y_k0: Change in IV at fixed moneyness k0
            delta_d_iv1y_dk_k0: Change in ∂IV/∂k at fixed moneyness k0
            
        Returns:
            PnLDecompositionResult with buckets: funding, ir_theta, delta, gamma, 
            vol_block, vanna_block, total, and diagnostics
        """
        if r is None:
            r = self.risk_free_rate
        
        # --- Pick √w inputs (priority: provided √w; else convert from IV(1y) under ACT/365) ---
        if sqrtw_t_kt is None:
            if iv1y_t_kt is None:
                raise ValueError(
                    "Provide either (sqrtw_t_kt, d_sqrtw_dk_t_kt, d2_sqrtw_dk2_t_kt) "
                    "or (iv1y_t_kt, d_iv1y_dk_t_kt, d2_iv1y_dk2_t_kt)."
                )
            sqrtw = self.sqrtw_from_iv_1y(iv1y_t_kt, tau_years)
            d_sqrtw_dk = self.d_sqrtw_dk_from_dsigma_dk(d_iv1y_dk_t_kt, tau_years)
            d2_sqrtw_dk2 = self.d2_sqrtw_dk2_from_d2sigma_dk2(d2_iv1y_dk2_t_kt, tau_years)
        else:
            if d_sqrtw_dk_t_kt is None or d2_sqrtw_dk2_t_kt is None:
                raise ValueError(
                    "When providing sqrtw_t_kt, also provide d_sqrtw_dk_t_kt and d2_sqrtw_dk2_t_kt."
                )
            sqrtw = float(sqrtw_t_kt)
            d_sqrtw_dk = float(d_sqrtw_dk_t_kt)
            d2_sqrtw_dk2 = float(d2_sqrtw_dk2_t_kt)
        
        # --- Convert *moves* at k0 if needed ---
        if (delta_sqrtw_k0 is None) != (delta_d_sqrtw_dk_k0 is None):
            raise ValueError(
                "Pass both delta_sqrtw_k0 and delta_d_sqrtw_dk_k0, "
                "or pass both delta_iv1y_k0 and delta_d_iv1y_dk_k0."
            )
        if delta_sqrtw_k0 is None:
            if (delta_iv1y_k0 is None) or (delta_d_iv1y_dk_k0 is None):
                raise ValueError(
                    "Provide IV moves at k0: either in √w-space or IV(1y)-space."
                )
            delta_sqrtw = delta_iv1y_k0 * sqrt(max(tau_years, 0.0))
            delta_d_sqrtw_dk = delta_d_iv1y_dk_k0 * sqrt(max(tau_years, 0.0))
        else:
            delta_sqrtw = float(delta_sqrtw_k0)
            delta_d_sqrtw_dk = float(delta_d_sqrtw_dk_k0)
        
        # --- Spot moves ---
        dlogF = log(F_next / F_t)
        dF = F_next - F_t
        k_t = log(K / F_t)
        
        
        # --- Geometry at (t, k_t) built on total variance w = (√w)^2 ---
        if sqrtw <= 0.0:
            # safety for degenerate very-short-tenor edge cases
            logger.warning("sqrtw <= 0, returning zero P&L buckets")
            return PnLDecompositionResult(
                funding=0.0,
                ir_theta=0.0,
                delta=0.0,
                gamma=0.0,
                vol_block=0.0,
                vanna_block=0.0,
                total=0.0,
                diagnostics={
                    "k_t": k_t,
                    "sqrtw": sqrtw,
                    "dlogF": dlogF,
                    "dF": dF,
                    "tau_years": tau_years,
                    "note": "Degenerate case: sqrtw <= 0"
                }
            )
    
 
        d1 = -k_t / sqrtw + 0.5 * sqrtw
        d2 = d1 - sqrtw
        disc = exp(-r * tau_years)
        
        # Discounted BS delta, $-gamma
        # For calls: delta = disc * norm.cdf(d1)
        # For puts: delta = -disc * norm.cdf(-d1)
        if option_type.lower() == 'put':
            delta_bs_disc = -disc * norm.cdf(-d1)
        else:  # call (default)
            delta_bs_disc = disc * norm.cdf(d1)
        cash_gamma = disc * (F_t * norm.pdf(d1) / sqrtw)  # $Γ_BS (same for calls and puts)
        
        # ----------------------
        # MODEL-FREE DELTA (∂V/∂F)
        # ----------------------
        # dV_dF = delta_bs_disc - disc * norm.pdf(d1) * d_sqrtw_dk
        dV_dF = delta_bs_disc 
        
        
        # ----------------------
        # MODEL-FREE GAMMA via g(t, k) from slides
        # g = (1 - k/√w * ∂√w/∂k)^2 - (w/4)*(∂√w/∂k)^2 + √w * ∂²√w/∂k²  ≥ 0
        # ----------------------
        g = (1.0 - (k_t / sqrtw) * d_sqrtw_dk) ** 2 \
            - (sqrtw ** 2 / 4.0) * (d_sqrtw_dk ** 2) \
            + sqrtw * d2_sqrtw_dk2
        g = max(g, 0.0)  # numerical guard
        
        # -------------
        # Buckets
        # -------------
        funding = r * (Pi_t - V_t) * dt
        ir_theta = r * V_t * dt
        delta_pnl = dV_dF * dF
        print(f"dV_dF: {dV_dF}")
        print(f"dF: {dF}")
        print(f"delta_pnl: {delta_pnl}")
        
        gamma_pnl = 0.5 * cash_gamma * g * (dlogF ** 2)
        
        # Vol block (vol-theta + vega + volga) via IV move at fixed k0
        vol_block = cash_gamma * (sqrtw * delta_sqrtw + 0.5 * d1 * d2 * (delta_sqrtw ** 2))
        
        # Vanna block via IV & skew moves at fixed k0
        vanna_block = -cash_gamma * (
            sqrtw * delta_d_sqrtw_dk + d2 * (1.0 + d1 * d_sqrtw_dk) * delta_sqrtw
        ) * dlogF
      
        total = funding + ir_theta + delta_pnl + gamma_pnl + vol_block + vanna_block
        
        diagnostics = {
            "k_t": k_t,
            "sqrtw": sqrtw,
            "d1": d1,
            "d2": d2,
            "disc_delta_bs": delta_bs_disc,
            "$Gamma_BS": cash_gamma,
            "dV_dF_model_free": dV_dF,
            "g_model_free": g,
            "dlogF": dlogF,
            "dF": dF,
            "tau_years": tau_years
        }
        
        return PnLDecompositionResult(
            funding=funding,
            ir_theta=ir_theta,
            delta=delta_pnl,
            gamma=gamma_pnl,
            vol_block=vol_block,
            vanna_block=vanna_block,
            total=total,
            diagnostics=diagnostics
        )
    
    def decompose_daily(
        self,
        *,
        F_t: float,
        F_next: float,
        K: float,
        days_to_expiry: float,
        r: Optional[float] = None,
        dt_days: float = 1.0,
        V_t: float = 0.0,
        Pi_t: float = 0.0,
        sqrtw_t_kt: Optional[float] = None,
        d_sqrtw_dk_t_kt: Optional[float] = None,
        d2_sqrtw_dk2_t_kt: Optional[float] = None,
        iv1y_t_kt: Optional[float] = None,
        d_iv1y_dk_t_kt: Optional[float] = None,
        d2_iv1y_dk2_t_kt: Optional[float] = None,
        delta_sqrtw_k0: Optional[float] = None,
        delta_d_sqrtw_dk_k0: Optional[float] = None,
        delta_iv1y_k0: Optional[float] = None,
        delta_d_iv1y_dk_k0: Optional[float] = None,
    ) -> PnLDecompositionResult:
        """
        Convenience method for daily P&L decomposition.
        
        Automatically converts days to years using ACT/365 convention.
        
        Args:
            F_t: Forward/spot price at time t
            F_next: Forward/spot price next day
            K: Strike price
            days_to_expiry: Days until expiration
            r: Continuous risk-free rate (defaults to instance risk_free_rate)
            dt_days: Bar length in days (default: 1.0 day)
            V_t: Option MtM at t
            Pi_t: Portfolio MtM at t
            # ... (same as decompose() for other parameters)
            
        Returns:
            PnLDecompositionResult
        """
        tau_years = self.tau_years_from_days(days_to_expiry)
        dt_years = self.tau_years_from_days(dt_days)
        
        return self.decompose(
            F_t=F_t,
            F_next=F_next,
            K=K,
            tau_years=tau_years,
            r=r,
            dt=dt_years,
            V_t=V_t,
            Pi_t=Pi_t,
            sqrtw_t_kt=sqrtw_t_kt,
            d_sqrtw_dk_t_kt=d_sqrtw_dk_t_kt,
            d2_sqrtw_dk2_t_kt=d2_sqrtw_dk2_t_kt,
            iv1y_t_kt=iv1y_t_kt,
            d_iv1y_dk_t_kt=d_iv1y_dk_t_kt,
            d2_iv1y_dk2_t_kt=d2_iv1y_dk2_t_kt,
            delta_sqrtw_k0=delta_sqrtw_k0,
            delta_d_sqrtw_dk_k0=delta_d_sqrtw_dk_k0,
            delta_iv1y_k0=delta_iv1y_k0,
            delta_d_iv1y_dk_k0=delta_d_iv1y_dk_k0,
        )


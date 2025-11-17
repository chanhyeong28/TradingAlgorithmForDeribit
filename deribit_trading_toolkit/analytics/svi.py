"""
SVI (Stochastic Volatility Inspired) Parameterization Module

Provides SVI parameterization for implied volatility curve fitting.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.optimize import minimize, Bounds
import logging

logger = logging.getLogger(__name__)


# ----- SVI (raw) -----
# w(k) = a + b * { rho*(k - m) + sqrt( (k - m)^2 + sigma^2 ) }
def w_svi_raw(k, a, b, rho, m, sigma):
    """Calculate total variance using SVI formula"""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


@dataclass
class SVIParams:
    """SVI parameters"""
    a: float
    b: float
    rho: float
    m: float
    sigma: float


# ----- Helpers -----
def to_log_moneyness(K: np.ndarray, F: float) -> np.ndarray:
    """k = ln(K/F)"""
    return np.log(np.asarray(K, dtype=float) / float(F))


def to_total_variance(iv: np.ndarray, T: float) -> np.ndarray:
    """w = T * iv^2"""
    iv = np.asarray(iv, dtype=float)
    return float(T) * iv ** 2


def to_iv_from_w(w: np.ndarray, T: float) -> np.ndarray:
    """iv = sqrt(w / T)"""
    w = np.asarray(w, dtype=float)
    return np.sqrt(w / float(T))


# ----- Basic single-slice calibration -----
def fit_svi_basic(
    k: np.ndarray,
    w_mkt: np.ndarray,
    weights: Optional[np.ndarray] = None,
    x0: Optional[Tuple[float, float, float, float, float]] = None,
    bounds: Optional[Bounds] = None,
) -> SVIParams:
    """
    Basic SVI calibration for ONE expiry slice.
    Minimizes squared error on total variance without extra constraints.
    """
    k = np.asarray(k, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    if weights is None:
        weights = np.ones_like(w_mkt, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # Default initial guess (improved based on data characteristics)
    if x0 is None:
        # Better initial guess based on data
        w_min = np.nanmin(w_mkt)
        w_max = np.nanmax(w_mkt)
        w_median = np.nanmedian(w_mkt)
        
        # 'a' should be close to minimum total variance
        a0 = max(1e-6, 0.3 * w_min)
        
        # 'b' controls the slope - estimate from spread
        w_spread = w_max - w_min
        k_range = np.nanmax(k) - np.nanmin(k) if len(k) > 1 else 1.0
        b0 = min(2.0, max(0.05, w_spread / (k_range + 0.1)))
        
        # 'rho' controls skew - estimate from asymmetry
        k_median_idx = np.argmin(np.abs(k))
        if k_median_idx < len(w_mkt) - 1 and k_median_idx > 0:
            # Check if left side (puts) has higher variance than right side (calls)
            left_w = np.nanmean(w_mkt[:k_median_idx+1]) if k_median_idx > 0 else w_mkt[k_median_idx]
            right_w = np.nanmean(w_mkt[k_median_idx:]) if k_median_idx < len(w_mkt) - 1 else w_mkt[k_median_idx]
            rho0 = -0.3 if left_w > right_w else 0.0
        else:
            rho0 = -0.2
        
        # 'm' should be near the center of k range
        m0 = np.nanmedian(k) if len(k) > 0 else 0.0
        
        # 'sigma' controls the smoothness - start with reasonable value
        sigma0 = max(0.05, min(0.5, k_range / 4.0)) if k_range > 0 else 0.1
        
        x0 = np.array([a0, b0, rho0, m0, sigma0], dtype=float)

    # Simple bounds (keep parameters reasonable)
    if bounds is None:
        a_lo, a_hi = 1e-8, max(4.0, float(np.nanmax(w_mkt)) * 2.0)
        b_lo, b_hi = 1e-4, 5.0
        rho_lo, rho_hi = -0.999, 0.999
        m_lo, m_hi = min(k) - 2.0, max(k) + 2.0
        s_lo, s_hi = 1e-3, 5.0
        bounds = Bounds([a_lo, b_lo, rho_lo, m_lo, s_lo],
                        [a_hi, b_hi, rho_hi, m_hi, s_hi])

    def loss(theta):
        a, b, rho, m, sigma = theta
        w_fit = w_svi_raw(k, a, b, rho, m, sigma)
        r = (w_fit - w_mkt)
        return np.sum(weights * r * r)

    try:
        # Try multiple starting points for better fit
        best_res = None
        best_loss = float('inf')
        
        # Original starting point
        res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={'maxiter': 1000})
        if res.success and res.fun < best_loss:
            best_res = res
            best_loss = res.fun
        
        # Try alternative starting point (more symmetric)
        # Extract initial values from x0
        a0_init, b0_init, rho0_init, m0_init, sigma0_init = x0
        x0_alt = np.array([
            a0_init,
            b0_init * 0.5,  # Lower slope
            0.0,  # No skew
            m0_init,
            sigma0_init * 1.5  # Wider
        ], dtype=float)
        res_alt = minimize(loss, x0_alt, method="L-BFGS-B", bounds=bounds, options={'maxiter': 1000})
        if res_alt.success and res_alt.fun < best_loss:
            best_res = res_alt
            best_loss = res_alt.fun
        
        # Use best result
        if best_res is not None:
            a, b, rho, m, sigma = best_res.x
            if not best_res.success:
                logger.debug(f"SVI optimization warning: {best_res.message}, but using result")
            return SVIParams(a, b, rho, m, sigma)
        else:
            # Fallback to original result even if not fully successful
            a, b, rho, m, sigma = res.x
            logger.warning(f"SVI optimization did not converge: {res.message}, using result anyway")
            return SVIParams(a, b, rho, m, sigma)
    except Exception as e:
        logger.error(f"Error in SVI fitting: {e}")
        # Return default parameters based on data
        return SVIParams(a=max(1e-6, 0.3 * np.nanmin(w_mkt)), b=0.1, rho=-0.2, m=0.0, sigma=0.1)


# ----- Convenience wrappers -----
def calibrate_svi_from_iv(
    strikes: np.ndarray,
    iv: np.ndarray,
    T: float,
    forward: float,
    weights: Optional[np.ndarray] = None,
) -> SVIParams:
    """
    Given strikes, Black-Scholes IVs, maturity T (in years), and forward F,
    fit raw-SVI parameters for that expiry.
    """
    k = to_log_moneyness(strikes, forward)
    w = to_total_variance(iv, T)
    return fit_svi_basic(k, w, weights=weights)


def svi_iv_curve(
    strikes: np.ndarray,
    T: float,
    forward: float,
    params: SVIParams,
) -> np.ndarray:
    """Return fitted Black-Scholes IVs across given strikes (single expiry)."""
    k = to_log_moneyness(strikes, forward)
    w_fit = w_svi_raw(k, params.a, params.b, params.rho, params.m, params.sigma)
    return to_iv_from_w(w_fit, T)


def svi_iv_at_moneyness(
    moneyness: np.ndarray,
    T: float,
    params: SVIParams,
) -> np.ndarray:
    """Return fitted Black-Scholes IVs at given log-moneyness values."""
    w_fit = w_svi_raw(moneyness, params.a, params.b, params.rho, params.m, params.sigma)
    return to_iv_from_w(w_fit, T)


def svi_slope_at_moneyness(
    moneyness: float,
    T: float,
    params: SVIParams,
) -> float:
    """Calculate d(IV)/dk at given moneyness using SVI parameters."""
    k = moneyness
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    
    # d(w)/dk = b * (rho + (k - m) / sqrt((k - m)^2 + sigma^2))
    dw_dk = b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma ** 2))
    
    # d(IV)/dk = d(sqrt(w/T))/dk = (1/(2*sqrt(w*T))) * dw/dk
    w = w_svi_raw(k, a, b, rho, m, sigma)
    if w <= 0:
        return 0.0
    
    div_dk = (1.0 / (2.0 * np.sqrt(w * T))) * dw_dk
    return float(div_dk)


def svi_curvature_at_moneyness(
    moneyness: float,
    T: float,
    params: SVIParams,
) -> float:
    """Calculate d²(IV)/dk² at given moneyness using SVI parameters."""
    k = moneyness
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    
    # First derivative of w
    sqrt_term = np.sqrt((k - m) ** 2 + sigma ** 2)
    dw_dk = b * (rho + (k - m) / sqrt_term)
    
    # Second derivative of w
    d2w_dk2 = b * sigma ** 2 / (sqrt_term ** 3)
    
    # Second derivative of IV
    w = w_svi_raw(k, a, b, rho, m, sigma)
    if w <= 0:
        return 0.0
    
    sqrt_wT = np.sqrt(w * T)
    d2iv_dk2 = (1.0 / (2.0 * sqrt_wT)) * d2w_dk2 - (1.0 / (4.0 * w * sqrt_wT)) * (dw_dk ** 2)
    return float(d2iv_dk2)


def svi_sqrtw_at_moneyness(
    moneyness: float,
    T: float,
    params: SVIParams,
) -> float:
    """
    Calculate √w at given moneyness using SVI parameters.
    
    Args:
        moneyness: Log-moneyness k = ln(K/F)
        T: Time to expiry in years
        params: SVI parameters
        
    Returns:
        √w = sqrt(w) where w = T * iv^2
    """
    k = moneyness
    w = w_svi_raw(k, params.a, params.b, params.rho, params.m, params.sigma)
    if w <= 0:
        return 0.0
    return float(np.sqrt(w))


def svi_d_sqrtw_dk_at_moneyness(
    moneyness: float,
    T: float,
    params: SVIParams,
) -> float:
    """
    Calculate d(√w)/dk at given moneyness using SVI parameters.
    
    Args:
        moneyness: Log-moneyness k = ln(K/F)
        T: Time to expiry in years
        params: SVI parameters
        
    Returns:
        d(√w)/dk = (1/(2*sqrt(w))) * dw/dk
    """
    k = moneyness
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    
    # d(w)/dk = b * (rho + (k - m) / sqrt((k - m)^2 + sigma^2))
    sqrt_term = np.sqrt((k - m) ** 2 + sigma ** 2)
    dw_dk = b * (rho + (k - m) / sqrt_term)
    
    # d(√w)/dk = (1/(2*sqrt(w))) * dw/dk
    w = w_svi_raw(k, a, b, rho, m, sigma)
    if w <= 0:
        return 0.0
    
    d_sqrtw_dk = (1.0 / (2.0 * np.sqrt(w))) * dw_dk
    return float(d_sqrtw_dk)


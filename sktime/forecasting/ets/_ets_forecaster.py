"""
ETS (Error, Trend, Seasonality) Implementation

Exponential smoothing state space models following Hyndman et al. (2008).
Includes admissibility checking and robust parameter estimation.
"""

__all__ = ["ETSForecaster"]
__author__ = ["resul.akay@taf-society.org"]

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Literal, List
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, boxcox, jarque_bera, shapiro
from scipy.special import inv_boxcox
import warnings
import pandas as pd

from sktime.forecasting.base import BaseForecaster

# Type mappings
ERROR_TYPES = {"N": 0, "A": 1, "M": 2}
TREND_TYPES = {"N": 0, "A": 1, "M": 2}
SEASON_TYPES = {"N": 0, "A": 1, "M": 2}


def is_constant(y: NDArray[np.float64]) -> bool:
    """Check if series is constant"""
    return np.all(y == y[0])


def admissible(alpha: Optional[float],
               beta: Optional[float],
               gamma: Optional[float],
               phi: Optional[float],
               m: int) -> bool:
    """
    Check if ETS parameters satisfy admissibility conditions

    Implements constraints from Hyndman et al. (2008) to ensure finite forecast variance.
    Uses polynomial root checking for seasonal models.

    Parameters
    ----------
    alpha, beta, gamma, phi : float or None
        ETS smoothing parameters
    m : int
        Seasonal period

    Returns
    -------
    bool
        True if parameters are admissible
    """
    # Handle None and convert to float
    if phi is None:
        phi = 1.0

    if phi < 0 or phi > 1 + 1e-8:
        return False

    # Non-seasonal model (gamma is None)
    if gamma is None:
        if alpha is None:
            return True

        # Check alpha bounds
        if alpha < 1 - 1/phi or alpha > 1 + 1/phi:
            return False

        # Check beta if present
        if beta is not None:
            if beta < alpha * (phi - 1) or beta > (1 + phi) * (2 - alpha):
                return False

    # Seasonal model (gamma is not None and m > 1)
    elif m > 1:
        # Alpha must be present for seasonal models
        if alpha is None:
            return False

        # Default beta to 0 if None
        if beta is None:
            beta = 0.0

        # Check gamma bounds
        if gamma < max(1 - 1/phi - alpha, 0.0) or gamma > 1 + 1/phi - alpha:
            return False

        # Check alpha lower bound
        if alpha < 1 - 1/phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m):
            return False

        # Check beta lower bound
        if beta < -(1 - phi) * (gamma / m + alpha):
            return False

        # Polynomial root check for seasonal models
        # Construct characteristic polynomial
        a = phi * (1 - alpha - gamma)
        b = alpha + beta - alpha * phi + gamma - 1
        c_coef = alpha + beta - alpha * phi
        d = alpha + beta - phi

        # Build polynomial coefficients: [a, b, c, c, ..., c, d, 1]
        # where c appears (m-2) times
        P = np.array([a, b] + [c_coef] * max(0, m - 2) + [d, 1.0])

        # Find roots and check maximum absolute value
        try:
            poly_roots = np.roots(P)
            if np.max(np.abs(poly_roots)) > 1 + 1e-10:
                return False
        except:
            # If polynomial root finding fails, reject
            return False

    # Passed all tests
    return True


def check_param(alpha: Optional[float],
                beta: Optional[float],
                gamma: Optional[float],
                phi: Optional[float],
                lower: NDArray[np.float64],
                upper: NDArray[np.float64],
                bounds: str,
                m: int) -> bool:
    """
    Check if parameters satisfy both usual bounds and admissibility conditions

    Parameters
    ----------
    alpha, beta, gamma, phi : float or None
        Smoothing parameters
    lower, upper : array
        Parameter bounds [alpha, beta, gamma, phi]
    bounds : str
        Type of bounds: "usual", "admissible", or "both"
    m : int
        Seasonal period

    Returns
    -------
    bool
        True if parameters pass all checks
    """
    # Check usual bounds
    if bounds != "admissible":
        if alpha is not None and not np.isnan(alpha):
            if alpha < lower[0] or alpha > upper[0]:
                return False

        if beta is not None and not np.isnan(beta):
            if beta < lower[1] or beta > alpha or beta > upper[1]:
                return False

        if phi is not None and not np.isnan(phi):
            if phi < lower[3] or phi > upper[3]:
                return False

        if gamma is not None and not np.isnan(gamma):
            if gamma < lower[2] or gamma > 1 - alpha or gamma > upper[2]:
                return False

    # Check admissibility conditions
    if bounds != "usual":
        if not admissible(alpha, beta, gamma, phi, m):
            return False

    return True


@dataclass
class ETSConfig:
    """Configuration for ETS model"""
    error: Literal["A", "M"] = "A"
    trend: Literal["N", "A", "M"] = "N"
    season: Literal["N", "A", "M"] = "N"
    damped: bool = False
    m: int = 1  # seasonal period

    @property
    def error_code(self) -> int:
        return ERROR_TYPES[self.error]

    @property
    def trend_code(self) -> int:
        return TREND_TYPES[self.trend]

    @property
    def season_code(self) -> int:
        return SEASON_TYPES[self.season]

    @property
    def n_states(self) -> int:
        """Number of state variables"""
        n = 1  # level
        if self.trend != "N":
            n += 1  # trend
        if self.season != "N":
            n += self.m  # seasonal states
        return n


@dataclass
class ETSParams:
    """ETS smoothing parameters"""
    alpha: float = 0.1
    beta: float = 0.01
    gamma: float = 0.01
    phi: float = 0.98
    init_states: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def to_vector(self, config: ETSConfig) -> NDArray[np.float64]:
        """Convert parameters to optimization vector"""
        params = [self.alpha]
        if config.trend != "N":
            params.append(self.beta)
        if config.season != "N":
            params.append(self.gamma)
        if config.damped:
            params.append(self.phi)
        return np.concatenate([params, self.init_states])

    @staticmethod
    def from_vector(x: NDArray[np.float64], config: ETSConfig) -> 'ETSParams':
        """Create parameters from optimization vector"""
        idx = 0
        alpha = x[idx]; idx += 1
        beta = x[idx] if config.trend != "N" else 0.0
        if config.trend != "N":
            idx += 1
        gamma = x[idx] if config.season != "N" else 0.0
        if config.season != "N":
            idx += 1
        phi = x[idx] if config.damped else 1.0
        if config.damped:
            idx += 1
        init_states = x[idx:]
        return ETSParams(alpha, beta, gamma, phi, init_states)


@dataclass
class ETSModel:
    """Fitted ETS model"""
    config: ETSConfig
    params: ETSParams
    fitted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    states: NDArray[np.float64]
    loglik: float
    aic: float
    bic: float
    sigma2: float
    y_original: Optional[NDArray[np.float64]] = None  # Original data before transformation
    transform: Optional['BoxCoxTransform'] = None  # Transformation applied


@dataclass
class BoxCoxTransform:
    """Box-Cox transformation for variance stabilization"""
    lambda_param: float
    shift: float = 0.0  # Shift to make data positive

    @staticmethod
    def find_lambda(y: NDArray[np.float64], lambda_range: Tuple[float, float] = (-1, 2)) -> float:
        """Find optimal Box-Cox lambda using maximum likelihood"""
        if np.any(y <= 0):
            shift = np.abs(np.min(y)) + 1.0
            y_shifted = y + shift
        else:
            shift = 0.0
            y_shifted = y

        def neg_log_likelihood(lam):
            if abs(lam) < 1e-10:
                y_trans = np.log(y_shifted)
            else:
                y_trans = (y_shifted ** lam - 1) / lam
            return np.var(y_trans)

        result = minimize_scalar(neg_log_likelihood, bounds=lambda_range, method='bounded')
        return result.x

    def transform(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Box-Cox transformation"""
        y_shifted = y + self.shift
        if abs(self.lambda_param) < 1e-10:
            return np.log(y_shifted)
        else:
            return (y_shifted ** self.lambda_param - 1) / self.lambda_param

    def inverse_transform(self, y_trans: NDArray[np.float64],
                         bias_adjust: bool = False,
                         variance: Optional[float] = None) -> NDArray[np.float64]:
        """Inverse Box-Cox transformation with optional bias adjustment"""
        if abs(self.lambda_param) < 1e-10:
            y_back = np.exp(y_trans)
            if bias_adjust and variance is not None:
                # Bias adjustment for log transformation
                y_back *= np.exp(variance / 2)
        else:
            y_back = (self.lambda_param * y_trans + 1) ** (1 / self.lambda_param)
            if bias_adjust and variance is not None:
                # Taylor series bias adjustment for power transformations
                correction = (1 - self.lambda_param) * variance / (2 * y_back ** (2 * self.lambda_param))
                y_back += correction

        return y_back - self.shift


@njit(cache=True)
def _ets_step(l: float, b: float, s: NDArray[np.float64], y: float,
              m: int, error: int, trend: int, season: int,
              alpha: float, beta: float, gamma: float, phi: float) -> Tuple:
    """Single ETS update step (JIT compiled)

    Follows Hyndman et al. framework as implemented in Durbyn.jl
    """
    TOL = 1e-10

    # STEP 1: Compute one-step-ahead forecast component q (trend-adjusted level)
    if trend == 0:  # No trend
        q = l
        phib = 0.0
    elif trend == 1:  # Additive trend
        phib = phi * b
        q = l + phib
    else:  # Multiplicative trend (trend == 2)
        phib = b ** phi if b > 0 else 1.0
        q = l * phib if l > 0 else TOL

    # STEP 2: Compute one-step-ahead forecast yhat (add/multiply seasonality)
    # Use s[m-1] which is the oldest seasonal (m periods ago)
    if season == 0:  # No seasonality
        yhat = q
    elif season == 1:  # Additive seasonality
        yhat = q + s[m-1]
    else:  # Multiplicative seasonality
        yhat = q * s[m-1]

    if abs(yhat) < TOL:
        yhat = TOL

    # STEP 3: Compute error
    if error == 1:  # Additive error
        e = y - yhat
    else:  # Multiplicative error
        e = (y - yhat) / yhat

    # STEP 4: Deseasonalize observation to get p
    # Use s[m-1] which is the oldest seasonal (m periods ago)
    if season == 0:  # No seasonality
        p = y
    elif season == 1:  # Additive seasonality
        p = y - s[m-1]
    else:  # Multiplicative seasonality
        p = y / max(s[m-1], TOL)

    # STEP 5: Update level
    l_new = q + alpha * (p - q)

    # STEP 6: Update trend
    b_new = b
    if trend == 1:  # Additive trend
        r = l_new - l
        b_new = phib + (beta / alpha) * (r - phib)
    elif trend == 2:  # Multiplicative trend
        r = l_new / max(l, TOL)
        b_new = phib + (beta / alpha) * (r - phib)

    # STEP 7: Update seasonal
    # Compute new seasonal based on s[m-1] (oldest), then rotate
    s_new = s.copy()
    if season > 0:
        if season == 1:  # Additive seasonality
            t = y - q
        else:  # Multiplicative seasonality
            t = y / max(q, TOL)
        # New seasonal value based on oldest seasonal
        new_seasonal = s[m-1] + gamma * (t - s[m-1])
        # Rotate: new_seasonal goes to front, others shift right
        s_new[0] = new_seasonal
        s_new[1:m] = s[0:m-1]

    return l_new, b_new, s_new, yhat, e


@njit(cache=True)
def _ets_likelihood(y: NDArray[np.float64], init_states: NDArray[np.float64],
                    m: int, error: int, trend: int, season: int,
                    alpha: float, beta: float, gamma: float, phi: float) -> Tuple:
    """Compute ETS likelihood (JIT compiled)"""
    n = len(y)
    n_states = len(init_states)

    # Initialize states
    l = init_states[0]
    b = init_states[1] if trend > 0 else 0.0
    if season > 0:
        s = init_states[1 + (1 if trend > 0 else 0):].copy()
    else:
        s = np.zeros(max(m, 1))

    # Storage
    residuals = np.zeros(n)
    fitted = np.zeros(n)
    sum_e2 = 0.0
    sum_log_yhat = 0.0

    # Iterate through observations
    for i in range(n):
        l, b, s, yhat, e = _ets_step(l, b, s, y[i], m, error, trend, season,
                                      alpha, beta, gamma, phi)

        if yhat < -99998:  # Invalid forecast
            return np.inf, residuals, fitted, init_states

        fitted[i] = yhat
        residuals[i] = e

        # Accumulate components for log-likelihood
        sum_e2 += e * e
        if error == 2:  # Multiplicative error
            sum_log_yhat += np.log(max(abs(yhat), 1e-10))

    # Final log-likelihood
    if error == 1:  # Additive error
        loglik = n * np.log(sum_e2 / n)
    else:  # Multiplicative error
        loglik = n * np.log(sum_e2 / n) + 2 * sum_log_yhat

    # Build final state vector
    final_state = np.zeros(n_states)
    final_state[0] = l
    if trend > 0:
        final_state[1] = b
    if season > 0:
        offset = 1 + (1 if trend > 0 else 0)
        final_state[offset:offset + m] = s[:m]

    return loglik, residuals, fitted, final_state


def init_states(y: NDArray[np.float64], config: ETSConfig) -> NDArray[np.float64]:
    """Initialize ETS states using simple heuristics"""
    n = len(y)
    m = config.m

    states = []

    # Initialize level
    if config.season == "N":
        l0 = np.mean(y[:min(10, n)])
    else:
        l0 = np.mean(y[:min(2*m, n)])
    states.append(l0)

    # Initialize trend
    if config.trend != "N":
        if n >= 2:
            if config.trend == "A":
                b0 = (y[min(m, n-1)] - y[0]) / min(m, n-1)
            else:  # Multiplicative
                b0 = (y[min(m, n-1)] / max(y[0], 1e-10)) ** (1 / min(m, n-1))
        else:
            b0 = 1.0 if config.trend == "M" else 0.0
        states.append(b0)

    # Initialize seasonal
    if config.season != "N":
        if n >= 2 * m:
            # Simple seasonal averages
            seasonal = np.zeros(m)
            for i in range(m):
                seasonal[i] = np.mean(y[i::m][:2])

            if config.season == "A":
                seasonal -= np.mean(seasonal)
            else:  # Multiplicative
                seasonal /= np.mean(seasonal)
        else:
            seasonal = np.zeros(m) if config.season == "A" else np.ones(m)

        states.extend(seasonal)

    return np.array(states)


def get_bounds(config: ETSConfig) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get parameter bounds for optimization"""
    lower = [1e-4]  # alpha
    upper = [0.9999]

    if config.trend != "N":
        lower.append(1e-4)  # beta
        upper.append(0.9999)

    if config.season != "N":
        lower.append(1e-4)  # gamma
        upper.append(0.9999)

    if config.damped:
        lower.append(0.8)  # phi
        upper.append(0.98)

    # Add bounds for initial states (quite loose)
    n_states = config.n_states
    lower.extend([-1e6] * n_states)
    upper.extend([1e6] * n_states)

    return np.array(lower), np.array(upper)


def ets(y: NDArray[np.float64],
        m: int = 1,
        model: str = "ANN",
        damped: bool = False,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        phi: Optional[float] = None,
        lambda_param: Optional[float] = None,
        lambda_auto: bool = False,
        bias_adjust: bool = False,
        bounds: str = "both") -> ETSModel:
    """
    Fit ETS model using scipy optimization

    Parameters
    ----------
    y : array_like
        Time series data
    m : int
        Seasonal period
    model : str
        Three-letter model specification (e.g., "ANN", "AAA", "MAM")
        First letter: Error (A=Additive, M=Multiplicative)
        Second letter: Trend (N=None, A=Additive, M=Multiplicative)
        Third letter: Season (N=None, A=Additive, M=Multiplicative)
    damped : bool
        Whether to use damped trend
    alpha, beta, gamma, phi : float, optional
        Fixed parameter values (if None, will be estimated)
    lambda_param : float, optional
        Box-Cox transformation parameter. If None, no transformation
    lambda_auto : bool
        If True, automatically select optimal lambda
    bias_adjust : bool
        Apply bias adjustment when back-transforming forecasts
    bounds : str
        Parameter bounds type: "usual", "admissible", or "both" (default)

    Returns
    -------
    ETSModel
        Fitted model
    """
    y = np.asarray(y, dtype=np.float64)
    y_original = y.copy()
    n = len(y)

    # Handle constant series (matches Julia behavior)
    if is_constant(y):
        warnings.warn("Series is constant. Fitting simple exponential smoothing with alpha=0.99999")
        # Return simple ETS(A,N,N) with high alpha (essentially returns constant forecast)
        config = ETSConfig(error="A", trend="N", season="N", damped=False, m=1)
        alpha_const = 0.99999
        l0 = y[0]

        # Simple fitted values (all equal to the constant)
        fitted = np.full(n, y[0])
        residuals = np.zeros(n)

        return ETSModel(
            config=config,
            params=ETSParams(alpha=alpha_const, beta=0.0, gamma=0.0, phi=1.0,
                           init_states=np.array([l0])),
            fitted=fitted,
            residuals=residuals,
            states=np.array([l0]),
            loglik=0.0,  # Perfect fit
            aic=2.0,  # Minimal AIC
            bic=2.0,
            sigma2=0.0,
            y_original=y_original,
            transform=None
        )

    # Basic validation only (Julia/R handle even 1 observation)
    if n < 1:
        raise ValueError(f"Need at least 1 observation to fit ETS model, got {n}")

    # Parse model specification
    if len(model) != 3:
        raise ValueError(f"Model must be 3 characters (e.g., 'AAN', 'MAM'), got '{model}'")

    # Check for seasonal models with insufficient data (matches R behavior)
    # R silently drops seasonality when n < m, we should at least warn/error
    season_type = model[2]
    if season_type != "N" and m > 1 and n < m:
        raise ValueError(
            f"Cannot fit seasonal model: need at least m={m} observations for seasonal period, but got n={n}. "
            f"R would drop seasonality and fit {model[:2]}N instead. "
            f"Either provide more data or use a non-seasonal model."
        )

    # Apply Box-Cox transformation if requested
    transform = None
    if lambda_auto:
        shift = np.abs(np.min(y)) + 1.0 if np.any(y <= 0) else 0.0
        lambda_opt = BoxCoxTransform.find_lambda(y)
        transform = BoxCoxTransform(lambda_opt, shift)
        y = transform.transform(y)
    elif lambda_param is not None:
        shift = np.abs(np.min(y)) + 1.0 if np.any(y <= 0) else 0.0
        transform = BoxCoxTransform(lambda_param, shift)
        y = transform.transform(y)

    # Parse model specification
    if len(model) != 3:
        raise ValueError("Model must be 3 characters (e.g., 'ANN', 'AAA')")

    config = ETSConfig(
        error=model[0],
        trend=model[1],
        season=model[2],
        damped=damped,
        m=m
    )

    # Initialize parameters
    init_state_vec = init_states(y, config)
    init_params = ETSParams(
        alpha=alpha if alpha is not None else 0.1,
        beta=beta if beta is not None else 0.01,
        gamma=gamma if gamma is not None else 0.01,
        phi=phi if phi is not None else 0.98,
        init_states=init_state_vec
    )

    # Get bounds
    lower, upper = get_bounds(config)

    # Objective function with bounds enforcement and admissibility checking
    def objective(x):
        # Check bounds - return high penalty if violated
        if np.any(x < lower) or np.any(x > upper):
            return 1e10

        params = ETSParams.from_vector(x, config)

        # Extract smoothing parameters for checking
        alpha_check = params.alpha
        beta_check = params.beta if config.trend != "N" else None
        gamma_check = params.gamma if config.season != "N" else None
        phi_check = params.phi if damped else None

        # Check parameter admissibility
        if not check_param(alpha_check, beta_check, gamma_check, phi_check,
                          lower, upper, bounds, config.m):
            return 1e10

        loglik, _, _, _ = _ets_likelihood(
            y, params.init_states,
            config.m, config.error_code, config.trend_code, config.season_code,
            params.alpha, params.beta, params.gamma, params.phi
        )

        # Return high penalty for invalid likelihood
        if np.isnan(loglik) or np.isinf(loglik):
            return 1e10

        return loglik

    # Optimize using Nelder-Mead (matches Julia/R implementation)
    x0 = init_params.to_vector(config)
    bounds = list(zip(lower, upper))

    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True
        }
    )

    # Extract fitted parameters
    fitted_params = ETSParams.from_vector(result.x, config)

    # Compute final likelihood, residuals, and final states
    loglik, residuals, fitted_vals, final_states = _ets_likelihood(
        y, fitted_params.init_states,
        config.m, config.error_code, config.trend_code, config.season_code,
        fitted_params.alpha, fitted_params.beta, fitted_params.gamma, fitted_params.phi
    )

    # Compute information criteria
    n_params = len(result.x)
    aic = 2 * loglik + 2 * n_params
    bic = 2 * loglik + n_params * np.log(n)
    sigma2 = np.sum(residuals ** 2) / (n - n_params)

    # Back-transform fitted values if transformation was applied
    fitted_original = fitted_vals
    if transform is not None:
        fitted_original = transform.inverse_transform(fitted_vals, bias_adjust, sigma2)

    return ETSModel(
        config=config,
        params=fitted_params,
        fitted=fitted_original,
        residuals=y_original - fitted_original,
        states=final_states,  # Use FINAL states, not initial!
        loglik=-0.5 * loglik,
        aic=aic,
        bic=bic,
        sigma2=sigma2,
        y_original=y_original,
        transform=transform
    )


@njit(cache=True)
def _forecast_ets(l: float, b: float, s: NDArray[np.float64],
                  h: int, m: int, trend: int, season: int, phi: float) -> NDArray[np.float64]:
    """Generate h-step ahead forecasts"""
    forecasts = np.zeros(h)
    phi_sum = phi

    for i in range(h):
        # Base forecast
        if trend == 0:  # No trend
            fc = l
        elif trend == 1:  # Additive
            fc = l + phi_sum * b
        else:  # Multiplicative
            fc = l * (b ** phi_sum) if b > 0 else 0.0

        # Add seasonal - use reverse order to match Julia
        # Seasonal array: s[0]=newest, s[1]=1 period ago, ..., s[m-1]=oldest
        # For forecast i steps ahead, we need seasonal from (m-1-i) periods ago
        s_idx = (m - 1 - i) % m if m > 0 else 0
        if season == 1:  # Additive
            fc += s[s_idx]
        elif season == 2:  # Multiplicative
            fc *= s[s_idx]

        forecasts[i] = fc

        # Update phi sum for damping
        if i < h - 1:
            phi_sum += phi ** (i + 2)

    return forecasts


def _compute_prediction_variance(model: ETSModel, h: int) -> NDArray[np.float64]:
    """
    Compute analytical prediction variance for ETS models

    Uses analytical formulas for Class 1 and Class 2 models (Hyndman et al. 2008)
    Falls back to simulation for complex models
    """
    sigma = model.sigma2
    m = model.config.m
    error = model.config.error
    trend = model.config.trend
    season = model.config.season
    damped = model.config.damped

    alpha = model.params.alpha
    beta = model.params.beta
    gamma = model.params.gamma
    phi = model.params.phi

    steps = np.arange(1, h + 1)

    # Class 1: Additive error models
    if error == "A":
        if trend == "N" and season == "N":
            # ANN
            var = sigma * (1 + alpha**2 * (steps - 1))

        elif trend == "A" and season == "N" and not damped:
            # AAN
            var = sigma * (1 + (steps - 1) * (alpha**2 + alpha * beta * steps +
                          (1/6) * beta**2 * steps * (2 * steps - 1)))

        elif trend == "A" and season == "N" and damped:
            # AAdN
            exp1 = (beta * phi * steps) / (1 - phi)**2
            exp2 = 2 * alpha * (1 - phi) + beta * phi
            exp3 = (beta * phi * (1 - phi**steps)) / ((1 - phi)**2 * (1 - phi**2))
            exp4 = 2 * alpha * (1 - phi**2) + beta * phi * (1 + 2 * phi - phi**steps)
            var = sigma * (1 + alpha**2 * (steps - 1) + exp1 * exp2 - exp3 * exp4)

        elif trend == "N" and season == "A":
            # ANA
            hm = np.floor((steps - 1) / m)
            var = sigma * (1 + alpha**2 * (steps - 1) + gamma * hm * (2 * alpha + gamma))

        elif trend == "A" and season == "A" and not damped:
            # AAA
            hm = np.floor((steps - 1) / m)
            exp1 = alpha**2 + alpha * beta * steps + (1/6) * beta**2 * steps * (2 * steps - 1)
            exp2 = 2 * alpha + gamma + beta * m * (hm + 1)
            var = sigma * (1 + (steps - 1) * exp1 + gamma * hm * exp2)

        else:
            # Fallback to simulation for other additive models
            var = None

    else:
        # Class 2/3: Multiplicative error - use simulation
        var = None

    return var


def forecast_ets(model: ETSModel, h: int = 10, bias_adjust: bool = True,
                level: Optional[List[float]] = None) -> Dict[str, NDArray[np.float64]]:
    """
    Generate forecasts from fitted ETS model with optional prediction intervals

    Parameters
    ----------
    model : ETSModel
        Fitted model
    h : int
        Forecast horizon
    bias_adjust : bool
        Apply bias adjustment if Box-Cox transformation was used
    level : list of float, optional
        Confidence levels for prediction intervals (e.g., [80, 95])
        If None, only return point forecasts

    Returns
    -------
    dict
        Dictionary with:
        - 'mean': Point forecasts
        - 'lower_XX': Lower bounds for XX% intervals (if level provided)
        - 'upper_XX': Upper bounds for XX% intervals (if level provided)
    """
    # Extract final states
    l = model.states[0]
    b = model.states[1] if model.config.trend != "N" else 0.0

    if model.config.season != "N":
        s_start = 1 + (1 if model.config.trend != "N" else 0)
        s = model.states[s_start:]
    else:
        s = np.zeros(1)

    forecasts = _forecast_ets(
        l, b, s, h,
        model.config.m,
        model.config.trend_code,
        model.config.season_code,
        model.params.phi
    )

    # Back-transform if needed
    if model.transform is not None:
        forecasts = model.transform.inverse_transform(forecasts, bias_adjust, model.sigma2)

    result = {'mean': forecasts}

    # Compute prediction intervals if requested
    if level is not None:
        # Check if sigma2 is valid for prediction intervals
        if model.sigma2 <= 0:
            import warnings
            warnings.warn(
                f"Cannot compute prediction intervals: model has invalid residual variance "
                f"(sigma2={model.sigma2:.2e}). This usually means the model is overfit or "
                f"there is insufficient data. Returning point forecasts only.",
                UserWarning
            )
            # Return only point forecasts, skip prediction intervals
            return result

        # Try analytical variance first
        var = _compute_prediction_variance(model, h)

        if var is not None:
            # Use analytical formulas
            for lv in level:
                z = norm.ppf(0.5 + lv / 200)
                std = np.sqrt(var)
                result[f'lower_{int(lv)}'] = forecasts - z * std
                result[f'upper_{int(lv)}'] = forecasts + z * std
        else:
            # Fall back to simulation for complex models
            try:
                simulations = simulate_ets(model, h=h, n_sim=1000)
                for lv in level:
                    result[f'lower_{int(lv)}'] = np.percentile(simulations, 50 - lv/2, axis=0)
                    result[f'upper_{int(lv)}'] = np.percentile(simulations, 50 + lv/2, axis=0)
            except ValueError as e:
                # If simulation fails, warn and return point forecasts only
                import warnings
                warnings.warn(
                    f"Cannot compute prediction intervals via simulation: {str(e)}. "
                    f"Returning point forecasts only.",
                    UserWarning
                )

    return result


def simulate_ets(model: ETSModel, h: int = 10, n_sim: int = 1000) -> NDArray[np.float64]:
    """Simulate future paths from ETS model"""
    # Validate sigma2
    if model.sigma2 <= 0:
        raise ValueError(
            f"Cannot simulate: model has invalid residual variance (sigma2={model.sigma2:.2e}). "
            f"This usually means the model is overfit or there is insufficient data."
        )

    simulations = np.zeros((n_sim, h))

    for i in range(n_sim):
        # Generate random errors
        if model.config.error == "A":
            errors = norm.rvs(loc=0, scale=np.sqrt(model.sigma2), size=h)
        else:  # Multiplicative
            errors = norm.rvs(loc=0, scale=np.sqrt(model.sigma2), size=h)

        # Simulate forward
        l = model.states[0]
        b = model.states[1] if model.config.trend != "N" else 0.0

        if model.config.season != "N":
            s_start = 1 + (1 if model.config.trend != "N" else 0)
            s = model.states[s_start:].copy()
        else:
            s = np.zeros(max(model.config.m, 1))

        for t in range(h):
            # Forecast
            fc = _forecast_ets(l, b, s, 1, model.config.m,
                              model.config.trend_code, model.config.season_code,
                              model.params.phi)[0]

            # Add error
            if model.config.error == "A":
                y_new = fc + errors[t]
            else:
                y_new = fc * (1 + errors[t])

            simulations[i, t] = y_new

            # Update states
            l, b, s, _, _ = _ets_step(
                l, b, s, y_new,
                model.config.m,
                model.config.error_code,
                model.config.trend_code,
                model.config.season_code,
                model.params.alpha,
                model.params.beta,
                model.params.gamma,
                model.params.phi
            )

    return simulations


def auto_ets(y: NDArray[np.float64],
             m: int = 1,
             seasonal: bool = True,
             trend: Optional[bool] = None,
             damped: Optional[bool] = None,
             ic: Literal["aic", "aicc", "bic"] = "aicc",
             allow_multiplicative: bool = True,
             allow_multiplicative_trend: bool = False,
             lambda_auto: bool = False,
             max_models: Optional[int] = None,
             verbose: bool = False) -> ETSModel:
    """
    Automatic ETS model selection

    Parameters
    ----------
    y : array_like
        Time series data
    m : int
        Seasonal period
    seasonal : bool
        Allow seasonal models
    trend : bool, optional
        If None, try both with and without trend. If True, only trending models. If False, only non-trending models
    damped : bool, optional
        If None, try both damped and non-damped. If True/False, only try that variant
    ic : str
        Information criterion for model selection ("aic", "aicc", "bic")
    allow_multiplicative : bool
        Allow multiplicative error and season models (default True)
    allow_multiplicative_trend : bool
        Allow multiplicative trend models (default False, matches Julia/R)
        More conservative as multiplicative trend can be unstable
    lambda_auto : bool
        Automatically select Box-Cox transformation
    max_models : int, optional
        Maximum number of models to try (None = try all)
    verbose : bool
        Print progress

    Returns
    -------
    ETSModel
        Best model according to information criterion
    """
    # Basic validation only (Julia/R handle even very small datasets)
    n = len(y)
    if n < 1:
        raise ValueError(f"Need at least 1 observation, got {n}")

    # Detect trend if auto-detection requested
    has_trend = False
    if trend is None:
        # Simple trend test: compare first half vs second half means
        # If data is trending, second half should be significantly different
        mid = len(y) // 2
        first_half_mean = np.mean(y[:mid])
        second_half_mean = np.mean(y[mid:])
        # Use a threshold of 10% change
        pct_change = abs(second_half_mean - first_half_mean) / first_half_mean
        has_trend = pct_change > 0.10
        if verbose and has_trend:
            print(f"Trend detected: {pct_change:.1%} change from first to second half")

    # Generate candidate models
    error_types = ["A", "M"] if allow_multiplicative else ["A"]

    # Trend component logic
    if trend is None:
        if has_trend:
            # Trend detected - prefer models with trend
            trend_types = ["A"]
            if allow_multiplicative_trend:
                trend_types.append("M")
            # Also try non-trending to be safe
            trend_types.append("N")
        else:
            # No trend detected - try both but prefer simpler
            trend_types = ["N", "A"]
            if allow_multiplicative_trend:
                trend_types.append("M")
    elif trend:
        # User wants trend
        trend_types = ["A"]
        if allow_multiplicative_trend:
            trend_types.append("M")
    else:
        # User explicitly doesn't want trend (trend=False)
        trend_types = ["N"]

    # Seasonal component logic
    if m == 1:
        # Non-seasonal data - force no seasonality
        season_types = ["N"]
    elif not seasonal:
        # User explicitly doesn't want seasonal models
        season_types = ["N"]
    elif n < m:
        # Insufficient data for seasonal model (matches R behavior)
        # R drops seasonality when n < m to avoid overfitting
        season_types = ["N"]
        if verbose:
            print(f"Insufficient data for seasonality (n={n} < m={m}), trying non-seasonal models only")
    else:
        # Seasonal data (m > 1) and seasonal=True
        # Only try seasonal models - don't include non-seasonal
        if allow_multiplicative:
            season_types = ["A", "M"]
        else:
            season_types = ["A"]

    damped_opts = [True, False] if damped is None else [damped]

    # Build model list
    models_to_try = []
    for e in error_types:
        for t in trend_types:
            for s in season_types:
                for d in damped_opts:
                    # Skip invalid combinations
                    if t == "N" and d:  # Can't have damped with no trend
                        continue

                    # Restrict unstable combinations (matches Julia/R behavior)
                    # 1. Additive error with multiplicative components is unstable
                    if e == "A" and (t == "M" or s == "M"):
                        continue
                    # 2. MMA is unstable (multiplicative error + trend with additive season)
                    if e == "M" and t == "M" and s == "A":
                        continue

                    models_to_try.append((f"{e}{t}{s}", d))

    # Limit number of models if requested
    if max_models is not None and len(models_to_try) > max_models:
        # Prioritize seasonal models when m > 1, otherwise simpler models
        if m > 1:
            # Prioritize: seasonal models > non-seasonal, then by complexity
            models_to_try = sorted(models_to_try, key=lambda x: (x[0][2] == 'N', x[1], x[0].count('M')))
        else:
            # Prioritize simpler models for non-seasonal data
            models_to_try = sorted(models_to_try, key=lambda x: (x[1], x[0].count('M')))
        models_to_try = models_to_try[:max_models]

    if verbose:
        print(f"Trying {len(models_to_try)} models...")

    # Helper function to format model name properly
    def format_model_name(model_spec: str, damped: bool) -> str:
        """Format model name with proper ETS notation (e.g., MAdM instead of MAMd)"""
        if damped and model_spec[1] != "N":  # Damped only applies to trend
            # Insert 'd' after trend component: MAM + damped → MAdM
            return f"{model_spec[0]}{model_spec[1]}d{model_spec[2]}"
        return model_spec

    # Fit all models and track best
    best_model = None
    best_ic_value = np.inf
    best_ic_original = np.inf
    results = []

    for model_spec, damped_flag in models_to_try:
        try:
            model = ets(y, m=m, model=model_spec, damped=damped_flag, lambda_auto=lambda_auto, bounds="both")

            # Get IC value
            if ic == "aic":
                ic_value = model.aic
            elif ic == "aicc":
                # Compute AICc
                n = len(y)
                k = (1 + (model.config.trend != "N") + (model.config.season != "N") +
                     damped_flag + model.config.n_states)
                ic_value = model.aic + (2 * k * (k + 1)) / (n - k - 1)
            else:  # bic
                ic_value = model.bic

            # Apply penalty to non-trending models when trend is detected
            ic_value_adj = ic_value
            model_name = format_model_name(model_spec, damped_flag)

            if has_trend and model.config.trend == "N":
                # Penalize non-trending models when trend is detected
                # Use a modest penalty that can be overcome if fit is much better
                ic_value_adj = ic_value + 5.0
                if verbose:
                    print(f"  {model_name:5s}: {ic.upper()}={ic_value:.2f} (penalized: {ic_value_adj:.2f})")
            elif verbose:
                print(f"  {model_name:5s}: {ic.upper()}={ic_value:.2f}")

            results.append((model_spec, damped_flag, ic_value, model))

            if ic_value_adj < best_ic_value:
                best_ic_value = ic_value_adj
                best_ic_original = ic_value
                best_model = model

        except Exception as e:
            if verbose:
                model_name = format_model_name(model_spec, damped_flag)
                print(f"  {model_name:5s}: Failed ({str(e)})")
            continue

    if best_model is None:
        raise ValueError("No model could be fitted successfully")

    if verbose:
        best_model_name = format_model_name(
            f"{best_model.config.error}{best_model.config.trend}{best_model.config.season}",
            best_model.config.damped
        )
        print(f"\nBest model: {best_model_name} ({ic.upper()}={best_ic_original:.2f})")

    return best_model


def residual_diagnostics(model: ETSModel) -> Dict[str, any]:
    """
    Compute residual diagnostics for ETS model

    Parameters
    ----------
    model : ETSModel
        Fitted ETS model

    Returns
    -------
    dict
        Dictionary with diagnostic statistics:
        - mean: mean of residuals (should be ~0)
        - std: standard deviation of residuals
        - ljung_box_p: Ljung-Box test p-value (>0.05 suggests no autocorrelation)
        - jarque_bera_p: Jarque-Bera test p-value (>0.05 suggests normality)
        - shapiro_p: Shapiro-Wilk test p-value (>0.05 suggests normality)
        - acf: Autocorrelation function (first 10 lags)
    """
    residuals = model.residuals
    n = len(residuals)

    # Basic statistics
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals, ddof=1)

    # Normality tests
    try:
        jb_stat, jb_p = jarque_bera(residuals)
    except:
        jb_stat, jb_p = np.nan, np.nan

    try:
        if n >= 3:
            shapiro_stat, shapiro_p = shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
    except:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Autocorrelation function (simple implementation)
    max_lag = min(10, n // 4)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    residuals_centered = residuals - mean_resid
    c0 = np.sum(residuals_centered ** 2) / n

    for lag in range(1, max_lag + 1):
        c_lag = np.sum(residuals_centered[:-lag] * residuals_centered[lag:]) / n
        acf[lag] = c_lag / c0

    # Ljung-Box test (approximate)
    lb_stat = n * (n + 2) * np.sum(acf[1:max_lag+1] ** 2 / (n - np.arange(1, max_lag+1)))
    from scipy.stats import chi2
    lb_p = 1 - chi2.cdf(lb_stat, max_lag)

    return {
        "mean": mean_resid,
        "std": std_resid,
        "mae": np.mean(np.abs(residuals)),
        "rmse": np.sqrt(np.mean(residuals ** 2)),
        "mape": np.mean(np.abs(residuals / model.y_original)) * 100 if model.y_original is not None else np.nan,
        "ljung_box_stat": lb_stat,
        "ljung_box_p": lb_p,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p": jb_p,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "acf": acf,
    }


# ============================================================================
# ETSForecaster class (sktime interface)
# ============================================================================


class ETSForecaster(BaseForecaster):
    """ETS (Error, Trend, Seasonality) forecaster for sktime.

    Exponential smoothing state space models following Hyndman et al. (2008).
    This class provides a sktime interface to the ETS forecasting methodology,
    which includes various combinations of error, trend, and seasonal components.

    The ETS framework encompasses a family of models including:
    - Simple exponential smoothing (ANN)
    - Holt's linear method (AAN)
    - Damped trend methods (AAdN)
    - Holt-Winters seasonal methods (AAA, MAA, etc.)

    Parameters
    ----------
    m : int, default=1
        Seasonal period. Use 1 for non-seasonal data, 12 for monthly data
        with yearly seasonality, 4 for quarterly data, etc.
    model : str, default="ZZZ"
        Three-letter model specification or "ZZZ" for automatic selection:
        - First letter: Error type (A=Additive, M=Multiplicative, Z=Auto)
        - Second letter: Trend type (N=None, A=Additive, M=Multiplicative, Z=Auto)
        - Third letter: Season type (N=None, A=Additive, M=Multiplicative, Z=Auto)
        Examples: "ANN" (simple exponential smoothing), "AAN" (Holt's linear),
        "AAA" (additive Holt-Winters), "MAM" (multiplicative Holt-Winters)
    damped : bool or None, default=None
        Whether to use damped trend. Only applicable when trend is not "N".
        If None and model selection is automatic, both damped and non-damped
        variants will be tried.
    alpha : float or None, default=None
        Level smoothing parameter (0 < alpha < 1). If None, will be estimated.
    beta : float or None, default=None
        Trend smoothing parameter (0 < beta < alpha). If None, will be estimated.
    gamma : float or None, default=None
        Seasonal smoothing parameter (0 < gamma < 1-alpha). If None, will be estimated.
    phi : float or None, default=None
        Damping parameter (0.8 < phi < 1). If None, will be estimated.
    lambda_param : float or None, default=None
        Box-Cox transformation parameter. If None, no transformation is applied.
    lambda_auto : bool, default=False
        If True, automatically select optimal Box-Cox transformation parameter.
    bias_adjust : bool, default=False
        If True, apply bias adjustment when back-transforming forecasts.
    bounds : str, default="both"
        Parameter bounds type: "usual", "admissible", or "both".
        "admissible" uses constraints from Hyndman et al. (2008) to ensure
        finite forecast variance.

    Attributes
    ----------
    model_ : ETSModel
        Fitted ETS model containing parameters, states, and diagnostics.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.ets import ETSForecaster
    >>> y = load_airline()
    >>> forecaster = ETSForecaster(m=12, model="AAA")  # Additive Holt-Winters
    >>> forecaster.fit(y)
    ETSForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    >>> # Automatic model selection
    >>> forecaster_auto = ETSForecaster(m=12, model="ZZZ")
    >>> forecaster_auto.fit(y)
    >>> y_pred_auto = forecaster_auto.predict(fh=[1, 2, 3])

    References
    ----------
    .. [1] Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
           Forecasting with exponential smoothing: the state space approach,
           Springer-Verlag.
    """

    _tags = {
        # packaging info
        "authors": ["resul.akay@taf-society.org"],
        "maintainers": ["resul.akay@taf-society.org"],
        # estimator type
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "capability:exogenous": False,
        "capability:missing_values": False,
        "capability:pred_int": True,
        "capability:pred_var": False,
        "capability:insample": False,
    }

    def __init__(
        self,
        m=1,
        model="ZZZ",
        damped=None,
        alpha=None,
        beta=None,
        gamma=None,
        phi=None,
        lambda_param=None,
        lambda_auto=False,
        bias_adjust=False,
        bounds="both",
    ):
        self.m = m
        self.model = model
        self.damped = damped
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        self.lambda_param = lambda_param
        self.lambda_auto = lambda_auto
        self.bias_adjust = bias_adjust
        self.bounds = bounds
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored, not supported by ETS).
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon (not required in fit).

        Returns
        -------
        self : reference to self
        """
        # Convert to numpy array for fitting
        y_np = y.values

        # Determine if we need automatic model selection
        if self.model == "ZZZ" or "Z" in self.model:
            # Automatic model selection
            self.model_ = auto_ets(
                y_np,
                m=self.m,
                seasonal=(self.model[2] != "N") if len(self.model) == 3 else True,
                trend=None if self.model[1] == "Z" else (self.model[1] != "N"),
                damped=self.damped,
                lambda_auto=self.lambda_auto,
                allow_multiplicative=True,
                allow_multiplicative_trend=False,
                verbose=False,
            )
        else:
            # Fit specified model
            self.model_ = ets(
                y_np,
                m=self.m,
                model=self.model,
                damped=self.damped if self.damped is not None else False,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                phi=self.phi,
                lambda_param=self.lambda_param,
                lambda_auto=self.lambda_auto,
                bias_adjust=self.bias_adjust,
                bounds=self.bounds,
            )

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored, not supported by ETS).

        Returns
        -------
        y_pred : pd.Series
            Point predictions for the forecast horizon.
        """
        # Get the forecast horizon as integer steps
        fh_int = fh.to_relative(self.cutoff)
        h = int(fh_int.max())

        # Generate forecasts
        forecast_dict = forecast_ets(
            self.model_, h=h, bias_adjust=self.bias_adjust, level=None
        )

        # Extract mean predictions for the requested horizon
        # Convert fh_int to numpy array for indexing (0-indexed)
        fh_idx = np.asarray(fh_int) - 1
        y_pred_values = forecast_dict["mean"][fh_idx]

        # Create index for predictions
        fh_abs = fh.to_absolute(self.cutoff)
        index = fh_abs.to_pandas()

        # Return as pandas Series
        return pd.Series(y_pred_values, index=index, name=self._y.name)

    def _predict_interval(self, fh, X=None, coverage=0.90):
        """Compute prediction intervals.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored, not supported by ETS).
        coverage : float or list of float, optional (default=0.90)
            Nominal coverage(s) of the prediction intervals.

        Returns
        -------
        pred_int : pd.DataFrame
            Prediction intervals with columns for each coverage level.
        """
        # Convert coverage to level (percentage)
        if not isinstance(coverage, list):
            coverage = [coverage]

        # Convert to percentage
        level = [c * 100 for c in coverage]

        # Get the forecast horizon as integer steps
        fh_int = fh.to_relative(self.cutoff)
        h = int(fh_int.max())

        # Generate forecasts with intervals
        forecast_dict = forecast_ets(
            self.model_, h=h, bias_adjust=self.bias_adjust, level=level
        )

        # Create index for predictions
        fh_abs = fh.to_absolute(self.cutoff)
        index = fh_abs.to_pandas()

        # Extract intervals for the requested horizon
        # Convert fh_int to numpy array for indexing (0-indexed)
        fh_idx = np.asarray(fh_int) - 1

        pred_int_dict = {}
        for cov in coverage:
            lv = int(cov * 100)
            lower_key = f"lower_{lv}"
            upper_key = f"upper_{lv}"

            # Check if intervals were computed
            if lower_key in forecast_dict and upper_key in forecast_dict:
                lower_values = forecast_dict[lower_key][fh_idx]
                upper_values = forecast_dict[upper_key][fh_idx]

                pred_int_dict[(cov, "lower")] = pd.Series(
                    lower_values, index=index, name=self._y.name
                )
                pred_int_dict[(cov, "upper")] = pd.Series(
                    upper_values, index=index, name=self._y.name
                )

        # Create MultiIndex DataFrame
        if pred_int_dict:
            pred_int = pd.DataFrame(pred_int_dict)
            pred_int.columns.names = ["Coverage", "Interval"]
            return pred_int
        else:
            # Return empty DataFrame if intervals couldn't be computed
            warnings.warn(
                "Prediction intervals could not be computed. "
                "This may be due to insufficient data or model issues.",
                UserWarning,
            )
            return pd.DataFrame(index=index)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute quantile forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored, not supported by ETS).
        alpha : list of float, optional (default=[0.05, 0.95])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
            second level being the values of alpha passed to the function.
        """
        if alpha is None:
            alpha = [0.05, 0.95]

        # Convert alpha to coverage levels
        coverage_dict = {}
        for a in alpha:
            if a < 0.5:
                coverage = 1 - 2 * a
                coverage_dict[a] = (coverage, "lower")
            else:
                coverage = 2 * (a - 0.5)
                coverage_dict[a] = (coverage, "upper")

        # Get unique coverage levels
        unique_coverage = list(set([v[0] for v in coverage_dict.values()]))

        # Get prediction intervals
        pred_int = self._predict_interval(fh, X=X, coverage=unique_coverage)

        # If intervals couldn't be computed, return empty DataFrame
        if pred_int.empty:
            fh_abs = fh.to_absolute(self.cutoff)
            index = fh_abs.to_pandas()
            var_name = self._y.name if self._y.name is not None else 0
            return pd.DataFrame(
                index=index,
                columns=pd.MultiIndex.from_product(
                    [[var_name], alpha], names=["variable", "alpha"]
                ),
            )

        # Extract quantiles from intervals
        quantile_dict = {}
        for a in alpha:
            coverage, bound = coverage_dict[a]
            if (coverage, bound) in pred_int.columns:
                quantile_dict[a] = pred_int[(coverage, bound)]
            else:
                # If this specific interval wasn't computed, fill with NaN
                quantile_dict[a] = pd.Series(
                    np.nan, index=pred_int.index, name=self._y.name
                )

        # Create DataFrame with variable name as first level
        quantiles = pd.DataFrame(quantile_dict)
        quantiles.columns.name = "alpha"

        # Add variable name as first level
        var_name = self._y.name if self._y.name is not None else 0
        quantiles.columns = pd.MultiIndex.from_product(
            [[var_name], quantiles.columns], names=["variable", "alpha"]
        )

        return quantiles

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        params1 = {}  # Default simple exponential smoothing
        params2 = {"m": 12, "model": "AAN"}  # Holt's linear method
        params3 = {"m": 12, "model": "AAA"}  # Additive Holt-Winters
        return [params1, params2, params3]

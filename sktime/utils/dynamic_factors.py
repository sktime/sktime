"""Utilities for dynamic factor models.

NumPy-only helpers for VAR/AR and state-space computations used by dynamic
factor forecasters. Kept independent of specific estimator implementations.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "solve_regularized_ols",
    "estimate_var",
    "get_idio",
    "get_transition_params",
    "build_ddfm_state_space",
    "forecast_factors_ar1",
]


def solve_regularized_ols(
    X: np.ndarray, Y: np.ndarray, regularization: float = 1e-6
) -> np.ndarray:
    r"""Solve ridge-regularized OLS: \((X'X + λI)^{-1} X'Y\) (NumPy only)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    xtx = X.T @ X
    xtx = xtx + regularization * np.eye(xtx.shape[0], dtype=np.float64)
    xty = X.T @ Y
    return np.linalg.solve(xtx, xty)


def estimate_var(factors: np.ndarray, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(1) dynamics for factors."""
    if order != 1:
        raise ValueError("Only VAR(1) is supported")
    factors = np.asarray(factors, dtype=np.float64)
    T, m = factors.shape
    if T < 2:
        A = np.eye(m, dtype=np.float64)
        Q = np.eye(m, dtype=np.float64) * 0.01
        return A, Q

    Y = factors[1:, :]
    X = factors[:-1, :]
    A = solve_regularized_ols(X, Y, regularization=1e-6).T

    residuals = Y - X @ A.T
    Q = np.cov(residuals.T, bias=False)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    Q = 0.5 * (Q + Q.T)
    return A, Q


def get_idio(eps: np.ndarray, idx_no_missings: np.ndarray, min_obs: int = 5):
    """Compute AR(1) coefficients and moments for idiosyncratic residuals."""
    eps = np.asarray(eps, dtype=np.float64)
    idx_no_missings = np.asarray(idx_no_missings, dtype=bool)
    Phi = np.zeros((eps.shape[1], eps.shape[1]), dtype=np.float64)
    mu_eps = np.zeros(eps.shape[1], dtype=np.float64)
    std_eps = np.zeros(eps.shape[1], dtype=np.float64)

    for j in range(eps.shape[1]):
        to_select = idx_no_missings[:, j]
        to_select = np.hstack((np.array([False]), to_select[:-1] & to_select[1:]))
        if np.sum(to_select) >= min_obs:
            this_eps = eps[to_select, j]
        else:
            Phi[j, j] = 0.0
            mu_eps[j] = 0.0
            std_eps[j] = 1.0
            continue
        mu_eps[j] = float(np.mean(this_eps))
        std_eps[j] = float(np.std(this_eps)) if float(np.std(this_eps)) > 0 else 1.0
        cov1_eps = np.cov(this_eps[1:], this_eps[:-1])[0][1]
        Phi[j, j] = float(cov1_eps / (std_eps[j] ** 2))
    return Phi, mu_eps, std_eps


def get_transition_params(
    f_t: np.ndarray, eps_t: np.ndarray, bool_no_miss: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute joint VAR(1) factor and AR(1) idiosyncratic transition parameters.

    Returns
    -------
    A : np.ndarray
        Block transition matrix (factors and idiosyncratic).
    Q : np.ndarray
        Process noise covariance.
    mu_0 : np.ndarray
        Initial state mean.
    Sigma_0 : np.ndarray
        Initial state covariance.
    x_t : np.ndarray
        Stacked state [f_t, eps_t] over time.
    """
    f_t = np.asarray(f_t, dtype=np.float64)
    eps_t = np.asarray(eps_t, dtype=np.float64)
    bool_no_miss = np.asarray(bool_no_miss, dtype=bool)

    f_past_safe = np.nan_to_num(f_t[:-1, :], nan=0.0, posinf=0.0, neginf=0.0)
    f_next_safe = np.nan_to_num(f_t[1:, :], nan=0.0, posinf=0.0, neginf=0.0)

    XtX = f_past_safe.T @ f_past_safe
    Xty = f_past_safe.T @ f_next_safe

    try:
        cond_num = np.linalg.cond(XtX)
        ridge_scale = max(1e-6, min(1e-4, 1e-6 * (1 + np.log10(max(cond_num, 1)))))
    except (np.linalg.LinAlgError, ValueError):
        ridge_scale = 1e-6

    ridge = ridge_scale * np.eye(XtX.shape[0], dtype=XtX.dtype)
    try:
        A_f = (np.linalg.pinv(XtX + ridge) @ Xty).T
        if not np.all(np.isfinite(A_f)):
            raise ValueError("pinv produced non-finite values")
    except (np.linalg.LinAlgError, ValueError):
        try:
            A_t, _residuals, _rank, _s = np.linalg.lstsq(XtX + ridge, Xty, rcond=None)
            A_f = A_t.T
        except np.linalg.LinAlgError:
            A_f = np.eye(f_past_safe.shape[1], dtype=f_past_safe.dtype)

    Phi, _, _ = get_idio(eps_t, bool_no_miss)

    x_t = np.vstack((f_t.T, eps_t.T))
    A = np.vstack(
        (
            np.hstack((A_f, np.zeros((A_f.shape[0], eps_t.shape[1])))),
            np.hstack((np.zeros((eps_t.shape[1], A_f.shape[1])), Phi)),
        )
    )

    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    Q = np.diag(np.diag(np.cov(w_t)))
    mu_0 = np.mean(x_t, axis=1)
    Sigma_0 = np.cov(x_t)
    Sigma_0[: A_f.shape[1], A_f.shape[1] :] = 0
    Sigma_0[A_f.shape[1] :, : A_f.shape[1]] = 0
    Sigma_0[A_f.shape[1] :, A_f.shape[1] :] = np.diag(
        np.diag(Sigma_0[A_f.shape[1] :, A_f.shape[1] :])
    )
    return A, Q, mu_0, Sigma_0, x_t


def build_ddfm_state_space(
    *,
    factors: np.ndarray,
    eps: np.ndarray,
    decoder_weight: np.ndarray,
    observed_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build state-space transition and observation matrices (F, H) from factors and decoder weights."""
    factors = np.asarray(factors, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    decoder_weight = np.asarray(decoder_weight, dtype=np.float64)
    observed_y = np.asarray(observed_y, dtype=bool)

    factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
    eps = np.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)

    num_factors = factors.shape[1]
    H = decoder_weight[:, :num_factors]
    F_full, _Q_full, _mu0_full, _Sigma0_full, _x_t = get_transition_params(
        factors, eps, bool_no_miss=observed_y
    )
    F = F_full[:num_factors, :num_factors]
    return F, H


def forecast_factors_ar1(z_last: np.ndarray, F: np.ndarray, horizon: int) -> np.ndarray:
    r"""Roll factors forward with VAR(1) dynamics: \(z_{t+1} = z_t F^T\) (row-vector convention)."""
    z_last = np.asarray(z_last, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    out = np.empty((horizon, z_last.shape[0]), dtype=np.float64)
    z = z_last
    for i in range(horizon):
        z = z @ F.T
        out[i] = z
    return out


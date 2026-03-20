# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""State-space and factor-model helpers for representation-learning forecasters.

Used by forecasters that predict from a latent state (e.g. factor models,
autoencoder-based models). NumPy-only; no torch dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sktime.utils.dynamic_factors import build_ddfm_state_space, estimate_var


@dataclass
class _StateSpaceParams:
    """Linear state-space parameters: transition F, observation H and intercept b.

    Generic container for any representation-learning forecaster that predicts
    from a latent state with linear dynamics and linear observation. Used by
    factor models (e.g. DDFM), not tied to a specific estimator.

    Attributes
    ----------
    F : np.ndarray of shape (n_factors, n_factors)
        Transition matrix for the latent state.
    H : np.ndarray of shape (n_targets, n_factors)
        Observation matrix from latent state to targets.
    b : np.ndarray of shape (n_targets,)
        Observation intercept.
    """

    F: np.ndarray  # (n_factors, n_factors)
    H: np.ndarray  # (n_targets, n_factors)
    b: np.ndarray  # (n_targets,)


@dataclass
class _FittedFactorModel:
    """Fitted state of a factor model: latent factors and state-space params.

    Used by DeepDynamicFactor and any other factor-model forecaster that
    shares this representation (factors array + linear state-space F, H, b).
    """

    factors: np.ndarray  # (n_timepoints, n_factors)
    state_space: _StateSpaceParams


def _linear_decoder_ddfm_dynamics(
    *,
    factors: np.ndarray,
    eps: np.ndarray,
    decoder_weight: np.ndarray,
    observed_y: np.ndarray,
) -> _StateSpaceParams:
    """Factor-dynamics backend: DDFM-style state-space from linear decoder.

    Parameters
    ----------
    factors : np.ndarray of shape (n_timepoints, n_factors)
        Latent factors.
    eps : np.ndarray of shape (n_timepoints, n_targets)
        Idiosyncratic residuals.
    decoder_weight : np.ndarray of shape (n_targets, n_factors) or (n_targets, n_factors+1)
        Decoder weight matrix (linear layer); only first n_factors columns used for H.
    observed_y : np.ndarray of shape (n_timepoints, n_targets), bool
        Mask of observed target values.

    Returns
    -------
    _StateSpaceParams
        F (factor transition), H (observation matrix), b (zero intercept).
    """
    F, H = build_ddfm_state_space(
        factors=factors,
        eps=eps,
        decoder_weight=decoder_weight,
        observed_y=observed_y,
    )
    b = np.zeros((H.shape[0],), dtype=np.float64)
    return _StateSpaceParams(F=F, H=H, b=b)


def _generic_var1_dynamics(
    *,
    factors: np.ndarray,
    y_scaled: np.ndarray,
) -> _StateSpaceParams:
    """Factor-dynamics backend: VAR(1) for factors and ridge regression for H, b.

    Parameters
    ----------
    factors : np.ndarray of shape (n_timepoints, n_factors)
        Latent factors.
    y_scaled : np.ndarray of shape (n_timepoints, n_targets)
        Scaled target values.

    Returns
    -------
    _StateSpaceParams
        F from VAR(1) on factors, H and b from ridge regression of y_scaled on factors.
    """
    F, _Q = estimate_var(factors, order=1)
    H, b = _estimate_H_b_from_factors(factors, y_scaled)
    return _StateSpaceParams(F=F, H=H, b=b)


def _estimate_H_b_from_factors(
    factors: np.ndarray, y_scaled: np.ndarray, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate linear mapping from factors to scaled targets via ridge regression.

    Returns
    -------
    H : np.ndarray of shape (n_targets, n_factors)
        Observation matrix.
    b : np.ndarray of shape (n_targets,)
        Observation intercept.
    """
    Z = np.asarray(factors, dtype=np.float64)
    Y = np.asarray(y_scaled, dtype=np.float64)
    if Z.ndim != 2 or Y.ndim != 2:
        raise ValueError("factors and y_scaled must be 2D arrays")
    if Z.shape[0] != Y.shape[0]:
        raise ValueError("factors and y_scaled must have same n_timepoints")

    ones = np.ones((Z.shape[0], 1), dtype=np.float64)
    Z1 = np.concatenate([Z, ones], axis=1)  # (T, r+1)
    xtx = Z1.T @ Z1
    xtx = xtx + ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xty = Z1.T @ Y
    beta = np.linalg.solve(xtx, xty)  # (r+1, n_targets)
    W = beta[:-1, :]  # (r, n_targets)
    b = beta[-1, :]  # (n_targets,)
    H = W.T  # (n_targets, r)
    return H, b

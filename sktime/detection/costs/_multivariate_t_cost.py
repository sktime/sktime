# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Multivariate T distribution likelihood cost."""

__author__ = ["johannvk"]

import numpy as np
import pandas as pd

from sktime.detection._costs._base import BaseCost
from sktime.detection._costs._multivariate_gaussian_cost import (
    _gaussian_cost_fixed_params,
    _gaussian_cost_mle_params,
)
from sktime.detection._utils import (
    CovType,
    MeanType,
    check_cov,
    check_in_interval,
    check_larger_than,
    check_mean,
    col_median,
)

# ---------------------------------------------------------------------------
# Helper math functions — pure numpy, no numba
# ---------------------------------------------------------------------------


def _log_gamma(x):
    """Log of the gamma function (uses scipy when available)."""
    try:
        from scipy.special import gammaln

        return gammaln(x)
    except ImportError:
        # Stirling approximation fallback
        return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi)


def _digamma(x):
    """Digamma function (psi)."""
    try:
        from scipy.special import digamma

        return digamma(x)
    except ImportError:
        # Simple series expansion
        result = 0.0
        while x < 6:
            result -= 1.0 / x
            x += 1
        result += np.log(x) - 1 / (2 * x)
        return result


def _trigamma(x):
    """Trigamma function (first derivative of digamma)."""
    try:
        from scipy.special import polygamma

        return polygamma(1, x)
    except ImportError:
        result = 0.0
        while x < 6:
            result += 1.0 / (x * x)
            x += 1
        result += 1 / x + 1 / (2 * x * x)
        return result


def _kurtosis_columnwise(X):
    """Compute excess kurtosis for each column of X."""
    mean = np.mean(X, axis=0)
    m2 = np.mean((X - mean) ** 2, axis=0)
    m4 = np.mean((X - mean) ** 4, axis=0)
    safe_m2 = np.where(m2 > 0, m2, 1.0)
    return np.where(m2 > 0, m4 / (safe_m2**2) - 3.0, 0.0)


# ---------------------------------------------------------------------------
# MLE scale matrix estimation (fixed-point iteration, pure numpy)
# ---------------------------------------------------------------------------


def _initial_scale_matrix_estimate(centered_samples, dof, loo_index=-1):
    """Estimate the scale matrix from centered samples."""
    n, p = centered_samples.shape
    sq_norms = np.sum(centered_samples * centered_samples, axis=1)
    nonzero = sq_norms > 1e-6

    weights = np.ones(n)
    weights[nonzero] = 1.0 / sq_norms[nonzero]
    weighted = centered_samples * weights[:, np.newaxis]
    scale_est = weighted.T @ centered_samples

    if loo_index >= 0:
        loo = centered_samples[loo_index, :].reshape(-1, 1)
        scale_est -= weights[loo_index] * loo @ loo.T
        scale_est /= n - 1
    else:
        scale_est /= n

    # Estimate trace from log-norm distribution
    log_sq_norms = np.log(sq_norms[nonzero])
    z_bar = log_sq_norms.mean()
    log_alpha = z_bar - np.log(dof) + _digamma(0.5 * dof) - _digamma(p / 2.0)
    trace_est = p * np.exp(log_alpha)

    current_trace = np.trace(scale_est)
    if current_trace > 0:
        scale_est *= trace_est / current_trace

    return scale_est


def _scale_matrix_fp_iteration(scale_matrix, dof, centered_samples, loo_index=-1):
    """One fixed-point iteration for the MLE scale matrix."""
    n, p = centered_samples.shape
    inv_scale = np.linalg.solve(scale_matrix, np.eye(p))
    mahal_sq = np.sum((centered_samples @ inv_scale) * centered_samples, axis=1)
    sample_weights = (p + dof) / (dof + mahal_sq)
    weighted = centered_samples * sample_weights[:, np.newaxis]
    new_scale = weighted.T @ centered_samples

    if loo_index >= 0:
        loo = centered_samples[loo_index, :].reshape(-1, 1)
        new_scale -= sample_weights[loo_index] * loo @ loo.T
        new_scale /= n - 1
    else:
        new_scale /= n

    return new_scale


def _solve_mle_scale_matrix(
    initial_scale, centered_samples, dof, max_iter, abs_tol, rel_tol, loo_index=-1
):
    """Fixed-point iteration to compute MLE scale matrix."""
    scale = initial_scale.copy()
    for iteration in range(1, max_iter + 1):
        new_scale = _scale_matrix_fp_iteration(scale, dof, centered_samples, loo_index)
        residual = np.linalg.norm(new_scale - scale)
        relative = residual / max(np.linalg.norm(scale), 1e-12)
        scale[:, :] = new_scale
        if residual < abs_tol or relative < rel_tol:
            break
    else:
        raise RuntimeError(f"MLE scale matrix: max iterations ({max_iter}) reached.")
    return scale


def _mle_scale_matrix(centered_samples, dof, abs_tol, rel_tol, max_iter, loo_index=-1):
    """Full MLE scale matrix estimate."""
    initial = _initial_scale_matrix_estimate(centered_samples, dof, loo_index)
    return _solve_mle_scale_matrix(
        initial, centered_samples, dof, max_iter, abs_tol, rel_tol, loo_index
    )


# ---------------------------------------------------------------------------
# Multivariate-T log-likelihood
# ---------------------------------------------------------------------------


def _mv_t_log_likelihood(scale_matrix, centered_samples, dof):
    """Calculate the log likelihood of a multivariate t-distribution."""
    p = centered_samples.shape[1]
    sign, log_det = np.linalg.slogdet(scale_matrix)
    if sign <= 0:
        return np.nan
    inv_scale = np.linalg.solve(scale_matrix, np.eye(p))
    return _mv_t_ll_fixed(centered_samples, dof, inv_scale, log_det)


def _mv_t_ll_fixed(centered_samples, dof, inv_scale, log_det_scale):
    """Log-likelihood with pre-computed inverse and log det."""
    n, p = centered_samples.shape
    mahal_sq = np.sum((centered_samples @ inv_scale) * centered_samples, axis=1)
    exponent = 0.5 * (dof + p)
    A = _log_gamma(exponent)
    B = _log_gamma(0.5 * dof)
    C = 0.5 * p * np.log(dof * np.pi)
    D = 0.5 * log_det_scale
    norm_contribution = n * (A - B - C - D)
    sample_contributions = -exponent * np.log1p(mahal_sq / dof)
    return norm_contribution + sample_contributions.sum()


# ---------------------------------------------------------------------------
# Degrees of freedom estimation
# ---------------------------------------------------------------------------


def _isotropic_dof_estimate(centered_samples, infinite_dof_threshold):
    """Isotropic DOF estimate from log-norm variance."""
    p = centered_samples.shape[1]
    sq_norms = np.sum(centered_samples * centered_samples, axis=1)
    log_sq_var = np.log(sq_norms[sq_norms > 1e-12]).var()
    b = log_sq_var - _trigamma(p / 2.0)
    inf_threshold = (2 * infinite_dof_threshold + 4) / (infinite_dof_threshold**2)
    if b < inf_threshold:
        return np.inf
    return (1 + np.sqrt(1 + 4 * b)) / b


def _kurtosis_dof_estimate(centered_samples, infinite_dof_threshold):
    """Kurtosis-based DOF estimate."""
    k = _kurtosis_columnwise(centered_samples).mean() / 3.0
    threshold = 2.0 / (infinite_dof_threshold - 4.0)
    if k < threshold:
        return np.inf
    return (2.0 / k) + 4.0


def _iterative_dof_estimate(
    centered_samples,
    initial_dof,
    infinite_dof_threshold,
    mle_scale_abs_tol,
    mle_scale_rel_tol,
    mle_scale_max_iter,
    dof_abs_tol=0.1,
    dof_rel_tol=0.05,
    dof_max_iter=10,
):
    """Estimate DOF iteratively."""
    n = centered_samples.shape[0]
    if initial_dof > infinite_dof_threshold:
        return np.inf

    inf_nu = infinite_dof_threshold / (infinite_dof_threshold - 2.0)
    sample_cov = (centered_samples.T @ centered_samples) / n
    mle_scale = _mle_scale_matrix(
        centered_samples,
        initial_dof,
        mle_scale_abs_tol,
        mle_scale_rel_tol,
        mle_scale_max_iter,
    )
    dof = initial_dof
    for _ in range(dof_max_iter):
        nu_i = np.trace(sample_cov) / np.trace(mle_scale)
        if nu_i < inf_nu:
            return np.inf
        old_dof = dof
        dof = 2 * nu_i / max(nu_i - 1, 1e-3)
        mle_scale = _solve_mle_scale_matrix(
            mle_scale,
            centered_samples,
            dof,
            mle_scale_max_iter,
            mle_scale_abs_tol,
            mle_scale_rel_tol,
        )
        if (
            abs(dof - old_dof) < dof_abs_tol
            or abs(dof - old_dof) / old_dof < dof_rel_tol
        ):
            break
    return dof


def _estimate_dof(
    X,
    infinite_dof_threshold,
    refine_dof_threshold,
    mle_scale_abs_tol,
    mle_scale_rel_tol,
    mle_scale_max_iter,
):
    """Estimate DOF via isotropic, kurtosis, then iterative refinement."""
    centered = X - col_median(X)
    iso = _isotropic_dof_estimate(centered, infinite_dof_threshold)
    kurt = _kurtosis_dof_estimate(centered, infinite_dof_threshold)

    if np.isfinite(iso) and np.isfinite(kurt):
        initial = np.sqrt(iso * kurt)
    elif np.isfinite(iso):
        initial = iso
    elif np.isfinite(kurt):
        initial = kurt
    else:
        initial = infinite_dof_threshold / 2.0

    return _iterative_dof_estimate(
        centered,
        initial,
        infinite_dof_threshold,
        mle_scale_abs_tol,
        mle_scale_rel_tol,
        mle_scale_max_iter,
    )


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def _mv_t_cost_mle_params(starts, ends, X, dof, abs_tol, rel_tol, max_iter):
    """Multivariate-T twice neg-log-likelihood cost at MLE params."""
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment = X[start:end]
        medians = col_median(segment)
        centered = segment - medians
        mle_scale = _mle_scale_matrix(centered, dof, abs_tol, rel_tol, max_iter)
        ll = _mv_t_log_likelihood(mle_scale, centered, dof)
        costs[i, 0] = -2.0 * ll
    return costs


def _mv_t_cost_fixed_params(starts, ends, X, mean, inv_scale, log_det_scale, dof):
    """Multivariate-T twice neg-log-likelihood cost at fixed params."""
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        centered = X[start:end] - mean
        ll = _mv_t_ll_fixed(centered, dof, inv_scale, log_det_scale)
        costs[i, 0] = -2.0 * ll
    return costs


class MultivariateTCost(BaseCost):
    """Multivariate T twice negative log likelihood cost.

    Parameters
    ----------
    param : 2-tuple of float or np.ndarray, or None (default=None)
        Fixed mean and scale matrix for the cost calculation.
    fixed_dof : float, optional (default=None)
        Fixed degrees of freedom.  If None, DOF is estimated from data.
    refine_dof_threshold : int, optional (default=100)
        Below this sample count, DOF is refined with a leave-one-out method.
    infinite_dof_threshold : float, optional (default=50.0)
        Above this DOF value, the T is approximated by a Gaussian.
    mle_scale_abs_tol : float, optional (default=1e-2)
        Absolute tolerance for MLE scale matrix convergence.
    mle_scale_rel_tol : float, optional (default=1e-2)
        Relative tolerance for MLE scale matrix convergence.
    mle_scale_max_iter : int, optional (default=100)
        Maximum iterations for MLE scale matrix estimation.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
        "is_aggregated": True,
    }

    def __init__(
        self,
        param: tuple[MeanType, CovType] | None = None,
        fixed_dof=None,
        refine_dof_threshold=100,
        infinite_dof_threshold=50.0,
        mle_scale_abs_tol=1e-2,
        mle_scale_rel_tol=1e-2,
        mle_scale_max_iter=100,
    ):
        self.fixed_dof = fixed_dof
        self.refine_dof_threshold = refine_dof_threshold
        self.infinite_dof_threshold = infinite_dof_threshold
        self.mle_scale_abs_tol = mle_scale_abs_tol
        self.mle_scale_rel_tol = mle_scale_rel_tol
        self.mle_scale_max_iter = mle_scale_max_iter

        check_in_interval(
            interval=pd.Interval(0, np.inf),
            value=self.fixed_dof,
            name="fixed_dof",
            allow_none=True,
        )
        check_larger_than(
            min_value=0.0,
            value=self.infinite_dof_threshold,
            name="infinite_dof_threshold",
        )
        check_larger_than(
            min_value=0,
            value=self.refine_dof_threshold,
            name="refine_dof_threshold",
            allow_none=True,
        )

        super().__init__(param)

    def _get_dof(self, X):
        """Get degrees of freedom — fixed or estimated from data."""
        if self.fixed_dof is not None:
            return self.fixed_dof
        return _estimate_dof(
            X,
            self.infinite_dof_threshold,
            self.refine_dof_threshold,
            self.mle_scale_abs_tol,
            self.mle_scale_rel_tol,
            self.mle_scale_max_iter,
        )

    def _evaluate_optim_param(self, X, starts, ends):
        dof = self._get_dof(X)
        if np.isposinf(dof):
            return _gaussian_cost_mle_params(starts, ends, X)
        return _mv_t_cost_mle_params(
            starts,
            ends,
            X,
            dof,
            self.mle_scale_abs_tol,
            self.mle_scale_rel_tol,
            self.mle_scale_max_iter,
        )

    def _evaluate_fixed_param(self, X, starts, ends, param):
        mean, scale = param
        inv_scale = np.linalg.inv(scale)
        _, log_det = np.linalg.slogdet(scale)
        dof = self._get_dof(X)
        if np.isposinf(dof):
            return _gaussian_cost_fixed_params(
                starts, ends, X, mean, log_det, inv_scale
            )
        return _mv_t_cost_fixed_params(starts, ends, X, mean, inv_scale, log_det, dof)

    def _check_fixed_param(self, param, X):
        if not isinstance(param, tuple) or len(param) != 2:
            raise ValueError("Fixed parameters must be (mean, scale_matrix).")
        mean, cov = param
        mean = check_mean(mean, X)
        cov = check_cov(cov, X, force_float=True)
        return mean, cov

    @property
    def min_size(self):
        return None

    def get_model_size(self, p):
        return 1 + p + p * (p + 1) // 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"param": None},
            {"param": (0.0, 1.0)},
            {"param": (np.zeros(1), np.eye(1))},
        ]

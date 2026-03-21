"""Multivariate T distribution likelihood cost."""

__author__ = ["johannvk"]
__all__ = ["MultivariateTCost"]

import numpy as np
import pandas as pd

from ..utils.numba import njit, numba_available, prange
from ..utils.numba.stats import (
    col_median,
    digamma,
    kurtosis,
    log_gamma,
    trigamma,
)
from ..utils.validation.parameters import check_in_interval, check_larger_than
from ._multivariate_gaussian_cost import (
    gaussian_cost_fixed_params,
    gaussian_cost_mle_params,
)
from ._utils import CovType, MeanType, check_cov, check_mean
from .base import BaseCost


@njit
def _estimate_scale_matrix_trace(
    centered_sample_squared_norms: np.ndarray,
    non_zero_norm_mask: np.ndarray,
    sample_dimension: int,
    dof: float,
    loo_index=-1,
):
    """Estimate the trace of the MLE multivariate T covariance matrix.

    Using an isotropic estimate of the multivariate T distribution,
    the authors of [1]_ developed an estimate of the trace of the
    maximum likelihood estimate scale matrix, using the squared
    norm of centered samples.

    Parameters
    ----------
    centered_sample_squared_norms : np.ndarray
        The squared norms of centered samples from a multivariate t-distribution.
    non_zero_norm_mask : np.ndarray
        A boolean mask indicating which squared norms are non-zero.
    sample_dimension : int
        The dimension of the samples.
    dof : float
        The degrees of freedom of the multivariate t-distribution.
    loo_index : int, optional (default=-1)
        The index of the leave-one-out sample. If -1, the full covariance matrix
        is estimated.

    References
    ----------
    .. [1] Aeschliman, Chad, & Johnny Park, & Avinash C. Kak. (2009). A Novel
       Parameter Estimation Algorithm for the Multivariate T-Distribution and Its
       Application to Computer Vision. In Computer Vision - ECCV 2010, 594-607.
       Berlin, Heidelberg: Springer

    Returns
    -------
    float
        The estimated trace of the MLE covariance matrix.
    """
    if loo_index >= 0:
        # Zero out the leave-one-out sample squared norm.
        centered_sample_squared_norms[loo_index] = 0.0
        non_zero_norm_mask[loo_index] = False

    z_bar = np.log(centered_sample_squared_norms[non_zero_norm_mask]).mean()
    log_alpha = (
        z_bar - np.log(dof) + digamma(0.5 * dof) - digamma(sample_dimension / 2.0)
    )
    return sample_dimension * np.exp(log_alpha)


@njit
def _initial_scale_matrix_estimate(
    centered_samples: np.ndarray,
    dof: float,
    loo_index: int = -1,
):
    """Estimate the scale matrix given centered samples and degrees of freedom.

    The direction of the scale matrix is estimated from standardized,
    centered samples, and the trace of the scale matrix is estimated
    using the squared norms of the centered samples, as described in [1]_.

    Parameters
    ----------
    centered_samples : np.ndarray
        Centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.
    loo_index : int, optional (default=-1)
        The index of the leave-one-out sample. If -1, the full covariance matrix
        is estimated.

    References
    ----------
    .. [1] Aeschliman, Chad, & Johnny Park, & Avinash C. Kak. (2009). A Novel
       Parameter Estimation Algorithm for the Multivariate T-Distribution and Its
       Application to Computer Vision. In Computer Vision - ECCV 2010, 594-607.
       Berlin, Heidelberg: Springer

    Returns
    -------
    np.ndarray
        The initial estimate of the scale matrix
    """
    num_samples, sample_dimension = centered_samples.shape

    # Estimate direction of the scale matrix from standardized, centered samples:
    centered_sample_squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    non_zero_norm_mask = centered_sample_squared_norms > 1.0e-6

    centered_sample_weights = np.ones(num_samples)
    centered_sample_weights[non_zero_norm_mask] *= (
        1.0 / centered_sample_squared_norms[non_zero_norm_mask]
    )
    weighted_samples = centered_samples * centered_sample_weights[:, np.newaxis]
    mle_scale_estimate = weighted_samples.T @ centered_samples

    if loo_index >= 0:
        # Subtract contribution from the leave-one-out sample:
        loo_sample = centered_samples[loo_index, :].reshape(-1, 1)
        mle_scale_estimate -= (
            centered_sample_weights[loo_index] * loo_sample @ loo_sample.T
        )

        # Scale by the number of samples minus the leave-one-out sample:
        mle_scale_estimate /= num_samples - 1
    else:
        mle_scale_estimate /= num_samples

    scale_trace_estimate = _estimate_scale_matrix_trace(
        centered_sample_squared_norms=centered_sample_squared_norms,
        non_zero_norm_mask=non_zero_norm_mask,
        sample_dimension=sample_dimension,
        dof=dof,
        loo_index=loo_index,
    )
    mle_scale_estimate *= scale_trace_estimate / np.trace(mle_scale_estimate)

    return mle_scale_estimate


@njit
def _scale_matrix_fixed_point_iteration(
    scale_matrix: np.ndarray,
    dof: float,
    centered_samples: np.ndarray,
    loo_index: int = -1,
):
    """Compute a multivariate T MLE scale matrix fixed point iteration.

    Parameters
    ----------
    scale_matrix : np.ndarray
        The current estimate of the scale matrix.
    dof : float
        The degrees of freedom of the multivariate t-distribution.
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    loo_index : int, optional (default=-1)
        The index of the leave-one-out sample. If -1, the full covariance matrix
        is estimated.

    Returns
    -------
    np.ndarray
        The updated estimate of the scale matrix.
    """
    num_samples, sample_dim = centered_samples.shape

    inverse_scale_matrix = np.linalg.solve(scale_matrix, np.eye(sample_dim))
    mahalanobis_squared_distances = np.sum(
        (centered_samples @ inverse_scale_matrix) * centered_samples, axis=1
    )

    sample_weights = (sample_dim + dof) / (dof + mahalanobis_squared_distances)
    weighted_samples = centered_samples * sample_weights[:, np.newaxis]

    reconstructed_scale_matrix = weighted_samples.T @ centered_samples
    if loo_index >= 0:
        # Subtract the leave-one-out sample contribution:
        loo_sample = centered_samples[loo_index, :].reshape(-1, 1)
        reconstructed_scale_matrix -= (
            sample_weights[loo_index] * loo_sample @ loo_sample.T
        )

        # Scale by the number of samples minus the leave-one-out sample:
        reconstructed_scale_matrix /= num_samples - 1
    else:
        reconstructed_scale_matrix /= num_samples

    return reconstructed_scale_matrix


@njit
def _solve_for_mle_scale_matrix(
    initial_scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    dof: float,
    max_iter: int,
    abs_tol: float,
    rel_tol: float,
    loo_index: int = -1,
) -> np.ndarray:
    """Perform fixed point iterations to compute the MLE scale matrix.

    Parameters
    ----------
    initial_scale_matrix : np.ndarray
        The initial estimate of the scale matrix.
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.
    max_iter : int
        The maximum number of iterations to perform.
    abs_tol : float
        The absolute tolerance for convergence.
    rel_tol : float
        The relative tolerance for convergence.
    loo_index : int, optional (default=-1)
        The index of the leave-one-out sample. If -1, the full covariance matrix
        is estimated.

    Returns
    -------
    np.ndarray
        The MLE scale matrix of the multivariate t-distribution.
    """
    scale_matrix = initial_scale_matrix.copy()
    for iteration in range(1, max_iter + 1):
        temp_cov_matrix = _scale_matrix_fixed_point_iteration(
            scale_matrix=scale_matrix,
            dof=dof,
            centered_samples=centered_samples,
            loo_index=loo_index,
        )

        # Note: 'ord = None' computes the Frobenius norm.
        absolute_residual_norm = np.linalg.norm(
            temp_cov_matrix - scale_matrix, ord=None
        )
        relative_residual_norm = absolute_residual_norm / max(
            np.linalg.norm(scale_matrix, ord=None), 1.0e-12
        )

        scale_matrix[:, :] = temp_cov_matrix[:, :]
        if absolute_residual_norm < abs_tol or relative_residual_norm < rel_tol:
            break

    if iteration == max_iter:
        raise RuntimeError(
            f"MultivariateTCost: Maximum number of iterations reached, ({max_iter}) "
            "in MLE scale matrix estimation. Relax the tolerance "
            "(mle_scale_abs_tol, mle_scale_rel_tol), "
            "or increase the maximum number of iterations (max_iter).",
        )

    return scale_matrix


@njit
def maximum_likelihood_mv_t_scale_matrix(
    centered_samples: np.ndarray,
    dof: float,
    abs_tol: float,
    rel_tol: float,
    max_iter: int,
    loo_index: int = -1,
) -> np.ndarray:
    """Compute the MLE scale matrix for a multivariate t-distribution.

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.
    abs_tol : float
        The absolute tolerance for convergence.
    rel_tol : float
        The relative tolerance for convergence.
    max_iter : int
        The maximum number of iterations to perform.
    loo_index : int, optional (default=-1)
        The index of the leave-one-out sample. If -1, the full covariance matrix
        is estimated.

    Returns
    -------
    np.ndarray
        The MLE covariance matrix of the multivariate t-distribution.
    """
    # Initialize the scale matrix maximum likelihood estimate:
    initial_mle_scale_matrix = _initial_scale_matrix_estimate(
        centered_samples,
        dof,
        loo_index=loo_index,
    )

    mle_scale_matrix = _solve_for_mle_scale_matrix(
        initial_scale_matrix=initial_mle_scale_matrix,
        centered_samples=centered_samples,
        dof=dof,
        loo_index=loo_index,
        max_iter=max_iter,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    return mle_scale_matrix


@njit
def _multivariate_t_log_likelihood(
    scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    dof: float,
) -> float:
    """Calculate the log likelihood of a multivariate t-distribution.

    Implemented from the definition of the multivariate t-distribution.
    Inspired by the scipy implementation of the multivariate t-distribution,
    but simplified.
    Source: https://en.wikipedia.org/wiki/Multivariate_t-distribution

    Parameters
    ----------
    scale_matrix : np.ndarray
        The scale matrix of the multivariate t-distribution.
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    float
        The log likelihood of the multivariate t-distribution.
    """
    sample_dim = centered_samples.shape[1]

    sign_det, log_det_scale_matrix = np.linalg.slogdet(scale_matrix)
    if sign_det <= 0:
        return np.nan
    inverse_scale_matrix = np.linalg.solve(scale_matrix, np.eye(sample_dim))

    total_log_likelihood = _fixed_param_multivariate_t_log_likelihood(
        centered_samples=centered_samples,
        dof=dof,
        inverse_scale_matrix=inverse_scale_matrix,
        log_det_scale_matrix=log_det_scale_matrix,
    )

    return total_log_likelihood


@njit
def _fixed_param_multivariate_t_log_likelihood(
    centered_samples: np.ndarray,
    dof: float,
    inverse_scale_matrix: np.ndarray,
    log_det_scale_matrix: float,
) -> float:
    """Calculate the log likelihood of a multivariate t-distribution.

    Directly from the definition of the multivariate t-distribution.
    Implementation inspired by the scipy implementation of
    the multivariate t-distribution, but simplified.
    Source: https://en.wikipedia.org/wiki/Multivariate_t-distribution

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.
    inverse_scale_matrix : np.ndarray
        The inverse of the scale matrix of the multivariate t-distribution.
    log_det_scale_matrix : float
        The log determinant of the scale matrix of the multivariate t-distribution.

    Returns
    -------
    float
        The log likelihood of the multivariate t-distribution.
    """
    num_samples, sample_dim = centered_samples.shape

    mahalanobis_squared_distances = np.sum(
        (centered_samples @ inverse_scale_matrix) * centered_samples, axis=1
    )

    # Normalization constants:
    exponent = 0.5 * (dof + sample_dim)
    A = log_gamma(exponent)
    B = log_gamma(0.5 * dof)
    C = 0.5 * sample_dim * np.log(dof * np.pi)
    D = 0.5 * log_det_scale_matrix

    normalization_contribution = num_samples * (A - B - C - D)
    sample_contributions = -exponent * np.log1p(mahalanobis_squared_distances / dof)
    total_log_likelihood = normalization_contribution + sample_contributions.sum()

    return total_log_likelihood


@njit
def _mv_t_ll_at_mle_params(
    X: np.ndarray,
    start: int,
    end: int,
    dof: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
) -> float:
    """Calculate multivariate T log likelihood at the MLE parameters for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).
    dof : float
        The degrees of freedom of the multivariate t-distribution.
    mle_scale_abs_tol : float
        The absolute tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_rel_tol : float
        The relative tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_max_iter : int
        The maximum number of iterations to perform for the MLE scale matrix estimation.

    Returns
    -------
    float
        The log likelihood of the observations in the
        interval ``[start, end)`` in the data matrix `X`,
        evaluated at the maximum likelihood parameter
        estimates for the mean and scale matrix, given
        the provided degrees of freedom.
    """
    X_segment = X[start:end]

    # Estimate the mean of each dimension through the sample medians:
    sample_medians = col_median(X_segment)
    X_centered = X_segment - sample_medians

    mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        X_centered,
        dof,
        abs_tol=mle_scale_abs_tol,
        rel_tol=mle_scale_rel_tol,
        max_iter=mle_scale_max_iter,
    )

    total_log_likelihood = _multivariate_t_log_likelihood(
        scale_matrix=mle_scale_matrix, centered_samples=X_centered, dof=dof
    )

    return total_log_likelihood


@njit(parallel=True)
def multivariate_t_cost_mle_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    dof: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
) -> np.ndarray:
    """Calculate the multivariate T twice negative log likelihood cost.

    At the maximum likelihood estimated mean and scale matrix values.

    Parameters
    ----------
    starts : np.ndarray
        The start indices of the segments.
    ends : np.ndarray
        The end indices of the segments.
    X : np.ndarray
        The data matrix. Rows are observations and columns are variables.
    dof : float
        The degrees of freedom for the cost calculation.
    mle_scale_abs_tol : float
        The absolute tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_rel_tol : float
        The relative tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_max_iter : int
        The maximum number of iterations to perform for the MLE scale matrix estimation.

    Returns
    -------
    np.ndarray
        The twice negative log likelihood costs for each segment.
    """
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))

    for i in prange(num_starts):
        segment_log_likelihood = _mv_t_ll_at_mle_params(
            X,
            starts[i],
            ends[i],
            dof=dof,
            mle_scale_abs_tol=mle_scale_abs_tol,
            mle_scale_rel_tol=mle_scale_rel_tol,
            mle_scale_max_iter=mle_scale_max_iter,
        )
        costs[i, 0] = -2.0 * segment_log_likelihood

    return costs


@njit
def _mv_t_ll_at_fixed_params(
    X: np.ndarray,
    start: int,
    end: int,
    mean: np.ndarray,
    inverse_scale_matrix: np.ndarray,
    log_det_scale_matrix: float,
    dof: float,
) -> float:
    """Calculate multivariate T log likelihood at the fixed parameters for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).
    mean : np.ndarray
        The mean of the multivariate t-distribution.
    inverse_scale_matrix : np.ndarray
        The inverse of the scale matrix of the multivariate t-distribution.
    log_det_scale_matrix : float
        The log determinant of the scale matrix of the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    float
        The log likelihood of the observations in the
        interval ``[start, end)`` in the data matrix `X`,
        evaluated at the fixed parameter values for the
        mean and scale matrix, given the provided degrees of freedom.
    """
    X_centered = X[start:end] - mean

    # Compute the log likelihood of the segment:
    total_log_likelihood = _fixed_param_multivariate_t_log_likelihood(
        inverse_scale_matrix=inverse_scale_matrix,
        log_det_scale_matrix=log_det_scale_matrix,
        centered_samples=X_centered,
        dof=dof,
    )

    return total_log_likelihood


@njit(parallel=True)
def multivariate_t_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    mean: np.ndarray,
    inverse_scale_matrix: np.ndarray,
    log_det_scale_matrix: float,
    dof: float,
) -> np.ndarray:
    """Calculate the multivariate T twice negative log likelihood cost.

    At fixed parameter values.

    Parameters
    ----------
    starts : np.ndarray
        The start indices of the segments.
    ends : np.ndarray
        The end indices of the segments.
    X : np.ndarray
        The data matrix.
    mean : np.ndarray
        The fixed mean for the cost calculation.
    inverse_scale_matrix : np.ndarray
        The fixed inverse scale matrix for the cost calculation.
    log_det_scale_matrix : float
        The log determinant of the scale matrix of the multivariate t-distribution.
    dof : float
        The fixed degrees of freedom for the cost calculation.

    Returns
    -------
    np.ndarray
        The twice negative log likelihood costs for each segment.
    """
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))

    for i in prange(num_starts):
        segment_log_likelihood = _mv_t_ll_at_fixed_params(
            X,
            starts[i],
            ends[i],
            mean=mean,
            inverse_scale_matrix=inverse_scale_matrix,
            log_det_scale_matrix=log_det_scale_matrix,
            dof=dof,
        )
        costs[i, 0] = -2.0 * segment_log_likelihood

    return costs


@njit
def _isotropic_mv_t_dof_estimate(
    centered_samples: np.ndarray, infinite_dof_threshold, zero_norm_tol=1.0e-6
) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution.

    From: A Novel Parameter Estimation Algorithm for the Multivariate
          t-Distribution and Its Application to Computer Vision.

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    infinite_dof_threshold : float
        The threshold at which the degrees of freedom are considered infinite.
        If the degrees of freedom are above this threshold,
        the multivariate t-distribution is approximated with
        a multivariate Gaussian distribution.
    zero_norm_tol : float, optional (default=1.0e-6)
        The tolerance for considering a squared norm as zero.

    Returns
    -------
    float
        The estimated degrees of freedom of the multivariate t-distribution.
    """
    sample_dim = centered_samples.shape[1]

    squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    log_norm_sq_var = np.log(squared_norms[squared_norms > zero_norm_tol**2]).var()

    b = log_norm_sq_var - trigamma(sample_dim / 2.0)
    inf_dof_b_threshold = (2 * infinite_dof_threshold + 4) / (infinite_dof_threshold**2)
    if b < inf_dof_b_threshold:
        # The dof estimate formula would exceed the infinite dof threshold,
        # (or break down due to a negative value), so we return infinity.
        return np.inf

    dof_estimate = (1 + np.sqrt(1 + 4 * b)) / b

    return dof_estimate


@njit
def _kurtosis_mv_t_dof_estimate(
    centered_samples: np.ndarray, infinite_dof_threshold: float
) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution.

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    infinite_dof_threshold : float
        The threshold at which the degrees of freedom are considered infinite.
        If the degrees of freedom are above this threshold,
        the multivariate t-distribution is approximated with
        a multivariate Gaussian distribution.

    Returns
    -------
    float
        The estimated degrees of freedom of the multivariate t-distribution.
    """
    sample_ellipitical_kurtosis = kurtosis(centered_samples).mean() / 3.0

    inf_dof_kurtosis_threshold = 2.0 / (infinite_dof_threshold - 4.0)
    if sample_ellipitical_kurtosis < inf_dof_kurtosis_threshold:
        # The elliptical kurtosis estimate is below the threshold
        # which would lead to a degrees of freedom estimate above the
        # infinite degrees of freedom threshold. We return infinity.
        return np.inf

    dof_estimate = (2.0 / sample_ellipitical_kurtosis) + 4.0
    return dof_estimate


@njit
def _iterative_mv_t_dof_estimate(
    centered_samples: np.ndarray,
    initial_dof: float,
    infinite_dof_threshold: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
    dof_abs_tol=1.0e-1,
    dof_rel_tol=5.0e-2,
    dof_max_iter=10,
) -> float:
    """Estimate dof. for a multivariate T distribution, iteratively.

    Using algorithm 1 from [1]_, we iteratively estimate the degrees of freedom
    parameter of a multivariate T distribution. The algorithm iteratively
    computes the MLE scale matrix with the current degrees of freedom estimate,
    and then updates the degrees of freedom estimate based on the relative
    trace of the sample covariance matrix and the MLE scale matrix.

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    initial_dof : float
        The initial degrees of freedom estimate.
    infinite_dof_threshold : float
        The threshold at which the degrees of freedom are considered infinite.
        If the degrees of freedom are above this threshold,
        the multivariate t-distribution is approximated with
        a multivariate Gaussian distribution.
    mle_scale_abs_tol : float
        The absolute tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_rel_tol : float
        The relative tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_max_iter : int
        The maximum number of iterations to perform for the MLE scale matrix estimation.
    dof_abs_tol : float, optional (default=1.0e-1)
        The absolute tolerance for convergence.
    dof_rel_tol : float, optional (default=5.0e-2)
        The relative tolerance for convergence.
    dof_max_iter : int, optional (default=10)
        The maximum number of iterations to perform.

    Returns
    -------
    float
        The estimated degrees of freedom of the multivariate t-distribution.

    .. [1] Ollila, Esa, & Daniel P. Palomar, & Frédéric Pascal. (2020). Shrinking the
       Eigenvalues of M-Estimators of Covariance Matrix. IEEE Transactions on Signal
       Processing, 256-269.
    """
    n = centered_samples.shape[0]
    if initial_dof > infinite_dof_threshold:
        return np.inf

    inf_dof_nu_threshold = infinite_dof_threshold / (infinite_dof_threshold - 2.0)

    sample_covariance = (centered_samples.T @ centered_samples) / n

    mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        centered_samples,
        initial_dof,
        max_iter=mle_scale_max_iter,
        abs_tol=mle_scale_abs_tol,
        rel_tol=mle_scale_rel_tol,
    )

    dof = initial_dof
    for _ in range(dof_max_iter):
        nu_i = np.trace(sample_covariance) / np.trace(mle_scale_matrix)
        if nu_i < inf_dof_nu_threshold:
            # The estimated degrees of freedom are high enough to approximate the
            # multivariate T distribution with a Gaussian.
            dof = np.inf
            break

        old_dof = dof
        dof = 2 * nu_i / max((nu_i - 1), 1.0e-3)

        mle_scale_matrix = _solve_for_mle_scale_matrix(
            initial_scale_matrix=mle_scale_matrix,
            centered_samples=centered_samples,
            dof=dof,
            abs_tol=mle_scale_abs_tol,
            rel_tol=mle_scale_rel_tol,
            max_iter=mle_scale_max_iter,
        )

        absolute_dof_diff = np.abs(dof - old_dof)
        rel_tol_satisfied = absolute_dof_diff / old_dof < dof_rel_tol
        abs_tol_satisfied = absolute_dof_diff < dof_abs_tol
        if rel_tol_satisfied or abs_tol_satisfied:
            break

    return dof


@njit(parallel=True)
def _loo_iterative_mv_t_dof_estimate(
    centered_samples: np.ndarray,
    initial_dof: float,
    infinite_dof_threshold: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
    dof_abs_tol=1.0e-1,
    dof_rel_tol=5.0e-2,
    dof_max_iter=5,
) -> float:
    """Leave-one-out iterative dof. estimate for a multivariate T distribution.

    Using an improved estimator, based on the algorithm in:
    'Improved estimation of the degree of freedom parameter of mv t-distribution''.
    However, the algorithm computes one MLE scale matrix estimate per samples,
    holding out one sample at a time, which increases computation cost.

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    initial_dof : float
        The initial degrees of freedom estimate.
    infinite_dof_threshold : float
        The threshold at which the degrees of freedom are considered infinite.
        If the degrees of freedom are above this threshold,
        the multivariate t-distribution is approximated with
        a multivariate Gaussian distribution.
    mle_scale_abs_tol : float
        The absolute tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_rel_tol : float
        The relative tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_max_iter : int
        The maximum number of iterations to perform for the MLE scale matrix estimation.
    dof_abs_tol : float, optional (default=1.0e-1)
        The absolute tolerance for convergence.
    dof_rel_tol : float, optional (default=5.0e-2)
        The relative tolerance for convergence.
    dof_max_iter : int, optional (default=5)
        The maximum number of iterations to perform.

    Returns
    -------
    float
        The estimated degrees of freedom of the multivariate T distribution.
    """
    if initial_dof > infinite_dof_threshold:
        return np.inf

    num_samples, sample_dimension = centered_samples.shape
    inf_dof_theta_threshold = infinite_dof_threshold / (infinite_dof_threshold - 2.0)

    sample_covariance = (centered_samples.T @ centered_samples) / num_samples
    grand_mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        centered_samples,
        dof=initial_dof,
        max_iter=mle_scale_max_iter,
        abs_tol=mle_scale_abs_tol,
        rel_tol=mle_scale_rel_tol,
    )
    contraction_estimate = np.trace(grand_mle_scale_matrix) / np.trace(
        sample_covariance
    )

    current_dof = initial_dof

    for _ in range(dof_max_iter):
        total_loo_mahalanobis_squared_distance = 0.0
        for sample in prange(num_samples):
            # Extract the leave-one-out sample as a column vector:
            loo_sample = centered_samples[sample, :].reshape(-1, 1)
            loo_sample_outer_product = loo_sample @ loo_sample.T

            # Initial estimate of the leave-one-out covariance matrix,
            # subtracting the contracted contribution of the leave-one-out sample:
            loo_scale_estimate = grand_mle_scale_matrix - contraction_estimate * (
                loo_sample_outer_product / num_samples
            )

            loo_mle_scale_matrix = _solve_for_mle_scale_matrix(
                initial_scale_matrix=loo_scale_estimate,
                centered_samples=centered_samples,
                dof=current_dof,
                loo_index=sample,
                abs_tol=mle_scale_abs_tol,
                rel_tol=mle_scale_rel_tol,
                max_iter=mle_scale_max_iter,
            )

            loo_mahalanobis_squared_distance = (
                loo_sample.T @ np.linalg.solve(loo_mle_scale_matrix, loo_sample)
            )[0, 0]
            total_loo_mahalanobis_squared_distance += loo_mahalanobis_squared_distance

        theta_k = (1 - sample_dimension / num_samples) * (
            (total_loo_mahalanobis_squared_distance / num_samples) / sample_dimension
        )
        if theta_k < inf_dof_theta_threshold:
            # The estimated degrees of freedom are high enough to approximate the
            # multivariate T distribution with a Gaussian distribution.
            current_dof = np.inf
            break

        new_dof = 2 * theta_k / (theta_k - 1)
        abs_dof_difference = np.abs(new_dof - current_dof)
        abs_tol_satisfied = abs_dof_difference < dof_abs_tol
        rel_tol_satisfied = (abs_dof_difference / current_dof) < dof_rel_tol

        current_dof = new_dof
        if abs_tol_satisfied or rel_tol_satisfied:
            break

    return current_dof


@njit
def _estimate_mv_t_dof(
    X: np.ndarray,
    infinite_dof_threshold: float,
    refine_dof_threshold: int,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
) -> float:
    """
    Estimate the degrees of freedom of a multivariate t-distribution.

    Parameters
    ----------
    X : np.ndarray
        The data matrix, where rows are observations and columns are variables.
    infinite_dof_threshold : float
        The threshold at which the degrees of freedom are considered infinite.
        If the degrees of freedom are above this threshold,
        the multivariate t-distribution is approximated with
        a multivariate Gaussian distribution.
    refine_dof_threshold : int
        The number of samples below which the degrees of freedom
        estimate is refined using a leave-one-out iterative method.
    mle_scale_abs_tol : float
        The absolute tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_rel_tol : float
        The relative tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_max_iter : int
        The maximum number of iterations to perform for the MLE scale matrix estimation.

    Returns
    -------
    float
        The estimated degrees of freedom of the multivariate t-distribution.
    """
    centered_samples = X - col_median(X)

    isotropic_dof = _isotropic_mv_t_dof_estimate(
        centered_samples, infinite_dof_threshold=infinite_dof_threshold
    )
    kurtosis_dof = _kurtosis_mv_t_dof_estimate(
        centered_samples, infinite_dof_threshold=infinite_dof_threshold
    )

    if np.isfinite(isotropic_dof) and np.isfinite(kurtosis_dof):
        # Initialize the iterative dof estimation method with the
        # geometric mean of the isotropic and kurtosis estimates:
        initial_dof_estimate = np.sqrt(isotropic_dof * kurtosis_dof)
    elif np.isfinite(isotropic_dof):
        initial_dof_estimate = isotropic_dof
    elif np.isfinite(kurtosis_dof):
        initial_dof_estimate = kurtosis_dof
    else:
        # Both initial estimates are infinite, start the
        # iterative method with a reasonably high initial dof:
        initial_dof_estimate = infinite_dof_threshold / 2.0

    dof_estimate = _iterative_mv_t_dof_estimate(
        centered_samples=centered_samples,
        initial_dof=initial_dof_estimate,
        infinite_dof_threshold=infinite_dof_threshold,
        mle_scale_abs_tol=mle_scale_abs_tol,
        mle_scale_rel_tol=mle_scale_rel_tol,
        mle_scale_max_iter=mle_scale_max_iter,
    )

    num_samples = X.shape[0]
    if num_samples <= refine_dof_threshold:
        dof_estimate = _loo_iterative_mv_t_dof_estimate(
            centered_samples=centered_samples,
            initial_dof=dof_estimate,
            infinite_dof_threshold=infinite_dof_threshold,
            mle_scale_abs_tol=mle_scale_abs_tol,
            mle_scale_rel_tol=mle_scale_rel_tol,
            mle_scale_max_iter=mle_scale_max_iter,
        )

    return dof_estimate


class MultivariateTCost(BaseCost):
    """Multivariate T twice negative log likelihood cost.

    The multivariate T-distribution is a generalization of the multivariate
    Gaussian distribution, allowing for heavier tails. The degrees of freedom
    parameter controls the tail heaviness, with higher values leading to
    distributions that are closer to the multivariate Gaussian distribution.
    With Numba installed the runtime of this cost is is between 5 and 10 times
    slower than the Gaussian likelihood cost, but more robust to outliers.

    The cost is calculated as the twice negative log likelihood of the
    multivariate T-distribution, given the data, mean, scale matrix, and
    degrees of freedom. The degrees of freedom can be fixed or estimated
    from the data. When estimating the degrees of freedom, we use several
    methods. Initially we compute the geometric mean of an isotropic dof.
    estimate from [1]_ and a kurtosis-based estimate from [2]_. This initial
    estimate is then fed into an iterative dof. estimate from [2]_. If the
    number of samples is below a given threshold (`refine_dof_threshold`),
    we refine the dof. estimate using a leave-one-out iterative method from [3]_.

    If the degrees of freedom are above a given threshold (`infinite_dof_threshold`),
    the multivariate T-distribution is approximated with a multivariate Gaussian
    distribution. The threshold is set to 50 by default, but can be adjusted
    with the `infinite_dof_threshold` parameter.

    As there is no analytical formula for the maximum likelihood estimate (MLE) of the
    scale matrix of the multivariate T-distribution, we use fixed point iterations
    to compute the MLE scale matrix within each segment. The tolerance parameters
    `mle_scale_abs_tol` and `mle_scale_rel_tol` for the MLE scale matrix estimation can
    be adjusted to control the convergence of the fixed point iterations, and when
    either of the tolerances are achieved, the fixed point iterations stop.

    The absolute tolerance (`mle_scale_abs_tol`) is achieved when the absolute change
    in the norm of the scale matrix changed less than `mle_scale_abs_tol` after an
    iteration, and the relative tolerance (`mle_scale_rel_tol`) is achieved when the
    relative change in the norm of the scale matrix is less than `mle_scale_rel_tol`
    after an iteration, relative to the pre-iteration norm.

    The maximum number of iterations (`mle_scale_max_iter`) is used to safeguard against
    non-convergence, and will raise a RuntimeError if the maximum number of iterations
    is reached. Reaching the maximum number of iterations is a sign that the tolerance
    parameters are too strict, and the fixed point iterations are not converging.
    In this case, the tolerance parameters can be relaxed, or the maximum number of
    iterations can be increased.

    Parameters
    ----------
    param : 2-tuple of float or np.ndarray, or None (default=None)
        Fixed mean and scale matrix for the cost calculation.
        If ``None``, the maximum likelihood estimates are used.
    fixed_dof : float, optional (default=None)
        Fixed degrees of freedom for the cost calculation.
        If None, the degrees of freedom are estimated from the data.
    refine_dof_threshold : int, optional
        (default=1000 with Numba installed, 100 without)
        The number of samples below which the degrees of freedom
        estimate is refined using a leave-one-out iterative method.
    infinite_dof_threshold : float, optional (default=50.0)
        The threshold at which the degrees of freedom are considered infinite.
        If the degrees of freedom are above this threshold,
        the multivariate t-distribution is approximated with
        a multivariate Gaussian distribution.
    mle_scale_abs_tol : float, optional (default=1.0e-2)
        The absolute tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_rel_tol : float, optional (default=1.0e-2)
        The relative tolerance for convergence in the MLE scale matrix estimation.
    mle_scale_max_iter : int, optional (default=100)
        The maximum number of iterations to perform for the MLE scale matrix estimation.
        Will raise a RuntimeError if the maximum number of iterations is reached.

    References
    ----------
    .. [1] Aeschliman, Chad, & Johnny Park, & Avinash C. Kak. (2009). A Novel
       Parameter Estimation Algorithm for the Multivariate T-Distribution and Its
       Application to Computer Vision. In Computer Vision - ECCV 2010, 594-607.
       Berlin, Heidelberg: Springer

    .. [2] Ollila, Esa, & Palomar, Daniel P. & Pascal, Frédéric. (2020). Shrinking the
       Eigenvalues of M-Estimators of Covariance Matrix. IEEE Transactions on Signal
       Processing, 256-269.

    .. [3] Pascal, Frédéric & Ollila, Esa & Palomar, Daniel P. (2021) Improved
       Estimation of the Degree of Freedom Parameter of Multivariate T-Distribution.
       In 2021 29th European Signal Processing Conference (EUSIPCO), 860-864.
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
        refine_dof_threshold=None,
        infinite_dof_threshold=5.0e1,
        mle_scale_abs_tol=1.0e-2,
        mle_scale_rel_tol=1.0e-2,
        mle_scale_max_iter=100,
    ):
        super().__init__(param)

        # Provided fixed degrees of freedom:
        self.fixed_dof = fixed_dof
        self.infinite_dof_threshold = infinite_dof_threshold
        self.refine_dof_threshold = refine_dof_threshold

        # Tolerance parameters for the MLE scale matrix estimation:
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
        check_in_interval(
            interval=pd.Interval(0, np.inf, closed="left"),
            value=self.mle_scale_abs_tol,
            name="mle_scale_abs_tol",
        )
        check_in_interval(
            interval=pd.Interval(0, np.inf, closed="left"),
            value=self.mle_scale_rel_tol,
            name="mle_scale_rel_tol",
        )
        check_larger_than(
            min_value=0, value=self.mle_scale_max_iter, name="mle_scale_max_iter"
        )

    def _check_fixed_param(
        self, param: tuple[MeanType, CovType], X: np.ndarray
    ) -> np.ndarray:
        """Check if the fixed mean parameter is valid.

        The covariance matrix is checked for positive definiteness,
        and forced to a floating point representation for numba compatibility.

        Parameters
        ----------
        param : 2-tuple of float or np.ndarray
            Fixed mean and covariance matrix for the cost calculation.
            Both are converted to float values or float arrays.
        X : np.ndarray
            Input data.

        Returns
        -------
        mean : np.ndarray
            Fixed mean for the cost calculation.
        """
        mean, cov = param
        mean = check_mean(mean, X)

        # Require floating point representation of
        # the covariance matrix for numba compatibility:
        cov = check_cov(cov, X, force_float=True)

        return mean, cov

    @property
    def min_size(self) -> int | None:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        """
        if self.is_fitted:
            return self.n_variables + 1
        else:
            return None

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return 1 + p + p * (p + 1) // 2

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method checks fixed distribution parameters, if provided, and
        precomputes quantities that are used in the cost evaluation.

        Additionally, the degrees of freedom for the multivariate T data generating
        distribution it estimated, if the degrees of freedom parameters was not set
        during the construction of the MultivariateTCost object.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        self._param = self._check_param(self.param, X)

        if self.param is not None:
            self._mean, scale_matrix = self._param
            self._inv_scale_matrix = np.linalg.inv(scale_matrix)
            _, self._log_det_scale_matrix = np.linalg.slogdet(scale_matrix)

        if self.fixed_dof is None:
            if self.refine_dof_threshold is None:
                if numba_available:
                    self.refine_dof_threshold = int(1.0e3)
                else:
                    self.refine_dof_threshold = 100

            self.dof_ = _estimate_mv_t_dof(
                X,
                infinite_dof_threshold=self.infinite_dof_threshold,
                refine_dof_threshold=self.refine_dof_threshold,
                mle_scale_abs_tol=self.mle_scale_abs_tol,
                mle_scale_rel_tol=self.mle_scale_rel_tol,
                mle_scale_max_iter=self.mle_scale_max_iter,
            )
        else:
            self.dof_ = self.fixed_dof

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the MLE parameters.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of columns
            is 1 since the MultivariateTCost is inherently multivariate.
        """
        if np.isposinf(self.dof_):
            return gaussian_cost_mle_params(starts, ends, X=self._X)
        else:
            return multivariate_t_cost_mle_params(
                starts,
                ends,
                X=self._X,
                dof=self.dof_,
                mle_scale_abs_tol=self.mle_scale_abs_tol,
                mle_scale_rel_tol=self.mle_scale_rel_tol,
                mle_scale_max_iter=self.mle_scale_max_iter,
            )

    def _evaluate_fixed_param(self, starts, ends):
        """Evaluate the cost for the fixed parameters.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of columns
            is 1 since the MultivariateGaussianCost is inherently multivariate.
        """
        if np.isposinf(self.dof_):
            return gaussian_cost_fixed_params(
                starts,
                ends,
                self._X,
                mean=self._mean,
                inv_cov=self._inv_scale_matrix,
                log_det_cov=self._log_det_scale_matrix,
            )
        else:
            return multivariate_t_cost_fixed_params(
                starts,
                ends,
                self._X,
                mean=self._mean,
                inverse_scale_matrix=self._inv_scale_matrix,
                log_det_scale_matrix=self._log_det_scale_matrix,
                dof=self.dof_,
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for interval evaluators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"param": None},
            {"param": (0.0, 1.0)},
            {"param": (np.zeros(1), np.eye(1))},
        ]
        return params

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Penalties and penalty functions for change and anomaly detection."""

__author__ = ["Tveten"]

import numpy as np

from sktime.detection._utils import check_larger_than


def make_bic_penalty(n_params, n, additional_cpts=1):
    """Create a BIC penalty.

    ``(n_params + additional_cpts) * log(n)``.

    Parameters
    ----------
    n_params : int
        Number of model parameters per segment.
    n : int
        Sample size.
    additional_cpts : int, default=1
        Number of additional change point parameters per segment.

    Returns
    -------
    float
    """
    check_larger_than(1, n_params, "n_params")
    check_larger_than(1, n, "n")
    check_larger_than(0, additional_cpts, "additional_cpts")
    return (n_params + additional_cpts) * np.log(n)


def make_chi2_penalty(n_params, n):
    """Create a chi-square penalty (CAPA "regime 1").

    ``n_params + 2 * sqrt(n_params * log(n)) + 2 * log(n)``.

    Parameters
    ----------
    n_params : int
        Number of model parameters per segment.
    n : int
        Sample size.

    Returns
    -------
    float
    """
    check_larger_than(1, n_params, "n_params")
    check_larger_than(1, n, "n")
    psi = np.log(n)
    return n_params + 2 * np.sqrt(n_params * psi) + 2 * psi


def make_linear_penalty(intercept, slope, p):
    """Create a linear penalty array.

    ``intercept + slope * (1, 2, ..., p)``.

    Parameters
    ----------
    intercept : float
        Intercept.
    slope : float
        Slope.
    p : int
        Number of variables.

    Returns
    -------
    np.ndarray of shape (p,)
    """
    check_larger_than(0.0, intercept, "intercept")
    check_larger_than(0.0, slope, "slope")
    check_larger_than(1, p, "p")
    return intercept + slope * np.arange(1, p + 1)


def make_linear_chi2_penalty(n_params_per_variable, n, p):
    """Create a linear chi-square penalty (MVCAPA "regime 2").

    Parameters
    ----------
    n_params_per_variable : int
        Number of model parameters per variable and segment.
    n : int
        Sample size.
    p : int
        Number of variables.

    Returns
    -------
    np.ndarray of shape (p,)
    """
    check_larger_than(1, n_params_per_variable, "n_params_per_variable")
    check_larger_than(1, n, "n")
    check_larger_than(1, p, "p")
    psi = np.log(n)
    component_penalty = 2 * np.log(n_params_per_variable * p)
    return 2 * psi + 2 * np.cumsum(np.full(p, component_penalty))


def make_nonlinear_chi2_penalty(n_params_per_variable, n, p):
    """Create a nonlinear chi-square penalty (MVCAPA "regime 3").

    Parameters
    ----------
    n_params_per_variable : int
        Number of model parameters per variable and segment.
    n : int
        Sample size.
    p : int
        Number of variables.

    Returns
    -------
    np.ndarray of shape (p,)

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset
       multivariate segment and point anomaly detection.
    """
    from scipy.stats import chi2

    check_larger_than(1, n_params_per_variable, "n_params_per_variable")
    check_larger_than(1, n, "n")
    check_larger_than(1, p, "p")

    if p == 1:
        return np.array([make_chi2_penalty(n_params_per_variable, n)])

    def _penalty_func(j):
        psi = np.log(n)
        c_j = chi2.ppf(1 - j / p, n_params_per_variable)
        f_j = chi2.pdf(c_j, n_params_per_variable)
        return (
            2 * (psi + np.log(p))
            + j * n_params_per_variable
            + 2 * p * c_j * f_j
            + 2
            * np.sqrt(
                (j * n_params_per_variable + 2 * p * c_j * f_j) * (psi + np.log(p))
            )
        )

    penalties = np.zeros(p, dtype=float)
    penalties[:-1] = np.vectorize(_penalty_func)(np.arange(1, p))
    penalties[-1] = penalties[-2]
    return penalties


def make_mvcapa_penalty(n_params_per_variable, n, p):
    """Create the default MVCAPA penalty (pointwise min of all three regimes).

    Parameters
    ----------
    n_params_per_variable : int
        Number of model parameters per variable and segment.
    n : int
        Sample size.
    p : int
        Number of variables.

    Returns
    -------
    np.ndarray of shape (p,)

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset
       multivariate segment and point anomaly detection.
    """
    n_params_total = n_params_per_variable * p
    constant = make_chi2_penalty(n_params_total, n)
    linear = make_linear_chi2_penalty(n_params_per_variable, n, p)
    nonlinear = make_nonlinear_chi2_penalty(n_params_per_variable, n, p)
    return np.fmin(constant, np.fmin(linear, nonlinear))


__all__ = [
    "make_bic_penalty",
    "make_chi2_penalty",
    "make_linear_penalty",
    "make_linear_chi2_penalty",
    "make_nonlinear_chi2_penalty",
    "make_mvcapa_penalty",
]

"""Non-linear penalties for change and anomaly detection."""

import numpy as np
from scipy.stats import chi2

from ..utils.validation.parameters import check_larger_than
from ._constant_penalties import make_chi2_penalty
from ._linear_penalties import make_linear_chi2_penalty


def make_nonlinear_chi2_penalty(
    n_params_per_variable: int, n: int, p: int
) -> np.ndarray:
    """Create a nonlinear chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 3" in the MVCAPA article [1]_, suitable for detecting
    both sparse and dense anomalies in the data. Sparse anomalies only affect a few
    variables, while dense anomalies affect many/all variables.

    Parameters
    ----------
    n_params_per_variable: int
        Number of model parameters per variable and segment.
    n : int
        Sample size.
    p : int
        Number of variables/columns in the data being analysed.

    Returns
    -------
    np.ndarray
        The non-decreasing nonlinear chi-square penalty values. The shape is ``(p,)``.
        Element ``i`` of the array is the penalty value for ``i+1`` variables being
        affected by a change or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.
    """
    check_larger_than(1, n_params_per_variable, "n_params_per_variable")
    check_larger_than(1, n, "n")
    check_larger_than(1, p, "p")

    if p == 1:
        # This penalty is not defined for p = 1, so we return a penalty value equal
        # to the constant chi square penalty.
        return np.array([make_chi2_penalty(n_params_per_variable, n)])

    def penalty_func(j: int) -> float:
        psi = np.log(n)
        c_j = chi2.ppf(1 - j / p, n_params_per_variable)
        f_j = chi2.pdf(c_j, n_params_per_variable)
        penalty = (
            2 * (psi + np.log(p))
            + j * n_params_per_variable
            + 2 * p * c_j * f_j
            + 2
            * np.sqrt(
                (j * n_params_per_variable + 2 * p * c_j * f_j) * (psi + np.log(p))
            )
        )
        return penalty

    penalties = np.zeros(p, dtype=float)
    penalties[:-1] = np.vectorize(penalty_func)(np.arange(1, p))
    # The penalty function is not defined for j = p, so the last value is duplicated
    penalties[-1] = penalties[-2]
    return penalties


def make_mvcapa_penalty(n_params_per_variable: int, n: int, p: int) -> np.ndarray:
    """Create the default penalty for the MVCAPA algorithm.

    The penalty is the pointwise minimum of the constant, linear, and nonlinear
    chi-square penalties: `make_chi2_penalty`, `make_linear_chi2_penalty`, and
    `make_nonlinear_chi2_penalty`. It is the recommended penalty for the MVCAPA
    algorithm [1]_.

    Parameters
    ----------
    n_params_per_variable: int
        Number of model parameters per variable and segment.
    n : int
        Sample size.
    p : int
        Number of variables/columns in the data being analysed.

    Returns
    -------
    np.ndarray
        The pointwise minimum penalty values. The shape is ``(p,)``. Element ``i`` of
        the array is the penalty value for ``i+1`` variables being affected by a change
        or anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.
    """
    n_params_total = n_params_per_variable * p
    constant_part = make_chi2_penalty(n_params_total, n)
    linear_part = make_linear_chi2_penalty(n_params_per_variable, n, p)
    nonlinear_part = make_nonlinear_chi2_penalty(n_params_per_variable, n, p)
    pointwise_minimum_penalty = np.fmin(
        constant_part, np.fmin(linear_part, nonlinear_part)
    )
    return pointwise_minimum_penalty

"""Linear penalties for change and anomaly detection."""

import numpy as np

from ..utils.validation.parameters import check_larger_than


def make_linear_penalty(intercept: float, slope: float, p: int) -> np.ndarray:
    """Create a linear penalty.

    The penalty is given by ``intercept + slope * (1, 2, ..., p)``, where `p` is the
    number of variables/columns in the data being analysed. The penalty is
    non-decreasing.

    Parameters
    ----------
    intercept : float
        Intercept of the linear penalty.
    slope : float
        Slope of the linear penalty.
    p : int
        Number of variables/columns in the data being analysed.

    Returns
    -------
    np.ndarray
        The non-decreasing linear penalty values. The shape is ``(p,)``. Element ``i``
        of the array is the penalty value for ``i+1`` variables being affected by a
        change or anomaly.
    """
    check_larger_than(0.0, intercept, "intercept")
    check_larger_than(0.0, slope, "slope")
    check_larger_than(1, p, "p")

    penalty_vector = intercept + slope * np.arange(1, p + 1)
    return penalty_vector


def make_linear_chi2_penalty(n_params_per_variable: int, n: int, p: int) -> np.ndarray:
    """Create a linear chi-square penalty.

    The penalty is a piece of the default penalty for the `MVCAPA` algorithm. It is
    described as "penalty regime 2" in the MVCAPA article [1]_, suitable for detecting
    sparse anomalies in the data. Sparse anomalies only affect a few variables.

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
        The non-decreasing linear chi-square penalty values. The shape is ``(p,)``.
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

    psi = np.log(n)
    component_penalty = 2 * np.log(n_params_per_variable * p)
    penalty_vector = 2 * psi + 2 * np.cumsum(np.full(p, component_penalty))
    return penalty_vector

"""Constant penalties for change and anomaly detection."""

import numpy as np

from ..utils.validation.parameters import check_larger_than


def make_bic_penalty(n_params: int, n: int, additional_cpts: int = 1) -> float:
    """Create a Bayesian Information Criterion (BIC) penalty.

    The BIC penalty is a constant penalty given by
    ``(n_params + additional_cpts) * log(n)``, where `n` is the sample size and
    `n_params` is the number of parameters per segment in the model across all
    variables, and `additional_cpts` is the number of additional change point parameters
    per segment. For change detection, this is 1.

    Parameters
    ----------
    n_params : int
        Number of model parameters per segment.
    n : int
        Sample size.
    additional_cpts: int, optional, default=1
        Number of additional change point parameters per segment. For change detection,
        this is 1.

    Returns
    -------
    float
        The BIC penalty value.
    """
    check_larger_than(1, n_params, "n_params")
    check_larger_than(1, n, "n")
    check_larger_than(0, additional_cpts, "additional_cpts")

    return (n_params + additional_cpts) * np.log(n)


def make_chi2_penalty(n_params: int, n: int) -> float:
    """Create a chi-square penalty.

    The penalty is the default penalty for the `CAPA` algorithm. It is described as
    "penalty regime 1" in the MVCAPA article [1]_. The penalty is based on a probability
    bound on the chi-squared distribution.

    The penalty is given by ``n_params + 2 * sqrt(n_params * log(n)) + 2 * log(n)``,
    where `n` is the sample size and `n_params` is the total number of parameters per
    segment in the model across all variables.

    Parameters
    ----------
    n_params : int
        Number of model parameters per segment.
    n : int
        Sample size.

    Returns
    -------
    float
        The chi-square penalty value.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.
    """
    check_larger_than(1, n_params, "n_params")
    check_larger_than(1, n, "n")

    psi = np.log(n)
    return n_params + 2 * np.sqrt(n_params * psi) + 2 * psi

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Poisson distribution cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._utils import (
    MeanType,
    check_non_negative_parameter,
    col_cumsum,
)
from sktime.detection.costs._base import BaseCost


def _poisson_cumsums(X):
    """Cumulative sums of the samples and of the log factorials of the samples.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_columns)
        non-negative integer counts

    Returns
    -------
    sums : np.ndarray of shape (n_samples + 1, n_columns)
        cumulative sums of ``X``, with a leading row of zeros
    log_factorial_sums : np.ndarray of shape (n_samples + 1, n_columns)
        cumulative sums of ``log(X!)``, with a leading row of zeros
    """
    X = X.astype(np.int64)
    sums = col_cumsum(X, init_zero=True)

    # log(x!) for every value of x present in the data, via a lookup table
    # of cumulative sums of log(k), k = 1, ..., max(X), so that every
    # log factorial is a sum of the same terms as in _poisson_log_likelihood
    max_sample = int(X.max()) if X.size > 0 else 0
    log_factorials = np.concatenate(
        [[0.0], np.cumsum(np.log(np.arange(1, max_sample + 1)))]
    )
    log_factorial_sums = col_cumsum(log_factorials[X], init_zero=True)

    return sums, log_factorial_sums


def _poisson_cost_mle_params(starts, ends, X):
    """Evaluate the Poisson cost with MLE parameters."""
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    sums, log_factorial_sums = _poisson_cumsums(X)

    n = (ends - starts).astype(float)[:, None]
    partial_sums = sums[ends] - sums[starts]
    partial_log_factorials = log_factorial_sums[ends] - log_factorial_sums[starts]

    # cost = -2 * (n * rate * (log(rate) - 1) - sum(log(x!))), rate = sum / n;
    # a zero sum, i.e., all samples zero, has cost zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mle_rate = partial_sums / n
        log_likelihood = partial_sums * (np.log(mle_rate) - 1.0)
    log_likelihood = np.where(partial_sums > 0, log_likelihood, 0.0)
    costs = -2.0 * (log_likelihood - partial_log_factorials)
    costs = np.where(partial_sums > 0, costs, 0.0)

    return costs


def _poisson_cost_fixed_params(starts, ends, X, rates):
    """Evaluate the Poisson cost for fixed rate parameters."""
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    rates = np.asarray(rates, dtype=float)
    sums, log_factorial_sums = _poisson_cumsums(X)

    n = (ends - starts).astype(float)[:, None]
    partial_sums = sums[ends] - sums[starts]
    partial_log_factorials = log_factorial_sums[ends] - log_factorial_sums[starts]

    log_likelihood = (
        -n * rates[None, :]
        + np.log(rates)[None, :] * partial_sums
        - partial_log_factorials
    )
    return -2.0 * log_likelihood


class PoissonCost(BaseCost):
    """Poisson distribution twice negative log likelihood cost.

    Parameters
    ----------
    param : float or array-like, optional (default=None)
        Fixed rate parameter of the Poisson distribution.
        If None, the cost is evaluated with rate set to
        the MLE estimate (sample mean) over each interval.

    Examples
    --------
    The data must consist of non-negative integer counts.

    >>> import numpy as np
    >>> from sktime.detection.costs import PoissonCost
    >>> X = np.array([1, 2, 1, 2, 20, 22, 20, 22]).reshape(-1, 1)
    >>> cost = PoissonCost()
    >>> cost.evaluate(X, [[0, 4], [4, 8], [0, 8]]).round(2)
    array([[  9.91],
           [ 19.75],
           [110.33]])

    The first two intervals are each well described by a single rate, so their
    cost is low. The third spans the change in rate and is expensive.

    Passing ``param`` evaluates the cost at a fixed rate instead of the MLE
    estimate of the interval:

    >>> cost = PoissonCost(param=2.0)
    >>> cost.evaluate(X, [[0, 4], [4, 8], [0, 8]]).round(2)
    array([[ 10.45],
           [262.78],
           [273.23]])
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
    }

    def __init__(self, param: MeanType | None = None):
        super().__init__(param)

    def _evaluate_optim_param(self, X, starts, ends):
        self._validate_poisson_data(X)
        return _poisson_cost_mle_params(starts, ends, X)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        self._validate_poisson_data(X)
        return _poisson_cost_fixed_params(starts, ends, X, param)

    def _validate_poisson_data(self, X):
        if not (np.issubdtype(X.dtype, np.integer) or np.all(X == X.astype(int))):
            raise ValueError("PoissonCost requires integer data.")
        if np.any(X < 0):
            raise ValueError("PoissonCost requires non-negative integer data.")

    def _check_fixed_param(self, param, X):
        return check_non_negative_parameter(param, X)

    @property
    def min_size(self):
        return 1

    def get_model_size(self, p):
        return p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"param": None}, {"param": 1.0}]

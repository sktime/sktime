# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""L2 cost function for change and anomaly detection."""

__author__ = ["Tveten"]


from sktime.detection._costs._base import BaseCost
from sktime.detection._utils import check_mean, col_cumsum


def _l2_cost_optim(starts, ends, sums, sums2):
    """Calculate L2 cost for optimal constant mean for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of input data, with a row of 0-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of squared input data, with a row of 0-entries
        as the first row.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    n = (ends - starts).reshape(-1, 1)
    costs = partial_sums2 - partial_sums**2 / n
    return costs


def _l2_cost_fixed(starts, ends, sums, sums2, mean):
    """Calculate L2 cost for a fixed constant mean for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of input data, with a row of 0-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of squared input data, with a row of 0-entries
        as the first row.
    mean : np.ndarray
        Fixed mean for the cost calculation.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    n = (ends - starts).reshape(-1, 1)
    costs = partial_sums2 - 2 * mean * partial_sums + n * mean**2
    return costs


# Try to compile with numba at first use
_l2_cost_optim_compiled = None
_l2_cost_fixed_compiled = None


def _get_l2_cost_optim():
    global _l2_cost_optim_compiled
    if _l2_cost_optim_compiled is not None:
        return _l2_cost_optim_compiled
    try:
        from numba import njit

        _l2_cost_optim_compiled = njit(_l2_cost_optim)
    except ImportError:
        _l2_cost_optim_compiled = _l2_cost_optim
    return _l2_cost_optim_compiled


def _get_l2_cost_fixed():
    global _l2_cost_fixed_compiled
    if _l2_cost_fixed_compiled is not None:
        return _l2_cost_fixed_compiled
    try:
        from numba import njit

        _l2_cost_fixed_compiled = njit(_l2_cost_fixed)
    except ImportError:
        _l2_cost_fixed_compiled = _l2_cost_fixed
    return _l2_cost_fixed_compiled


class L2Cost(BaseCost):
    """L2 cost of a constant mean.

    Parameters
    ----------
    param : float or array-like, optional (default=None)
        Fixed mean for the cost calculation. If ``None``, the optimal mean is
        calculated.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "supports_fixed_param": True,
    }

    def __init__(self, param=None):
        super().__init__(param)

    def _check_fixed_param(self, param, X):
        return check_mean(param, X)

    def _evaluate_optim_param(self, X, starts, ends):
        """Evaluate cost with optimal mean."""
        sums = col_cumsum(X, init_zero=True)
        sums2 = col_cumsum(X**2, init_zero=True)
        return _get_l2_cost_optim()(starts, ends, sums, sums2)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        """Evaluate cost with fixed mean."""
        sums = col_cumsum(X, init_zero=True)
        sums2 = col_cumsum(X**2, init_zero=True)
        return _get_l2_cost_fixed()(starts, ends, sums, sums2, param)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params = [
            {"param": None},
            {"param": 0.0},
        ]
        return params

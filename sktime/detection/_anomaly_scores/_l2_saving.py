# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Direct L2 saving for a zero-valued baseline mean."""

__author__ = ["Tveten"]


from sktime.detection._utils import col_cumsum
from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


def _l2_saving(starts, ends, sums):
    """Calculate the L2 saving for a zero-valued baseline mean.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of the input data, with a row of zeros as the first row.

    Returns
    -------
    savings : np.ndarray
        2D array of savings for each segment (rows) and component (columns).
    """
    n = (ends - starts).reshape(-1, 1)
    saving = (sums[ends] - sums[starts]) ** 2 / n
    return saving


class L2Saving(BaseIntervalScorer):
    """L2 saving for a zero-valued baseline mean.

    The L2 saving for a zero-mean can be computed more efficiently directly,
    rather than evaluating the cost separately for the baseline and optimized
    parameters. See [1]_ for details.

    Note that the data is assumed to have a zero-valued baseline mean for this
    saving to work; preprocess accordingly (e.g., subtract the median).

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear
       time method for the detection of collective and point anomalies.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "saving",
    }

    def __init__(self):
        super().__init__()

    def _evaluate(self, X, cuts):
        sums = col_cumsum(X, init_zero=True)
        starts = cuts[:, 0]
        ends = cuts[:, 1]
        return _l2_saving(starts, ends, sums)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{}, {}]

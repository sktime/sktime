"""The multivariate Gaussian test statistic for a change in mean and/or covariance."""

__author__ = ["johannvk"]
__all__ = ["MultivariateGaussianScore"]

import numpy as np

from ..base import BaseIntervalScorer
from ..costs._multivariate_gaussian_cost import MultivariateGaussianCost
from ..utils.numba import njit


@njit
def _half_integer_digamma(twice_n: int) -> float:
    """Calculate the digamma function for half integer values, i.e. ``twice_n/2``.

    The digamma function is the logarithmic derivative of the gamma function.
    This function is capable of calculating the
    digamma function for half integer values.

    Source: https://en.wikipedia.org/wiki/Digamma_function

    Parameters
    ----------
    twice_n : int
        Twice the integer value.

    Returns
    -------
    res : float
        Value of the digamma function for the half integer value.
    """
    assert twice_n > 0, "n must be a positive integer."

    if twice_n % 2 == 0:
        # Even integer: twice_n = 2n
        res = -np.euler_gamma
        n = twice_n // 2
        for k in range(0, n - 1):
            res += 1.0 / (k + 1.0)
    else:
        res = -2 * np.log(2) - np.euler_gamma
        # Odd integer: twice_n = 2n + 1
        n = (twice_n - 1) // 2
        for k in range(1, n + 1):
            res += 2.0 / (2.0 * k - 1.0)

    return res


@njit
def likelihood_ratio_expected_value(
    sequence_length: int, cut_point: int, dimension: int
) -> float:
    """Calculate the expected value of twice the negative log likelihood ratio.

    We check that the cut point is within the sequence length, and that both `k` and `n`
    are large enough relative to the dimension `p`, to ensure that the expected
    value is finite.
    Should at least have `p+1` points on each side of a split, for `p` dimensional data.

    Parameters
    ----------
    sequence_length : int
        Length of the sequence.
    cut_point : int
        Cut point of the sequence.
    dimension : int
        Dimension of the data.

    Returns
    -------
    g_k_n : float
        Expected value of twice the negative log likelihood ratio.
    """
    n, k, p = sequence_length, cut_point, dimension

    assert 0 < k < n, "Cut point `k` must be within the sequence length `n`."
    assert p > 0, "Dimension `p` must be a positive integer."
    assert k > p, "Cut point `k` must be larger than the dimension `p`."
    assert (
        n - k > p
    ), "Run length after cut point `n - k` must be larger than dimension `p`."

    g_k_n = p * (
        np.log(2)
        + (n - 1) * np.log(n - 1)
        - (n - k - 1) * np.log(n - k - 1)
        - (k - 1) * np.log(k - 1)
    )

    for j in range(1, p + 1):
        g_k_n += (
            (n - 1) * _half_integer_digamma(n - j)
            - (k - 1) * _half_integer_digamma(k - j)
            - (n - k - 1) * _half_integer_digamma(n - k - j)
        )

    return g_k_n


@njit
def compute_bartlett_corrections(
    sequence_lengths: np.ndarray, cut_points: np.ndarray, dimension: int
):
    """Calculate the Bartlett correction for the twice negated log likelihood ratio.

    Parameters
    ----------
    twice_negated_log_lr : float
        Twice the negative log likelihood ratio.
    sequence_length : int
        Length of the sequence.
    cut_point : int
        Cut point of the sequence.
    dimension : int
        Dimension of the data.

    Returns
    -------
    bartlett_corr_log_lr : float
    """
    bartlett_corrections = np.zeros(
        shape=(sequence_lengths.shape[0], 1), dtype=np.float64
    )

    for i, (sequence_length, cut_point) in enumerate(zip(sequence_lengths, cut_points)):
        g_k_n = likelihood_ratio_expected_value(
            sequence_length=sequence_length, cut_point=cut_point, dimension=dimension
        )
        bartlett_correction_factor = dimension * (dimension + 3.0) / g_k_n
        bartlett_corrections[i] = bartlett_correction_factor

    return bartlett_corrections


class MultivariateGaussianScore(BaseIntervalScorer):
    """Multivariate Gaussian change score for a change in mean and/or covariance.

    Scores are calculated as the likelihood ratio scores for a change
    in mean and covariance under a multivariate Gaussian distribution [1]_.

    To stabilize the score, the Bartlett correction is applied by default,
    which adjusts the score for the relative sizes of the left and right segments.

    Applying the Bartlett correction results in a more stable score,
    especially for small sample sizes, ensuring that the scores
    approach the chi-squared distribution asymptotically.

    Parameters
    ----------
    apply_bartlett_correction : bool, default=True
        Whether to apply the Bartlett correction to the change scores.

    References
    ----------
    .. [1] Zamba, K. D., & Hawkins, D. M. (2009). A Multivariate Change-Point Model
       for Change in Mean Vector and/or Covariance Structure. Journal of Quality
       Technology, 41(3), 285-303.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "task": "change_score",
        "is_aggregated": True,
    }

    def __init__(self, apply_bartlett_correction: bool = True):
        super().__init__()
        self._cost = MultivariateGaussianCost()
        self.apply_bartlett_correction = apply_bartlett_correction

    @property
    def min_size(self) -> int | None:
        """Minimum size of the interval to evaluate."""
        if self._is_fitted:
            return self._cost.min_size
        else:
            return None

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters to estimate over each interval.

        The primary use of this method is to determine an appropriate default penalty
        value in detectors.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p + p * (p + 1) // 2

    def _fit(self, X: np.ndarray, y=None):
        """Fit the multivariate Gaussian change score evaluator.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.
        """
        self._cost.fit(X)
        return self

    def _evaluate(self, cuts: np.ndarray):
        """Evaluate the change score for a split within an interval.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the ``start``, the second is the ``split``, and the
            third is the ``end`` of the interval to evaluate.
            The difference between subsets ``X[start:split]`` and ``X[split:end]`` is
            evaluated for each row in `cuts`.

        Returns
        -------
        scores : np.ndarray
            A 2D array of change scores. One row for each cut. The number of
            columns is 1 if the change score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the score is
            univariate. In this case, each column represents the univariate score for
            the corresponding input data column.
        """
        start_intervals = cuts[:, [0, 1]]
        end_intervals = cuts[:, [1, 2]]
        total_intervals = cuts[:, [0, 2]]

        raw_scores = self._cost.evaluate(total_intervals) - (
            self._cost.evaluate(start_intervals) + self._cost.evaluate(end_intervals)
        )

        if self.apply_bartlett_correction:
            segment_lengths = cuts[:, 2] - cuts[:, 0]
            segment_splits = cuts[:, 1] - cuts[:, 0]
            bartlett_corrections = compute_bartlett_corrections(
                sequence_lengths=segment_lengths,
                cut_points=segment_splits,
                dimension=self.n_variables,
            )
            return bartlett_corrections * raw_scores
        else:
            return raw_scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval scorers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"apply_bartlett_correction": False},
            {"apply_bartlett_correction": True},
        ]
        return params

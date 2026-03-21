"""The Moving Window algorithm for multiple changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["MovingWindow"]

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..change_scores import CUSUM, to_change_score
from ..compose.penalised_score import PenalisedScore
from ..utils.numba import njit
from ..utils.numba.general import where
from ..utils.validation.data import check_data
from ..utils.validation.interval_scorer import check_interval_scorer
from ..utils.validation.parameters import check_in_interval, check_larger_than
from ..utils.validation.penalties import check_penalty
from .base import BaseChangeDetector


@njit
def make_extended_moving_window_cuts(
    n_samples: int,
    bandwidth: int,
    min_size: int,
) -> np.ndarray:
    splits = np.arange(min_size, n_samples - min_size + 1)

    starts = splits - bandwidth
    starts[starts < 0] = 0
    max_start = n_samples - 2 * bandwidth
    starts[starts > max_start] = max_start

    ends = splits + bandwidth
    ends[ends > n_samples] = n_samples
    min_end = 2 * bandwidth
    ends[ends < min_end] = min_end

    cuts = np.column_stack((starts, splits, ends))
    return cuts


def transform_moving_window(
    penalised_score: BaseIntervalScorer,
    bandwidth: int,
) -> np.ndarray:
    penalised_score.check_is_penalised()
    penalised_score.check_is_fitted()

    n_samples = penalised_score.n_samples
    cuts = make_extended_moving_window_cuts(
        n_samples, bandwidth, penalised_score.min_size
    )
    scores = np.repeat(np.nan, n_samples)
    scores[cuts[:, 1]] = penalised_score.evaluate(cuts).reshape(-1)
    return scores


def transform_multiple_moving_window(
    penalised_score: BaseIntervalScorer,
    bandwidths: np.ndarray,
) -> np.ndarray:
    n_samples = penalised_score.n_samples
    scores = np.full((n_samples, len(bandwidths)), np.nan)
    for i, bw in enumerate(bandwidths):
        scores[:, i] = transform_moving_window(penalised_score, bw)
    return scores


@njit
def get_candidate_changepoints(
    scores: np.ndarray,
) -> tuple[list[int], list[tuple[int, int]]]:
    detection_intervals = where(scores > 0)
    changepoints = []
    for start, end in detection_intervals:
        cpt = start + np.argmax(scores[start:end])
        changepoints.append(cpt)
    return changepoints, detection_intervals


@njit
def select_changepoints_by_detection_length(
    scores: np.ndarray, min_detection_interval: int
) -> list:
    candidate_cpts, detection_intervals = get_candidate_changepoints(scores)
    cpts = [
        cpt
        for cpt, interval in zip(candidate_cpts, detection_intervals)
        if interval[1] - interval[0] >= min_detection_interval
    ]

    return cpts


@njit
def select_changepoints_by_local_optimum(
    scores: np.ndarray, selection_bandwidth: int
) -> list:
    candidate_cpts, _ = get_candidate_changepoints(scores)
    cpts = [
        cpt
        for cpt in candidate_cpts
        if np.isclose(
            scores[cpt],
            np.max(
                scores[
                    max(cpt - selection_bandwidth, 0) : cpt + selection_bandwidth + 1
                ]
            ),
        )
    ]

    return cpts


@njit
def select_changepoints_by_bottom_up(
    scores: np.ndarray, bandwidths: np.ndarray, local_optimum_fraction: float
) -> list:
    bandwidths = sorted(bandwidths)
    candidate_cpts = []
    for i, bw in enumerate(bandwidths):
        local_optimum_bandwidth = int(local_optimum_fraction * bw)
        candidate_cpts_bw = select_changepoints_by_local_optimum(
            scores[:, i], local_optimum_bandwidth
        )
        for candidate_cpt in candidate_cpts_bw:
            candidate_cpts.append((candidate_cpt, bw))

    cpts = [candidate_cpts[0][0]]
    for candidate_cpt, bw in candidate_cpts[1:]:
        distance_to_closest = np.min(np.abs(candidate_cpt - np.array(cpts)))
        local_optimum_bandwidth = int(local_optimum_fraction * bw)
        if distance_to_closest >= local_optimum_bandwidth:
            cpts.append(candidate_cpt)

    return cpts


class MovingWindow(BaseChangeDetector):
    """Moving window algorithm for multiple change-point detection.

    The MOSUM (moving sum) algorithm [1]_, but generalized to allow for any penalised
    and unpenalised change score. The basic algorithm runs a test statistic for a
    single change-point across the data in a moving window fashion.
    In each window, the data is split into two equal halves with `bandwidth` samples
    on either side of a split point.
    This process generates a time series of penalised scores, which are used to generate
    candidate change-points as local maxima within intervals where the penalised scores
    are all above zero.
    The final set of change-points is selected from the candidate change-points using
    one of the two selection methods described in [2]_.

    Several of the extensions available in the mosum R package [2]_ are also available
    in this implementation, including the ability to use multiple bandwidths. The
    CUSUM-type boundary extension for computing the test statistic for candidate change-
    points less than `bandwidth` samples from the start and end of the data is also
    implemented by default.

    Parameters
    ----------
    change_score : BaseIntervalScorer, optional, default=CUSUM()
        The change score to use in the algorithm. If a cost is given, it is
        converted to a change score using the `ChangeScore` class.
    penalty : np.ndarray or float, optional, default=None
        The penalty to use for change detection. If the penalty is
        penalised (`change_score.get_tag("is_penalised")`) the penalty will
        be ignored. The different types of penalties are as follows:

        * ``float``: A constant penalty applied to the sum of scores across all
          variables in the data.
        * ``np.ndarray``: A penalty array of the same length as the number of
          columns in the data, where element ``i`` of the array is the penalty for
          ``i+1`` variables being affected by a change. The penalty array
          must be positive and increasing (not strictly). A penalised score with a
          linear penalty array is faster to evaluate than a nonlinear penalty array.
        * ``None``: A default penalty is created in `predict` based on the fitted
          score using the `make_bic_penalty` function.

    bandwidth : int or list of int, default=20
        The bandwidth is the number of samples on either side of a candidate
        change-point. Must be 1 or greater. If a list of bandwidths is given, the
        algorithm will run for each bandwidth in the list and combine the results
        accoring to the "bottom-up" merging approach described in [2]_. A fibonacci
        sequence of bandwidths is recommended for multiple bandwidths by the authors
        in [2]_.
    selection_method : str, default="local_optimum"
        The method used to select the final set of change-points from a set of candidate
        change-points. The options are:

        * ``"detection_length"``: Accepts a candidate change-point if the
          ``min_detection_fraction * bandwidth`` consecutive penalised scores are above
          zero. Corresponds to the epsilon-criterion in [2]_. This method is only
          available for a single bandwidth.
        * ``"local_optimum"``: Accepts a candidate change-point if it is the local
          maximum in the scores within a neighbourhood of size
          ``local_optimum_fraction * bandwidth``. Corresponds to the eta-criterion
          in [2]_. This method is used within the "bottom-up" merging approach if
          multiple bandwidths are given.

    min_detection_fraction : float, default=0.2
        The minimum size of the detection interval for a candidate change-point to be
        accepted in the ``"detection_length"`` selection method.
        be between ``0`` (exclusive) and ``1/2`` (exclusive).
    local_optimum_fraction : float, default=0.4
        The size of the neighbourhood around a candidate change-point used in the
        ``"local_optimum"`` selection method. Must be larger than or equal to ``0``.

    References
    ----------
    .. [1] Eichinger, B., & Kirch, C. (2018). A MOSUM procedure for the estimation of
       multiple random change points.

    .. [2] Meier, A., Kirch, C., & Cho, H. (2021). mosum: A package for moving sums in
       change-point analysis. Journal of Statistical Software, 97, 1-42.

    Examples
    --------
    >>> from sktime.detection._skchange.change_detectors import MovingWindow
    >>> from sktime.detection._skchange.datasets import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=4, mean=10, segment_length=100, p=5)
    >>> detector = MovingWindow()
    >>> detector.fit_predict(df)
       ilocs
    0    100
    1    200
    2    300
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
        "fit_is_empty": True,
    }

    def __init__(
        self,
        change_score: BaseIntervalScorer | None = None,
        penalty: np.ndarray | float | None = None,
        bandwidth: int | list = 20,
        selection_method: str = "local_optimum",
        min_detection_fraction: float = 0.2,
        local_optimum_fraction: float = 0.4,
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.bandwidth = bandwidth
        self.selection_method = selection_method
        self.min_detection_fraction = min_detection_fraction
        self.local_optimum_fraction = local_optimum_fraction
        super().__init__()

        _score = CUSUM() if change_score is None else change_score
        check_interval_scorer(
            _score,
            "change_score",
            "MovingWindow",
            required_tasks=["cost", "change_score"],
            allow_penalised=True,
        )
        _change_score = to_change_score(_score)

        check_penalty(penalty, "penalty", "MovingWindow")
        self._penalised_score = (
            _change_score.clone()  # need to avoid modifying the input change_score
            if _change_score.get_tag("is_penalised")
            else PenalisedScore(_change_score, penalty)
        )

        if isinstance(bandwidth, int):
            check_larger_than(1, bandwidth, "bandwidth")
            self._bandwidth = np.array([bandwidth])
        elif isinstance(bandwidth, list):
            if len(bandwidth) == 0:
                raise ValueError("`bandwidth` must be a non-empty list.")
            if not all(isinstance(bw, int) for bw in bandwidth):
                raise TypeError("All elements of `bandwidth` must be integers.")
            if any(bw < 1 for bw in bandwidth):
                raise ValueError("All elements of `bandwidth` must be greater than 0.")
            self._bandwidth = np.array(bandwidth)
        else:
            raise TypeError(
                "`bandwidth` must be an integer or a list of integers. "
                f"Got {type(bandwidth)}."
            )

        check_in_interval(
            pd.Interval(0, 1 / 2, closed="neither"),
            min_detection_fraction,
            "min_detection_fraction",
        )
        check_larger_than(0, local_optimum_fraction, "local_optimum_fraction")

        valid_selection_methods = ["local_optimum", "detection_length"]
        if selection_method not in valid_selection_methods:
            raise ValueError(
                f"`selection_method` must be one of {valid_selection_methods}."
                f" Got {selection_method}."
            )
        if len(self._bandwidth) > 1 and self.selection_method == "detection_length":
            raise ValueError(
                "The selection method `detection_length` is not supported for multiple"
                " bandwidths. Use `'local_optimum'` instead."
            )

        self.clone_tags(_change_score, ["distribution_type"])

    def _transform_scores(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to score (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence `X`.

        Attributes
        ----------
        fitted_score : BaseIntervalScorer
            The fitted penalised change score used for the detection.
        """
        X = check_data(
            X,
            min_length=2 * max(self._bandwidth),
            min_length_name="2*max(bandwidth)",
        )

        self.fitted_score: BaseIntervalScorer = self._penalised_score.clone()
        self.fitted_score.fit(X)
        scores = transform_multiple_moving_window(self.fitted_score, self._bandwidth)
        formatted_scores = pd.DataFrame(
            scores,
            index=X.index,
            columns=pd.Index([bw for bw in self._bandwidth], name="bandwidth"),
        )
        return formatted_scores

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect change points in.

        Returns
        -------
        y_sparse : pd.DataFrame
            A `pd.DataFrame` with a range index and one column:
            * ``"ilocs"`` - integer locations of the changepoints.

        Attributes
        ----------
        fitted_score : BaseIntervalScorer
            The fitted penalised change score used for the detection.
        scores : pd.Series
            The detection scores obtained by the `transform_scores` method.
        """
        self.scores: pd.DataFrame = self.transform_scores(X)

        if self.selection_method == "detection_length":
            min_detection_length = int(self.min_detection_fraction * self.bandwidth)
            changepoints = select_changepoints_by_detection_length(
                self.scores.values.reshape(-1), min_detection_length
            )
        elif self.selection_method == "local_optimum" and len(self._bandwidth) == 1:
            local_optimum_bandwidth = int(self.local_optimum_fraction * self.bandwidth)
            changepoints = select_changepoints_by_local_optimum(
                self.scores.values.reshape(-1), local_optimum_bandwidth
            )
        else:
            changepoints = select_changepoints_by_bottom_up(
                self.scores.values, self._bandwidth, self.local_optimum_fraction
            )

        return self._format_sparse_output(changepoints)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.detection._skchange.costs import GaussianCost, L2Cost

        params = [
            {"change_score": L2Cost(), "bandwidth": 5, "penalty": 20},
            {"change_score": GaussianCost(), "bandwidth": 5, "penalty": 30},
        ]
        return params

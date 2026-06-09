"""The collective and point anomalies (CAPA) algorithm."""

__author__ = ["Tveten"]
__all__ = ["CAPA"]

import numpy as np
import pandas as pd

from ..anomaly_scores import L2Saving, to_saving
from ..base import BaseIntervalScorer
from ..compose.penalised_score import PenalisedScore
from ..penalties import make_chi2_penalty, make_linear_chi2_penalty
from ..utils.numba import njit
from ..utils.validation.data import check_data
from ..utils.validation.interval_scorer import check_interval_scorer
from ..utils.validation.parameters import check_larger_than, check_smaller_than
from ..utils.validation.penalties import check_penalty
from .base import BaseSegmentAnomalyDetector


def _make_nonlinear_chi2_penalty_from_score(score: BaseIntervalScorer) -> np.ndarray:
    from ..penalties import make_nonlinear_chi2_penalty

    score.check_is_fitted()
    n = score.n_samples
    p = score.n_variables
    return make_nonlinear_chi2_penalty(score.get_model_size(p), n, p)


@njit
def get_anomalies(
    anomaly_starts: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    segment_anomalies = []
    point_anomalies = []
    i = anomaly_starts.size - 1
    while i >= 0:
        start_i = anomaly_starts[i]
        size = i - start_i + 1
        if size > 1:
            segment_anomalies.append((int(start_i), i + 1))
            i = int(start_i)
        elif size == 1:
            point_anomalies.append((i, i + 1))
        i -= 1
    return segment_anomalies, point_anomalies


def get_affected_components(
    penalised_scorer: PenalisedScore,
    anomalies: list[tuple[int, int]],
) -> list[tuple[int, int, np.ndarray]]:
    penalised_scorer.check_is_penalised()
    penalised_scorer.check_is_fitted()
    new_anomalies = []
    for start, end in anomalies:
        saving_values = penalised_scorer.score_.evaluate(np.array([start, end]))[0]
        saving_order = np.argsort(-saving_values)  # Decreasing order.
        penalised_savings = (
            np.cumsum(saving_values[saving_order]) - penalised_scorer.penalty_
        )
        argmax = np.argmax(penalised_savings)
        new_anomalies.append((start, end, saving_order[: argmax + 1]))
    return new_anomalies


def run_capa(
    segment_penalised_saving: PenalisedScore,
    point_penalised_saving: PenalisedScore,
    min_segment_length: int,
    max_segment_length: int,
    find_affected_components: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    segment_penalised_saving.check_is_penalised()
    segment_penalised_saving.check_is_fitted()
    point_penalised_saving.check_is_penalised()
    point_penalised_saving.check_is_fitted()
    n_samples = segment_penalised_saving.n_samples
    if not n_samples == point_penalised_saving.n_samples:
        raise ValueError(
            "The segment and point saving costs must span the same number of samples."
        )

    opt_savings = np.zeros(n_samples + 1)

    # Store the optimal start and affected components of an anomaly for each t.
    # Used to get the final set of anomalies after the loop.
    opt_anomaly_starts = np.repeat(np.nan, n_samples)
    starts = np.array([], dtype=int)
    max_segment_penalty = np.max(segment_penalised_saving.penalty_)

    ts = np.arange(min_segment_length - 1, n_samples)
    for t in ts:
        # Segment anomalies
        t_array = np.array([t])

        starts = np.concatenate((starts, t_array - min_segment_length + 1))
        ends = np.repeat(t + 1, len(starts))
        intervals = np.column_stack((starts, ends))
        segment_savings = segment_penalised_saving.evaluate(intervals).flatten()
        candidate_savings = opt_savings[starts] + segment_savings
        candidate_argmax = np.argmax(candidate_savings)
        opt_segment_saving = candidate_savings[candidate_argmax]
        opt_start = starts[0] + candidate_argmax

        # Point anomalies
        point_interval = np.column_stack((t_array, t_array + 1))
        point_savings = point_penalised_saving.evaluate(point_interval).flatten()
        opt_point_saving = opt_savings[t] + point_savings[0]

        # Combine and store results
        savings = np.array([opt_savings[t], opt_segment_saving, opt_point_saving])
        argmax = np.argmax(savings)
        opt_savings[t + 1] = savings[argmax]
        if argmax == 1:
            opt_anomaly_starts[t] = opt_start
        elif argmax == 2:
            opt_anomaly_starts[t] = t

        # Pruning the admissible starts
        saving_too_low = candidate_savings + max_segment_penalty <= opt_savings[t + 1]
        too_long_segment = starts < t - max_segment_length + 2
        prune = saving_too_low | too_long_segment
        starts = starts[~prune]

    segment_anomalies, point_anomalies = get_anomalies(opt_anomaly_starts)

    if find_affected_components:
        segment_anomalies = get_affected_components(
            segment_penalised_saving, segment_anomalies
        )
        point_anomalies = get_affected_components(
            point_penalised_saving,
            point_anomalies,
        )
    return opt_savings[1:], segment_anomalies, point_anomalies


def _make_chi2_penalty_from_score(score: BaseIntervalScorer) -> float:
    score.check_is_fitted()
    n = score.n_samples
    p = score.n_variables
    return make_chi2_penalty(score.get_model_size(p), n)


def _make_linear_chi2_penalty_from_score(score: BaseIntervalScorer) -> np.ndarray:
    score.check_is_fitted()
    n = score.n_samples
    p = score.n_variables
    return make_linear_chi2_penalty(score.get_model_size(p), n, p)


def _check_capa_penalised_score(score: BaseIntervalScorer, name: str, caller_name: str):
    if score.get_tag("is_penalised") and not isinstance(score, PenalisedScore):
        raise ValueError(
            f"{caller_name} only supports a penalised `{name}` constructed"
            " by `PenalisedScore`."
        )


class CAPA(BaseSegmentAnomalyDetector):
    """The collective and point anomaly (CAPA) detection algorithm.

    An efficient implementation of the CAPA family of algorithms for anomaly detection.
    Supports both univariate data [1]_ (CAPA) and multivariate data with subset
    anomalies [2]_ (MVCAPA) by using the penalised saving formulation of the collective
    anomaly detection problem found in [2]_ and [3]_. For multivariat data, the
    algorithm can also be used to infer the affected components for each anomaly given
    a suitable penalty array.

    Parameters
    ----------
    segment_saving : BaseIntervalScorer, optional, default=L2Saving()
        The saving to use for segment anomaly detection.
        If a cost is given, the saving is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
        If a penalised saving is given, it must be constructed from `PenalisedScore`.
    segment_penalty : np.ndarray or float, optional, default=None
        The penalty to use for segment anomaly detection. If the segment penalty is
        penalised (`segment_penalty.get_tag("is_penalised")`) the penalty will
        be ignored. The different types of penalties are as follows:

        * ``float``: A constant penalty applied to the sum of scores across all
          variables in the data.
        * ``np.ndarray``: A penalty array of the same length as the number of
          columns in the data, where element ``i`` of the array is the penalty for
          ``i+1`` variables being affected by an anomaly. The penalty array
          must be positive and increasing (not strictly). A penalised score with a
          linear penalty array is faster to evaluate than a nonlinear penalty array.
        * ``None``: A default constant penalty is created in `predict` based on the
          fitted score using the `make_chi2_penalty` function.

    point_saving : BaseIntervalScorer, optional, default=L2Saving()
        The saving to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a cost is given, the saving is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
        If a penalised saving is given, it must be constructed from `PenalisedScore`.
    point_penalty : np.ndarray or float, optional, default=None
        The penalty to use for point anomaly detection. See the documentation for
        `segment_penalty` for details. For ``None`` input, the default is set using the
        `make_linear_chi2_penalty` function.
    min_segment_length : int, optional, default=2
        Minimum length of a segment. This may be overridden by the `min_size` of the
        fitted `segment_saving`.
    max_segment_length : int, optional, default=1000
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional, default=False
        If ``True``, detected point anomalies are not returned by `predict`. I.e., only
        segment anomalies are returned. If ``False``, point anomalies are included in
        the output as segment anomalies of length 1.
    find_affected_components : bool, optional, default=False
        If ``True``, the affected components for each segment anomaly are returned in
        the `"icolumns"` key of the `predict` output.
        Only relevant for multivariate data in combination with a penalty array.
        The affected components are sorted from the highest to lowest evidence
        of an anomaly being present in the variable.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear time method
        for the detection of collective and point anomalies. Statistical Analysis and
        DataMining: The ASA Data Science Journal, 15(4), 494-508.

    .. [2] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
        collective and point anomaly detection. Journal of Computational and Graphical
        Statistics, 31(2), 574-585.

    .. [3] Tveten, M., Eckley, I. A., & Fearnhead, P. (2022). Scalable change-point and
        anomaly detection in cross-correlated data with an application to condition
        monitoring. The Annals of Applied Statistics, 16(2), 721-743.

    Examples
    --------
    >>> from sktime.detection._skchange.anomaly_detectors import CAPA
    >>> from sktime.detection._skchange.datasets import generate_piecewise_normal_data
    >>> df = generate_piecewise_normal_data(
    ...     means=[0, 10, 0, 20, 0],
    ...     lengths=[100, 20, 100, 10, 100],
    ...     seed=2,
    ... )
    >>> detector = CAPA()
    >>> detector.fit_predict(df)
            ilocs  labels
    0  [100, 120)       1
    1  [220, 230)       2
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        segment_saving: BaseIntervalScorer | None = None,
        segment_penalty: np.ndarray | float | None = None,
        point_saving: BaseIntervalScorer | None = None,
        point_penalty: np.ndarray | float | None = None,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
        find_affected_components: bool = False,
    ):
        self.segment_saving = segment_saving
        self.segment_penalty = segment_penalty
        self.point_saving = point_saving
        self.point_penalty = point_penalty
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        self.find_affected_components = find_affected_components
        super().__init__()

        _segment_score = L2Saving() if segment_saving is None else segment_saving
        check_interval_scorer(
            _segment_score,
            "segment_saving",
            "CAPA",
            required_tasks=["cost", "saving"],
            allow_penalised=True,
        )
        _segment_saving = to_saving(_segment_score)

        check_penalty(segment_penalty, "segment_penalty", "CAPA")
        if _segment_saving.get_tag("is_penalised"):
            _check_capa_penalised_score(_segment_saving, "segment_saving", "CAPA")
            self._segment_penalised_saving = _segment_saving.clone()
        else:
            self._segment_penalised_saving = PenalisedScore(
                _segment_saving,
                segment_penalty,
                make_default_penalty=_make_chi2_penalty_from_score,
            )
        _point_score = L2Saving() if point_saving is None else point_saving
        check_interval_scorer(
            _point_score,
            "point_saving",
            "CAPA",
            required_tasks=["cost", "saving"],
            allow_penalised=True,
        )
        if _point_score.min_size != 1:
            raise ValueError("`point_saving` must have `min_size == 1`.")
        _point_saving = to_saving(_point_score)

        check_penalty(point_penalty, "point_penalty", "CAPA")
        if _point_saving.get_tag("is_penalised"):
            _check_capa_penalised_score(_point_saving, "point_saving", "CAPA")
            self._point_penalised_saving = _point_saving.clone()
        else:
            self._point_penalised_saving = PenalisedScore(
                _point_saving,
                point_penalty,
                make_default_penalty=_make_linear_chi2_penalty_from_score,
            )

        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

        self.clone_tags(_segment_saving, ["distribution_type"])
        self.set_tags(
            **{"capability:variable_identification": find_affected_components}
        )

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect anomalies in.

        Returns
        -------
        y_sparse: pd.DataFrame
            A `pd.DataFrame` with a range index and two columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.
            * ``"icolumns"`` - list of affected columns for each segment anomaly. Only
              included if `find_affected_components` is ``True``.

        Attributes
        ----------
        fitted_segment_saving : BaseIntervalScorer
            The fitted penalised segment saving used for the detection.
        fitted_point_saving : BaseIntervalScorer
            The fitted penalised point saving used for the detection.
        scores : pd.Series
            The cumulative optimal savings for the input data.
        """
        X = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )

        self.fitted_segment_saving: BaseIntervalScorer = (
            self._segment_penalised_saving.clone().fit(X)
        )
        self.fitted_point_saving: BaseIntervalScorer = (
            self._point_penalised_saving.clone().fit(X)
        )

        min_segment_length = max(
            self.min_segment_length, self.fitted_segment_saving.min_size
        )
        check_smaller_than(
            self.max_segment_length, min_segment_length, "min_segment_length"
        )
        check_larger_than(min_segment_length, X.shape[0], "X.shape[0]")
        opt_savings, segment_anomalies, point_anomalies = run_capa(
            segment_penalised_saving=self.fitted_segment_saving,
            point_penalised_saving=self.fitted_point_saving,
            min_segment_length=min_segment_length,
            max_segment_length=self.max_segment_length,
            find_affected_components=self.find_affected_components,
        )
        self.scores = pd.Series(opt_savings, index=X.index, name="score")

        anomalies = segment_anomalies
        if not self.ignore_point_anomalies:
            anomalies += point_anomalies
        anomalies = sorted(anomalies)

        return self._format_sparse_output(anomalies)

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

        Notes
        -----
        The CAPA scores are the cumulative optimal savings, so the scores are increasing
        and are not per observation scores.
        """
        self.predict(X)
        return self.scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
        from sktime.detection._skchange.anomaly_scores import L2Saving
        from sktime.detection._skchange.compose.penalised_score import PenalisedScore
        from sktime.detection._skchange.costs import L2Cost

        params = [
            {
                "segment_saving": PenalisedScore(
                    L2Saving(),
                    make_default_penalty=_make_nonlinear_chi2_penalty_from_score,
                ),
                "min_segment_length": 5,
                "max_segment_length": 100,
                "find_affected_components": True,
            },
            {
                "segment_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 5,
                "max_segment_length": 100,
            },
            {
                "segment_saving": L2Cost(param=0.0),
                "point_saving": L2Cost(param=0.0),
                "min_segment_length": 2,
                "max_segment_length": 20,
            },
        ]
        return params

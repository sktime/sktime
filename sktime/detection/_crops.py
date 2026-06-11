# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""CROPS algorithm for path solutions to the PELT algorithm."""

__author__ = ["johannvk"]
__all__ = ["CROPS"]

import warnings

import numpy as np
import pandas as pd

from sktime.detection._formatters import format_changepoints
from sktime.detection._pelt import (
    _run_pelt,
    _run_pelt_min_segment_length_one,
    _run_pelt_with_step_size,
)
from sktime.detection._utils import as_2d_array, check_data, check_interval_scorer
from sktime.detection.base import BaseDetector

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evaluate_segmentation(cost, X, segmentation):
    """Evaluate the cost of a segmentation.

    Parameters
    ----------
    cost : BaseCost
        Cost scorer.
    X : np.ndarray
        2D data array.
    segmentation : np.ndarray
        1D array of changepoint indices.

    Returns
    -------
    np.ndarray
        1D cost array.
    """
    if isinstance(segmentation, pd.Series):
        segmentation = segmentation.to_numpy()
    if segmentation.ndim > 1:
        segmentation = segmentation.reshape(-1)

    if len(segmentation) == 0:
        segmentation = np.array([0, X.shape[0]])
    elif segmentation[0] != 0 and segmentation[-1] != X.shape[0]:
        segmentation = np.concatenate(
            (np.array([0]), segmentation, np.array([X.shape[0]]))
        )

    cuts = np.vstack((segmentation[:-1], segmentation[1:])).T
    return np.sum(cost.evaluate(X, cuts), axis=0)


def _format_crops_results(penalty_to_solution_dict, penalty_nudge):
    """Format CROPS results into a DataFrame and lookup dict."""
    list_cpd_results = [
        (len(cpts), penalty, seg_cost, cpts)
        for penalty, (cpts, seg_cost) in penalty_to_solution_dict.items()
    ]
    list_cpd_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    for i in range(len(list_cpd_results) - 1):
        if list_cpd_results[i][0] == list_cpd_results[i + 1][0] + 1:
            low_cost = list_cpd_results[i][2]
            high_cost = list_cpd_results[i + 1][2]
            critical_penalty = (high_cost - low_cost) * (1.0 + penalty_nudge)
            list_cpd_results[i + 1] = (
                list_cpd_results[i + 1][0],
                critical_penalty,
                list_cpd_results[i + 1][2],
                list_cpd_results[i + 1][3],
            )

    encountered = set()
    unique = []
    for item in list_cpd_results:
        if item[0] not in encountered:
            unique.append(item)
            encountered.add(item[0])

    change_points_dict = {len(x[3]): x[3] for x in unique}
    metadata = [x[0:3] for x in unique][::-1]
    df = pd.DataFrame(
        metadata,
        columns=["num_change_points", "penalty", "segmentation_cost"],
    )
    df["num_change_points"] = df["num_change_points"].astype(int)
    df["optimum_value"] = (
        df["segmentation_cost"] + df["penalty"] * df["num_change_points"]
    )
    return df, change_points_dict


def _segmentation_bic_value(cost, X, change_points):
    """Calculate BIC for a given segmentation."""
    n_segments = len(change_points) + 1
    p = X.shape[1]
    n_params = cost.get_model_size(p)
    n_samples = X.shape[0]
    return _evaluate_segmentation(cost, X, change_points)[
        0
    ] + n_params * n_segments * np.log(n_samples)


def _crops_elbow_scores(num_cpts_and_optimum_df):
    """Calculate elbow scores."""
    from sktime.detection._change_scores._continuous_linear_trend_score import (
        _lin_reg_cont_piecewise_linear_trend_score,
    )

    n_seg = len(num_cpts_and_optimum_df)
    if n_seg < 3:
        warnings.warn(
            f"Not enough segmentations ({n_seg}) for elbow cost. "
            "Returning -inf for all."
        )
        return np.full(n_seg, -np.inf)

    times = num_cpts_and_optimum_df["num_change_points"].values
    optim_values = num_cpts_and_optimum_df["optimum_value"].values.reshape(-1, 1)
    elbow_values = _lin_reg_cont_piecewise_linear_trend_score(
        starts=np.repeat(0, n_seg - 2),
        splits=np.arange(1, n_seg - 1),
        ends=np.repeat(n_seg, n_seg - 2),
        X=optim_values,
        times=times,
    )
    return np.concatenate(
        (np.array([-np.inf]), elbow_values.reshape(-1), np.array([-np.inf]))
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CROPS(BaseDetector):
    """CROPS algorithm for path solutions to the PELT algorithm.

    Solves for all penalised optimal partitionings within
    ``[min_penalty, max_penalty]`` using CROPS [1]_, then selects the
    best segmentation via BIC or elbow criterion.

    Parameters
    ----------
    cost : BaseIntervalScorer
        Cost function.
    min_penalty : float
        Lower bound of penalty range.
    max_penalty : float
        Upper bound of penalty range.
    selection_method : str, default="bic"
        ``"bic"`` or ``"elbow"``.
    min_segment_length : int, default=1
        Minimum segment length.
    step_size : int, default=1
        JumpPELT step size.
    split_cost : float, default=0.0
        Additive split cost.
    prune : bool, default=True
        Enable pruning.
    pruning_margin : float, default=0.0
        Pruning margin.
    middle_penalty_nudge : float, default=1e-5
        Nudge factor for threshold penalties.

    References
    ----------
    .. [1] Haynes, K., Eckley, I. A. & Fearnhead, P. (2017). Computationally
       efficient changepoint detection for a range of penalties. JCGS, 26(1),
       134-143.

    Examples
    --------
    >>> from sktime.detection._crops import CROPS
    >>> from sktime.detection._costs._l2_cost import L2Cost
    >>> import numpy as np
    >>> X = np.concatenate([np.zeros(50), 10*np.ones(50)])
    >>> det = CROPS(cost=L2Cost(), min_penalty=0.5, max_penalty=50.0)
    >>> det.fit(X).predict(X)  # doctest: +SKIP
    """

    _tags = {
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "fit_is_empty": True,
        "authors": ["johannvk"],
        "maintainers": ["johannvk"],
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        cost,
        min_penalty,
        max_penalty,
        selection_method="bic",
        min_segment_length=1,
        step_size=1,
        split_cost=0.0,
        prune=True,
        pruning_margin=0.0,
        middle_penalty_nudge=1.0e-5,
    ):
        self.cost = cost
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.selection_method = selection_method
        self.min_segment_length = min_segment_length
        self.step_size = step_size
        self.split_cost = split_cost
        self.prune = prune
        self.pruning_margin = pruning_margin
        self.middle_penalty_nudge = middle_penalty_nudge
        super().__init__()

        self._cost = cost.clone()
        check_interval_scorer(
            self._cost,
            arg_name="cost",
            caller_name="CROPS",
            required_tasks=["cost"],
            allow_penalised=False,
        )

        if self.step_size > 1 and self.min_segment_length > self.step_size:
            raise ValueError(
                f"CROPS `min_segment_length`(={self.min_segment_length}) cannot be "
                f"greater than the `step_size`(={self.step_size}) > 1."
            )

        if selection_method not in ("bic", "elbow"):
            raise ValueError(
                f"Invalid selection_method: {selection_method}. "
                "Must be 'bic' or 'elbow'."
            )

        self.change_points_metadata = None
        self.change_points_lookup = None
        self.optimal_penalty = None

    def _fit(self, X, y=None):
        return self

    def _solve_for_changepoints(self, X_arr, penalty):
        """Solve PELT for a single penalty value."""
        if self.step_size > 1:
            result = _run_pelt_with_step_size(
                self._cost,
                X_arr,
                penalty=penalty,
                step_size=self.step_size,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        elif self.min_segment_length == 1:
            result = _run_pelt_min_segment_length_one(
                self._cost,
                X_arr,
                penalty=penalty,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        else:
            result = _run_pelt(
                self._cost,
                X_arr,
                penalty=penalty,
                min_segment_length=self.min_segment_length,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        return result.changepoints

    def _run_crops(self, X_arr):
        """Run CROPS binary search over penalty range."""
        min_cpts = self._solve_for_changepoints(X_arr, self.min_penalty)
        max_cpts = self._solve_for_changepoints(X_arr, self.max_penalty)

        min_cost = _evaluate_segmentation(self._cost, X_arr, min_cpts)[0]
        max_cost = _evaluate_segmentation(self._cost, X_arr, max_cpts)[0]

        penalty_to_solution = {
            self.min_penalty: (min_cpts, min_cost),
            self.max_penalty: (max_cpts, max_cost),
        }

        search_intervals = []
        if len(min_cpts) > len(max_cpts) + 1:
            search_intervals.append((self.min_penalty, self.max_penalty))

        while search_intervals:
            low_pen, high_pen = search_intervals.pop(0)
            low_cpts, low_cost = penalty_to_solution[low_pen]
            high_cpts, high_cost = penalty_to_solution[high_pen]

            if len(low_cpts) > len(high_cpts) + 1:
                threshold = (high_cost - low_cost) / (len(low_cpts) - len(high_cpts))
                additive = (
                    threshold + (high_pen - threshold) * self.middle_penalty_nudge
                )
                multiplicative = threshold * (1.0 + self.middle_penalty_nudge)
                mid_pen = min(additive, multiplicative)

                mid_cpts = self._solve_for_changepoints(X_arr, mid_pen)
                mid_cost = _evaluate_segmentation(self._cost, X_arr, mid_cpts)[0]
                penalty_to_solution[mid_pen] = (mid_cpts, mid_cost)

                if len(mid_cpts) == len(high_cpts):
                    continue
                elif len(mid_cpts) == len(low_cpts):
                    raise ValueError(
                        "PELT not solved exactly. Try non-zero `split_cost` "
                        "or increase `pruning_margin`."
                    )
                else:
                    search_intervals.append((low_pen, mid_pen))
                    search_intervals.append((mid_pen, high_pen))
                    search_intervals.sort(key=lambda x: x[0])

        df, lookup = _format_crops_results(
            penalty_to_solution, self.middle_penalty_nudge
        )
        return df, lookup

    def _predict(self, X):
        X_df = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )
        X_arr = as_2d_array(X_df)

        if X_arr.shape[1] > 1 and not self._cost.get_tag("is_aggregated"):
            raise ValueError(
                "CROPS only supports aggregated costs for multivariate data."
            )

        metadata, lookup = self._run_crops(X_arr)

        if self.selection_method == "elbow":
            change_slope_df = metadata[["num_change_points", "optimum_value"]].copy()
            change_slope_df["optimum_value"] = (
                change_slope_df["optimum_value"]
                - change_slope_df["optimum_value"].min()
            )
            metadata["elbow_score"] = _crops_elbow_scores(change_slope_df)
            opt_n = metadata.sort_values(by="elbow_score", ascending=False)[
                "num_change_points"
            ].iloc[0]
        else:
            metadata["bic_value"] = metadata["num_change_points"].apply(
                lambda n_cpts: _segmentation_bic_value(
                    self._cost, X_arr, lookup[n_cpts]
                )
            )
            opt_n = metadata.sort_values(by="bic_value")["num_change_points"].iloc[0]

        return format_changepoints(lookup[opt_n])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._costs._l2_cost import L2Cost

        return [
            {
                "cost": L2Cost(),
                "min_penalty": 0.5,
                "max_penalty": 50.0,
                "selection_method": "bic",
                "min_segment_length": 2,
            },
            {
                "cost": L2Cost(),
                "min_penalty": 1.0,
                "max_penalty": 10.0,
                "selection_method": "elbow",
                "min_segment_length": 2,
                "prune": False,
            },
        ]

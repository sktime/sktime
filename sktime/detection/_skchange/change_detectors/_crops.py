"""CROPS algorithm for path solutions to the PELT algorithm."""

import warnings

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..change_detectors._pelt import (
    _run_pelt,
    _run_pelt_min_segment_length_one,
    _run_pelt_with_step_size,
)
from ..change_scores._continuous_linear_trend_score import (
    lin_reg_cont_piecewise_linear_trend_score,
)
from ..costs.base import BaseCost
from ..utils.validation.interval_scorer import check_interval_scorer
from .base import BaseChangeDetector


def evaluate_segmentation(
    cost: BaseCost, segmentation: np.ndarray | pd.Series
) -> np.ndarray:
    """Evaluate the cost of a segmentation.

    Parameters
    ----------
    segmentation : np.ndarray
        A 1D array with the indices of the change points in the input data.
        Each change point signifies the first index of a new segment.

    Returns
    -------
    cost : float
        The cost of the segmentation.
    """
    cost.check_is_fitted()

    if isinstance(segmentation, pd.Series):
        segmentation = segmentation.to_numpy()
    if segmentation.ndim != 1 and segmentation.shape[1] != 1:
        raise ValueError("The segmentation must univariate")
    else:
        segmentation = segmentation.reshape(-1)

    # Prepend 0 and append the length of the data to the segmentation:
    # This is done to ensure that the first and last segments are included.
    # The segmentation is assumed to be sorted in increasing order.
    if np.any(np.diff(segmentation) <= 0):
        raise ValueError("The segmentation must contain strictly increasing entries.")
    if len(segmentation) == 0:
        segmentation = np.array([0, cost.n_samples])
    elif segmentation[0] != 0 and segmentation[-1] != cost.n_samples:
        segmentation = np.concatenate(
            (np.array([0]), segmentation, np.array([cost.n_samples]))
        )

    cuts = np.vstack((segmentation[:-1], segmentation[1:])).T

    # Aggregate the partition intervals into a 1D array of values:
    return np.sum(cost.evaluate(cuts), axis=0)


def _format_crops_results(
    penalty_to_solution_dict: dict[float, tuple[np.ndarray, float]],
    penalty_nudge: float,
) -> tuple[pd.DataFrame, dict]:
    """Format the CROPS results into a DataFrame.

    Parameters
    ----------
    cpd_results : dict
        Dictionary with penalty values as keys and tuples of change points and
        segmentation costs as values.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame with columns 'num_change_points', 'penalty', and 'segmentation_cost'.
        Dictionary with number of change points as keys and change points as values.
    """
    # Convert the dict lookup results to a list of tuples:
    list_cpd_results = [
        (len(change_points), penalty, segmentation_cost, change_points)
        for (
            penalty,
            (change_points, segmentation_cost),
        ) in penalty_to_solution_dict.items()
    ]

    # When different penalties give the same number of change points, we want to
    # keep the segmentation with the lowest penalty. So we sort by number of
    # change points, and then negative penalty, in reverse order.
    list_cpd_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    # If there are solutions that only differ by a single changepoint when we increase
    # the penalty, we can compute the penalty for which the number of change points
    # changes when increasing penalty from the lower value to the higher value.
    for i in range(len(list_cpd_results) - 1):
        # If the number of change points is the same, and the penalty is different,
        # we can compute the penalty for which the number of change points changes
        # when increasing penalty from the lower value to the higher value.
        lower_penalty_num_change_points = list_cpd_results[i][0]
        higher_penalty_num_change_points = list_cpd_results[i + 1][0]
        if lower_penalty_num_change_points == higher_penalty_num_change_points + 1:
            # Difference in segmentation cost is the critical penalty value.
            # Multiply by (1.0 + penalty_nudge) to avoid numerical instability.
            lower_penalty_segmentation_cost = list_cpd_results[i][2]
            higher_penalty_segmentation_cost = list_cpd_results[i + 1][2]
            critical_penalty = (
                higher_penalty_segmentation_cost - lower_penalty_segmentation_cost
            ) * (1.0 + penalty_nudge)

            # Replace the penalty value with the critical penalty value.
            list_cpd_results[i + 1] = (
                list_cpd_results[i + 1][0],
                critical_penalty,
                list_cpd_results[i + 1][2],
                list_cpd_results[i + 1][3],
            )

    # Remove duplicates, and keep the first one:
    encountered_num_change_points = set()
    unique_cpd_results = []
    for i in range(len(list_cpd_results)):
        num_change_points = list_cpd_results[i][0]
        if num_change_points in encountered_num_change_points:
            continue
        else:
            unique_cpd_results.append(list_cpd_results[i])
            encountered_num_change_points.add(num_change_points)

    # Extract out the change points into a dict with the number of change points
    # as the key, and the change points as the value.
    change_points_dict = {len(x[3]): x[3] for x in unique_cpd_results}

    # Extract out the change points metadata into a DataFrame.
    # Reverse the order of the list, so that the lowest number
    # of change points comes first.
    change_points_metadata = [x[0:3] for x in unique_cpd_results][::-1]
    penalty_change_point_metadata_df = pd.DataFrame(
        change_points_metadata,
        columns=[
            "num_change_points",
            "penalty",
            "segmentation_cost",
        ],
    )

    # Ensure that the number of change points is an integer:
    penalty_change_point_metadata_df["num_change_points"] = (
        penalty_change_point_metadata_df["num_change_points"].astype(int)
    )

    penalty_change_point_metadata_df["optimum_value"] = (
        penalty_change_point_metadata_df["segmentation_cost"]
        + penalty_change_point_metadata_df["penalty"]
        * penalty_change_point_metadata_df["num_change_points"]
    )

    return penalty_change_point_metadata_df, change_points_dict


def segmentation_bic_value(cost: BaseCost, change_points: np.ndarray) -> float:
    """Calculate the BIC score for a given segmentation.

    Parameters
    ----------
    cost : BaseCost
        The cost function to use.
    change_points : np.ndarray
        The change points to use.

    Returns
    -------
    float
        The BIC score for the given segmentation.
    """
    cost.check_is_fitted()
    num_segments = len(change_points) + 1
    data_dim = cost.n_variables
    num_parameters = cost.get_model_size(data_dim)
    n_samples = cost.n_samples

    bic_score = evaluate_segmentation(cost, change_points)[
        0
    ] + num_parameters * num_segments * np.log(n_samples)

    return bic_score


def crops_elbow_scores(
    num_change_points_and_optimum_value_df: pd.DataFrame,
) -> np.ndarray:
    """Calculate the elbow cost for a given segmentation.

    Specifically, the elbow score is calculated as the evidence for a change
    of slope in the segmentation cost as a function of the number of
    change points, at each intermediate number of change points.
    We cannot calculate the elbow score for the first and last number of
    change points, as there are not enough segmentations to calculate a change
    in slope before or after the first and last number of change points.

    Parameters
    ----------
    num_change_points : int
        The number of change points.
    segmentation_cost : float
        The segmentation cost.

    Returns
    -------
    pd.Series
        The elbow cost for each number of change points.
    """
    num_segmentations = len(num_change_points_and_optimum_value_df)
    if num_segmentations < 3:
        # Not enough segmentations to calculate the elbow cost.
        warnings.warn(
            f"Not enough segmentations {num_segmentations} to calculate "
            "the elbow cost. Returning -np.inf for all segmentations."
        )
        return pd.Series([-np.inf] * num_segmentations)

    # Calculate the elbow (change in slope) cost for each number of change points:
    times = num_change_points_and_optimum_value_df["num_change_points"].values
    optim_values = num_change_points_and_optimum_value_df[
        "optimum_value"
    ].values.reshape(-1, 1)
    elbow_values = lin_reg_cont_piecewise_linear_trend_score(
        starts=np.repeat(0, num_segmentations - 2),
        splits=np.arange(1, num_segmentations - 1),
        ends=np.repeat(num_segmentations, num_segmentations - 2),
        X=optim_values,
        times=times,
    )

    # Pad the elbow score with `-np.inf` for the first and last segmentations,
    # which we cannot calculate a "change in slope" score for.
    elbow_scores = np.concatenate(
        (
            np.array([-np.inf]),
            elbow_values.reshape(-1),
            np.array([-np.inf]),
        )
    )

    return elbow_scores


class CROPS(BaseChangeDetector):
    """CROPS algorithm for path solutions to the `PELT` algorithm.

    This change detector solves for all penalized optimal partitionings
    within the penalty range `[min_penalty, max_penalty]`, using the CROPS
    algorithm[1]_, which in turn employs the `PELT` algorithm to repeatedly
    solve penalized optimal partitioning problems for different penalties.

    When predicting change points through `predict()`, this change detector
    selects the best segmentation among the optimal partitionings within
    the penalty range, using the `selection_method` criterion.

    Parameters
    ----------
    cost : BaseIntervalScorer
        The cost function to use.
    min_penalty : float
        The start of the penalty solution interval.
    max_penalty : float
        The end of the penalty solution interval.
    selection_method : str, default="bic"
        The segmentation selection method to use when selecting the
        best segmentation among the optimal segmentations found
        within the penalty range `[min_penalty, max_penalty]`.
        The options are:

        * ``"bic"``: Select the segmentation with the lowest Bayesian Information
          criterion, defined as `segmentation_cost + model_size_per_segment *
          log(n_samples)`.
        * ``"elbow"``: Select the segmentation with the highest elbow score.
          Defined as the improvement in squared residuals when allowing
          a change in slope of segmentation cost regressed on the number
          of change points, at each intermediate number of change points.

    min_segment_length : int, default=1
        The minimum segment length to use.
    split_cost : float, optional, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    prune: bool, optional
        If False, drop the pruning step, reverting to optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional, default=0.0
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful when the cost function is imprecise, e.g.
        based on solving an optimization problem with large tolerance.
    middle_penalty_nudge : float, optional, default=1.0e-4
        When computing the threshold penalty value separating `PELT `solutions
        with differing numbers of change points, we need to nudge the penalty
        upwards in order to solve for the segmentation with fewer change points.
        By default set to 1.0e-4, which is sufficient for most cases.

    References
    ----------
    .. [1] Haynes, K., Eckley, I. A., & Fearnhead, P. (2017). Computationally efficient
    changepoint detection for a range of penalties. Journal of Computational and
    Graphical Statistics, 26(1), 134-143.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": ["johannvk"],
        "fit_is_empty": True,
    }

    def __init__(
        self,
        cost: BaseIntervalScorer,
        min_penalty: float,
        max_penalty: float,
        selection_method: str = "bic",
        min_segment_length: int = 1,
        step_size: int = 1,
        split_cost: float = 0.0,
        prune: bool = True,
        pruning_margin: float = 0.0,
        middle_penalty_nudge: float = 1.0e-5,
    ):
        super().__init__()
        self.cost = cost
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.selection_method = selection_method
        self.min_segment_length = min_segment_length
        self.step_size = step_size
        self.split_cost = split_cost
        self.pruning_margin = pruning_margin
        self.middle_penalty_nudge = middle_penalty_nudge
        self.prune = prune

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

        if selection_method not in ["bic", "elbow"]:
            raise ValueError(
                f"Invalid selection criterion: {selection_method}. "
                "Must be one of ['bic', 'elbow']."
            )

        # Storage for the CROPS results:
        self.change_points_metadata: pd.DataFrame | None = None
        self.change_points_lookup: dict[int, np.ndarray] | None = None
        self.optimal_penalty: float | None = None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Run the CROPS algorithm for path solutions to penalized CPD.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.

        Returns
        -------
        np.ndarray
            The optimal change points for penalties within
            `[self.min_penalty, self.max_penalty]`, as decided by the
            `segmentation_selection` criterion.

        Attributes
        ----------
        optimal_penalty : float
            The penalty value for which the optimal change points were found,
            w.r.t. the `segmentation_selection` criterion.
        """
        if X.ndim > 1 and X.shape[1] > 1 and not self._cost.get_tag("is_aggregated"):
            raise ValueError(
                "CROPS only supports costs that return a single value per cut "
                "when the input data has more than one column. "
                "Please use an aggregated cost function."
            )
        self._run_crops(X=X)

        if self.selection_method == "elbow":
            # Select the best segmentation using the elbow criterion.
            change_in_slope_df = self.change_points_metadata[
                ["num_change_points", "optimum_value"]
            ].copy()

            # Subrtract the minimum value from the optimum value
            # to improve conditioning number of linreg problems.
            change_in_slope_df["optimum_value"] = (
                change_in_slope_df["optimum_value"]
                - change_in_slope_df["optimum_value"].min()
            )
            self.change_points_metadata["elbow_score"] = crops_elbow_scores(
                change_in_slope_df,
            )
            optimal_num_change_points, optimal_penalty = (
                self.change_points_metadata.sort_values(
                    by="elbow_score", ascending=False
                )[["num_change_points", "penalty"]].iloc[0]
            )

        elif self.selection_method == "bic":
            # Select the best segmentation using the BIC criterion.
            self.change_points_metadata["bic_value"] = self.change_points_metadata[
                "num_change_points"
            ].apply(
                lambda num_change_points: segmentation_bic_value(
                    cost=self._cost,
                    change_points=self.change_points_lookup[num_change_points],
                )
            )
            optimal_num_change_points, optimal_penalty = (
                self.change_points_metadata.sort_values(by="bic_value")[
                    ["num_change_points", "penalty"]
                ].iloc[0]
            )

        self.optimal_penalty = optimal_penalty
        return self.change_points_lookup[optimal_num_change_points]

    def _solve_for_changepoints(self, penalty: float) -> np.ndarray:
        """Solve for the optimal changepoints given `penalty` using PELT.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.

        Returns
        -------
        changepoints : np.ndarray
            The optimal change points for the given penalty.
        """
        if self.step_size > 1:
            pelt_result = _run_pelt_with_step_size(
                self._cost,
                penalty=penalty,
                step_size=self.step_size,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        elif self.min_segment_length == 1:
            pelt_result = _run_pelt_min_segment_length_one(
                self._cost,
                penalty=penalty,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        else:
            pelt_result = _run_pelt(
                self._cost,
                penalty=penalty,
                min_segment_length=self.min_segment_length,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )

        return pelt_result.changepoints

    def _run_crops(self, X: np.ndarray) -> pd.DataFrame:
        """Run the CROPS algorithm for path solutions to penalized CPD on data `X`.

        Parameters
        ----------
        X : np.ndarray
            Data to search for change points in.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'num_change_points', 'penalty',
            and 'segmentation_cost'.

        Attributes
        ----------
        change_points_metadata : pd.DataFrame
            DataFrame with columns 'num_change_points', 'penalty',
            and 'segmentation_cost'.
            Contains metadata about the change points found by the CROPS algorithm.
        change_points_lookup : dict[int, np.ndarray]
            Dictionary with number of change points as keys and change points as values.
            The keys are the number of change points, and the values are arrays with the
            change point indices for that number of change points.
        """
        self._cost.fit(X)

        min_penalty_change_points = self._solve_for_changepoints(
            penalty=self.min_penalty
        )
        max_penalty_change_points = self._solve_for_changepoints(
            penalty=self.max_penalty
        )

        num_min_penalty_change_points = len(min_penalty_change_points)
        num_max_penalty_change_points = len(max_penalty_change_points)

        min_penalty_segmentation_cost = evaluate_segmentation(
            self._cost, min_penalty_change_points
        )[0]
        max_penalty_segmentation_cost = evaluate_segmentation(
            self._cost, max_penalty_change_points
        )[0]

        penalty_to_solution_dict: dict[float, (np.ndarray, float)] = dict()
        penalty_to_solution_dict[self.min_penalty] = (
            min_penalty_change_points,
            min_penalty_segmentation_cost,
        )
        penalty_to_solution_dict[self.max_penalty] = (
            max_penalty_change_points,
            max_penalty_segmentation_cost,
        )

        # Store intervals of penalty values in which to search for
        # differing numbers of change points.
        penalty_search_intervals = []
        if num_min_penalty_change_points > num_max_penalty_change_points + 1:
            # More than one change point in difference between min and max penalty.
            # Need to split the penalty intervals further.
            penalty_search_intervals.append((self.min_penalty, self.max_penalty))

        while len(penalty_search_intervals) > 0:
            # Pop the interval with the lowest penalty.
            low_penalty, high_penalty = penalty_search_intervals.pop(0)

            low_penalty_change_points, low_penalty_segmentation_cost = (
                penalty_to_solution_dict[low_penalty]
            )
            num_low_penalty_change_points = len(low_penalty_change_points)

            high_penalty_change_points, high_penalty_segmentation_cost = (
                penalty_to_solution_dict[high_penalty]
            )
            num_high_penalty_change_points = len(high_penalty_change_points)

            if num_low_penalty_change_points > (num_high_penalty_change_points + 1):
                # Compute the threshold penalty value, where the number of change
                # points decreases.
                threshold_penalty = (
                    high_penalty_segmentation_cost - low_penalty_segmentation_cost
                ) / (num_low_penalty_change_points - num_high_penalty_change_points)

                # Nudge the middle penalty towards the high penalty, to ensure that
                # the number of change points for the middle penalty is fewer than
                # for the low penalty.
                additive_nudged_middle_penalty = (
                    threshold_penalty
                    + (high_penalty - threshold_penalty) * self.middle_penalty_nudge
                )
                multiplicative_nudged_middle_penalty = threshold_penalty * (
                    1.0 + self.middle_penalty_nudge
                )
                # Middle penalty is the minimum of the two nudged values:
                middle_penalty = min(
                    additive_nudged_middle_penalty, multiplicative_nudged_middle_penalty
                )

                middle_penalty_change_points = self._solve_for_changepoints(
                    penalty=middle_penalty
                )
                middle_penalty_segmentation_cost = evaluate_segmentation(
                    self._cost, middle_penalty_change_points
                )[0]
                penalty_to_solution_dict[middle_penalty] = (
                    middle_penalty_change_points,
                    middle_penalty_segmentation_cost,
                )

                middle_penalty_matches_high_penalty = len(
                    middle_penalty_change_points
                ) == len(high_penalty_change_points)
                middle_penalty_matches_low_penalty = len(
                    middle_penalty_change_points
                ) == len(low_penalty_change_points)

                if middle_penalty_matches_high_penalty:
                    # The same number of change points for penalties in
                    # the interval [middle_penalty, high_penalty].
                    # Don't need to subdivide penalty intervals further.
                    continue
                elif middle_penalty_matches_low_penalty:
                    raise ValueError(  # pragma: no cover
                        "PELT optimization has not been solved exactly! "
                        "Number of change points should be greater for the "
                        "middle penalty than for the low penalty. "
                        "Attempt to set the `split_cost` parameter to a "
                        "non-zero value, or increase `pruning_margin`."
                    )
                else:
                    # Number of change points for middle penalty is different from both
                    # low_penalty and high_penalty. Need to explore further.
                    penalty_search_intervals.append((low_penalty, middle_penalty))
                    penalty_search_intervals.append((middle_penalty, high_penalty))

                    # Sort the intervals by lower penalty:
                    penalty_search_intervals.sort(key=lambda x: x[0])

        metadata_df, change_points_dict = _format_crops_results(
            penalty_to_solution_dict=penalty_to_solution_dict,
            penalty_nudge=self.middle_penalty_nudge,
        )
        self.change_points_metadata = metadata_df
        self.change_points_lookup = change_points_dict

        return self.change_points_metadata

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> list[dict]:
        """Get test parameters for the CROPS algorithm."""
        from sktime.detection._skchange.costs import L2Cost

        return [
            {
                "cost": L2Cost(),
                "min_penalty": 0.5,
                "max_penalty": 50.0,
                "selection_method": "bic",
                "min_segment_length": 10,
                "split_cost": 0.0,
                "pruning_margin": 0.0,
                "middle_penalty_nudge": 1.0e-4,
                "prune": True,
            },
            {
                "cost": L2Cost(),
                "min_penalty": 1.0,
                "max_penalty": 10.0,
                "selection_method": "elbow",
                "min_segment_length": 2,
                "split_cost": 0.0,
                "pruning_margin": 0.01,
                "middle_penalty_nudge": 1.0e-4,
                "prune": False,
            },
        ]

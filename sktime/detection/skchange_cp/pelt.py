"""Compatibility wrapper for PELT."""

from sktime.detection._skchange.change_detectors import PELT as _PELT


class PELT(_PELT):
    """Pruned exact linear time (PELT) changepoint detection.

    Implements the PELT algorithm [1]_ for changepoint detection.
    This method solves the penalized optimal partitioning problem,
    with pruning of the admissible starts set applied to improve performance.

    One can specify a minimum segment length for the partitions considered
    when detecting changepoints through the `min_segment_length` parameter,
    and when the minimum segment length is greater than one we use deferred
    pruning of the admissible starts [2]_ to ensure exact solutions.

    Additionally, one can specify a step size through the `step_size` parameter,
    which coarsens the search space for changepoints, allowing for faster detection
    at the cost of change point location granularity.

    Parameters
    ----------
    cost : BaseIntervalScorer, optional, default=`L2Cost`
        The cost to use for the changepoint detection. Expects a `BaseIntervalScorer`
        instance that implements the `cost` task. If `None`, defaults to `L2Cost`.
    penalty : float, optional
        The penalty to use for the changepoint detection. It must be non-negative. If
        `None`, the penalty is set to
        `make_bic_penalty(n=X.shape[0], n_params=cost.get_model_size(X.shape[1]))``,
        where ``X`` is the input data to `predict` changepoints in.
    min_segment_length : int, optional, default=1
        Minimum length of a segment. The minimum length of a segment to consider
        when detecting changepoints. Must be at least 1. If `step_size` is greater than
        1, this must be less than or equal to `step_size`.
    step_size: bool, optional, default=False
        If True, only indices that are multiples of `step_size` from the
        first data point (index `0`) are considered as potential changepoints.
        Implicitly ensures that `min_segment_length >= step_size`, but it's
        an error to specify `min_segment_length` greater than `step_size`.
    split_cost : float, optional, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)])``,
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    prune : bool, optional, default=False
        If True, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing. By default set to False.
    pruning_margin : float, optional, default=0.0
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American Statistical
    Association, 107(500), 1590-1598.

    .. [2] Bakka, Kristin Benedicte (2018). Changepoint model selection in Gaussian data
    by maximization of approximate Bayes Factors with the Pruned Exact Linear Time
    algorithm. Master's thesis, Norwegian University of Science and Technology (NTNU).
    URL: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2558597.
    """

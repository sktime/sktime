"""Compatibility wrapper for SeededBinarySegmentation."""

from sktime.detection._skchange.change_detectors import (
    SeededBinarySegmentation as _SeededBinarySegmentation,
)


class SeededBinarySegmentation(_SeededBinarySegmentation):
    """Seeded binary segmentation algorithm for multiple changepoint detection.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. The seeded
    binary segmentation algorithm is an efficient version of such algorithms, which
    tests for changepoints in intervals of exponentially growing length. It has the same
    theoretical guarantees as the original binary segmentation algorithm, but runs
    in log-linear time no matter the changepoint configuration.

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

    max_interval_length : int, default=200
        The maximum length of an interval to estimate a changepoint in. Must be greater
        than or equal to ``2 * change_score.min_size``.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size according to
        ``interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))```,
        starting at ``interval_len=min_interval_length``. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of ``1 + 1 / growth_factor``. Must be a float in
        ``(1, 2]``.
    selection_method : str, default="greedy"
        The method to use for selecting changepoints. The options are:

        * ``"greedy"``: Selects the changepoint with the highest score, then removes all
          intervals that contain the detected changepoint. This process is repeated
          until no intervals are left with a score above the threshold.
        * ``"narrowest"``: Searches among the intervals with scores above the threshold,
          and selects the one with the narrowest interval. It then removes all
          intervals that contain the detected changepoint, and repeats these two steps
          until no intervals are left with a score above the threshold.

    References
    ----------
    .. [1] Kovács, S., Bühlmann, P., Li, H., & Munk, A. (2023). Seeded binary
        segmentation: a general methodology for fast and optimal changepoint detection.
        Biometrika, 110(1), 249-256.

    .. [2] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). Narrowest-over-threshold
        detection of multiple change points and change-point-like features. Journal of
        the Royal Statistical Society Series B: Statistical Methodology, 81(3), 649-672.
    """

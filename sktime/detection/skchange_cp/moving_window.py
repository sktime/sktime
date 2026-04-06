"""Compatibility wrapper for MovingWindow."""

from sktime.detection._skchange.change_detectors import MovingWindow as _MovingWindow


class MovingWindow(_MovingWindow):
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
    """

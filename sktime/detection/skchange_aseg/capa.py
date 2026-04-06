"""Compatibility wrapper for CAPA."""

from sktime.detection._skchange.anomaly_detectors import CAPA as _CAPA


class CAPA(_CAPA):
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

"""Circular binary segmentation algorithm for multiple changepoint detection."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record("skchange.anomaly_detectors.circular_binseg")
class CircularBinarySegmentation(BaseDetector):
    """Circular binary segmentation algorithm for anomalous segment detection, skchange.

    Redirects to ``skchange.anomaly_detectors.circular_binseg``.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. Circular binary
    segmentation [1]_ is a variant of binary segmentation where the statistical test
    (anomaly score) is applied to compare the data behaviour of an inner interval subset
    with the surrounding data contained in an outer interval.

    In other words, the null hypothesis within each outer interval is that the data
    is stationary, while the alternative hypothesis is that there is a collective
    anomaly within the outer interval.

    Efficently implemented using numba.

    Parameters
    ----------
    anomaly_score : BaseLocalAnomalyScore or BaseCost, optional (default=L2Cost())
        The local anomaly score to use for anomaly detection. If a cost is given, it is
        converted to a local anomaly score using the `LocalAnomalyScore` class.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        `threshold_scale * 2 * p * np.sqrt(np.log(n))`, where `n` is the sample size
        and `p` is the number of variables. If None, the threshold is tuned on the
        data input to `.fit()`.
    level : float, default=0.01
        If `threshold_scale` is None, the threshold is set to the (1-`level`)-quantile
        of the changepoint scores of all the seeded intervals on the training data.
        For this to be correct, the training data must contain no changepoints.
    min_segment_length : int, default=5
        Minimum length between two changepoints. Must be greater than or equal to 1.
    max_interval_length : int, default=100
        The maximum length of an interval to estimate a changepoint in. Must be greater
        than or equal to `2 * min_segment_length`.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size according to
        `interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))`,
        starting at `interval_len=min_interval_length`. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of `1 + 1 / growth_factor`. Must be a float in
        `(1, 2]`.

    References
    ----------
    .. [1] Olshen, A. B., Venkatraman, E. S., Lucito, R., & Wigler, M. (2004). Circular
    binary segmentation for the analysis of array-based DNA copy number data.
    Biostatistics, 5(4), 557-572.

    Examples
    --------
    >>> from skchange.anomaly_detectors import CircularBinarySegmentation
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=5, mean=10, segment_length=20)
    >>> detector = CircularBinarySegmentation()
    >>> detector.fit_predict(df)
    0    [20, 40)
    1    [60, 80)
    Name: anomaly_interval, dtype: interval

    Notes
    -----
    Using costs to generate local anomaly scores will be significantly slower than using
    anomaly scores that are implemented directly. This is because the local anomaly
    score requires evaluating the cost at disjoint subsets of the data
    (before and after an anomaly), which is not a natural operation for costs
    implemented as interval evaluators.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
        "python_dependencies": "skchange>=0.6.0",
        # estimator type
        # --------------
        "task": "segmentation",
        "learning_type": "unsupervised",
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        anomaly_score=None,
        threshold_scale=None,
        level: float = 1e-8,
        min_segment_length: int = 5,
        max_interval_length: int = 1000,
        growth_factor: float = 1.5,
    ):
        self.anomaly_score = anomaly_score
        self.threshold_scale = threshold_scale  # Just holds the input value.
        self.level = level
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        super().__init__()

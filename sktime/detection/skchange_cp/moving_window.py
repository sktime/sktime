"""The Moving Window algorithm for multiple changepoint detection."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record("skchange.change_detectors.moving_window")
class MovingWindow(BaseDetector):
    """Moving window algorithm for multiple changepoint detection, from skchange.

    Redirects to ``skchange.change_detectors.moving_window``.

    A generalized version of the MOSUM (moving sum) algorithm [1]_ for changepoint
    detection. It runs a test statistic for a single changepoint at the midpoint in a
    moving window of length ``2 * bandwidth`` over the data. Efficiently implemented
    using numba.

    Parameters
    ----------
    change_score : BaseChangeScore or BaseCost, optional (default=``L2Cost``)
        The change score to use in the algorithm. If a cost function is given, it is
        converted to a change score using the ``ChangeScore`` class.
    bandwidth : int, default=30
        The bandwidth is the number of samples on either side of a candidate
        changepoint. The minimum bandwidth depends on the
        test statistic. For ``"mean"``, the minimum bandwidth is 1.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        ``threshold_scale * default_threshold``, where the default threshold depends on
        the number of samples, the number of variables, ``bandwidth`` and ``level``.
        If None, the threshold is tuned on the input data to ``fit``.
    level : float, default=0.01
        If ``threshold_scale`` is ``None``, the threshold is set to the
        (1-``level``)-quantile of the changepoint score on the training data. For this
        to be correct, the training data must contain no changepoints. If
        ``threshold_scale`` is a number, ``level`` is used in the default threshold,
        _before_ scaling.
    min_detection_interval : int, default=1
        Minimum number of consecutive scores above the threshold to be considered a
        changepoint. Must be between 1 and ``bandwidth/2``.

    References
    ----------
    .. [1] Eichinger, B., & Kirch, C. (2018). A MOSUM procedure for the estimation of
    multiple random change points.

    Examples
    --------
    >>> from skchange.change_detectors import MovingWindow
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(
            n_segments=4, mean=10, segment_length=100, p=5
        )
    >>> detector = MovingWindow()
    >>> detector.fit_predict(df)
    0    100
    1    200
    2    300
    Name: changepoint, dtype: int64
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
        "python_dependencies": "skchange>=0.6.0",
        # estimator type
        # --------------
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        change_score=None,
        bandwidth: int = 30,
        threshold_scale=2.0,
        level: float = 0.01,
        min_detection_interval: int = 1,
    ):
        self.change_score = change_score
        self.bandwidth = bandwidth
        self.threshold_scale = threshold_scale
        self.level = level
        self.min_detection_interval = min_detection_interval

        super().__init__()

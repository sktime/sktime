"""Seeded binary segmentation algorithm for multiple changepoint detection."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record("skchange.change_detectors.seeded_binseg")
class SeededBinarySegmentation(BaseDetector):
    """Seeded binary segmentation algorithm for changepoint detection, from skchange.

    Redirects to ``skchange.change_detectors.seeded_binseg``.

    Binary segmentation type changepoint detection algorithms recursively split the data
    into two segments, and test whether the two segments are different. The seeded
    binary segmentation algorithm is an efficient version of such algorithms, which
    tests for changepoints in intervals of exponentially growing length. It has the same
    theoretical guarantees as the original binary segmentation algorithm, but runs
    in log-linear time no matter the changepoint configuration.

    Efficiently implemented using numba.

    Parameters
    ----------
    change_score : BaseChangeScore or BaseCost, default=L2Cost()
        The change score to use in the algorithm. If a cost function is given, it is
        converted to a change score using the ``ChangeScore`` class.
    threshold_scale : float, default=2.0
        Scaling factor for the threshold. The threshold is set to
        ``threshold_scale * 2 * p * np.sqrt(np.log(n))``, where ``n`` is the sample size
        and ``p`` is the number of variables. If None, the threshold is tuned on the
        data input to ``fit``.
    level : float, default=0.01
        If ``threshold_scale`` is None, the threshold is set to the
        (1-`level`)-quantile of the changepoint scores of all the seeded intervals on
        the training data. For this to be correct, the training data must contain no
        changepoints.
    min_segment_length : int, default=5
        Minimum length between two changepoints. Must be greater than or equal to 1.
    max_interval_length : int, default=200
        The maximum length of an interval to estimate a changepoint in. Must be greater
        than or equal to ``2 * min_segment_length``.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size according to
        ``interval_len=max(interval_len + 1, np.floor(growth_factor * interval_len))``,
        starting at ``interval_len=min_interval_length``. It also governs the amount
        of overlap between intervals of the same length, as the start of each interval
        is shifted by a factor of ``1 + 1 / growth_factor``. Must be a float in (1, 2].

    References
    ----------
    .. [1] Kovács, S., Bühlmann, P., Li, H., & Munk, A. (2023). Seeded binary
    segmentation: a general methodology for fast and optimal changepoint detection.
    Biometrika, 110(1), 249-256.

    Examples
    --------
    >>> from skchange.change_detectors import SeededBinarySegmentation
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(
            n_segments=4, mean=10, segment_length=100, p=5
        )
    >>> detector = SeededBinarySegmentation()
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
        threshold_scale=2.0,
        level: float = 1e-8,
        min_segment_length: int = 5,
        max_interval_length: int = 200,
        growth_factor: float = 1.5,
    ):
        self.change_score = change_score
        self.threshold_scale = threshold_scale  # Just holds the input value.
        self.level = level
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor

        super().__init__()

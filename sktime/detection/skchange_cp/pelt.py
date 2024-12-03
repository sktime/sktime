"""The pruned exact linear time (PELT) algorithm."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record("skchange.change_detectors.pelt")
class PELT(BaseDetector):
    """Pruned exact linear time changepoint detection, from skchange.

    Redirects to ``skchange.change_detectors.pelt``.

    An efficient implementation of the PELT algorithm [1]_ for changepoint detection.

    Parameters
    ----------
    cost : BaseCost, optional (default=``L2Cost``)
        The cost function to use for the changepoint detection.
    penalty_scale : float, optional (default=2.0)
        Scaling factor for the penalty. The penalty is set to
        ``penalty_scale * 2 * p * np.log(n)``, where ``n`` is the sample size
        and ``p`` is the number of variables. If None, the penalty is tuned on the data
        input to ``fit``.
    min_segment_length : int, optional (default=2)
        Minimum length of a segment.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American Statistical
    Association, 107(500), 1590-1598.

    Examples
    --------
    >>> from skchange.change_detectors import PELT
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=2, mean=10, segment_length=100, p=5)
    >>> detector = PELT()
    >>> detector.fit_predict(df)
    0    100
    Name: changepoint, dtype: int64
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Tveten", "johannvk"],
        "maintainers": ["Tveten", "johannvk"],
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
        cost=None,
        penalty_scale=2.0,
        min_segment_length=2,
    ):
        self.cost = cost
        self.penalty_scale = penalty_scale
        self.min_segment_length = min_segment_length

        super().__init__()

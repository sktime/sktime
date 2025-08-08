"""Dummy anomaly detector which detects no anomalies."""

from sktime.detection.base import BaseDetector


class ZeroAnomalies(BaseDetector):
    """Dummy anomaly detector which detects no anomalies ever.

    Naive method that can serve as benchmarking pipeline or API test.

    Detects no anomalies.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import ZeroAnomalies
    >>> X = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> d = ZeroAnomalies()
    >>> Xt = d.fit_transform(X)
    """

    _tags = {
        "authors": ["fkiraly"],
        "capability:multivariate": True,
        "capability:missing_values": True,
        "fit_is_empty": True,
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
    }

    def __init__(self):
        super().__init__()

    def _predict(self, X, y=None):
        """Create labels on test/deployment data.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is "segmentation", the values are ``pd.Interval`` objects.
        """
        return BaseDetector._empty_sparse()

"""Windowed Isolation Forest anomaly detector."""

__author__ = ["rupeshca007"]

import datetime
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from sktime.detection.base import BaseDetector


class SubIsolationForest(BaseDetector):
    """Windowed Isolation Forest anomaly detector for time series.

    Applies scikit-learn's ``IsolationForest`` on non-overlapping windows of
    a time series, allowing the model to adapt to local temporal structure.
    This is analogous to ``SubLOF``, but uses the Isolation Forest algorithm,
    which is especially effective for high-dimensional and large-scale data.

    Parameters
    ----------
    window_size : int, float, or datetime.timedelta
        Size of the non-overlapping windows. If the time series has a
        ``DatetimeIndex``, ``window_size`` may be a ``timedelta``; otherwise
        it must be an integer number of observations.
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : int, float, or "auto", default="auto"
        The number of samples to draw to train each base estimator.

        - If ``int``, then draw ``max_samples`` samples.
        - If ``float``, then draw ``max(round(n_samples * max_samples), 1)``
          samples.
        - If ``"auto"``, then ``max_samples=min(256, n_samples)``.

    contamination : float or "auto", default="auto"
        The proportion of outliers in the dataset.

        - If ``"auto"``, the threshold is determined as in the original paper.
        - If a float, it must be in the range ``(0, 0.5]``.

    max_features : int or float, default=1.0
        The number of features to draw to train each base estimator.

        - If ``int``, then draw ``max_features`` features.
        - If ``float``, then draw ``max(1, int(max_features * n_features_in_))``
          features.

    bootstrap : bool, default=False
        If ``True``, individual trees are fit on random subsets of the training
        data sampled with replacement. If ``False``, sampling without
        replacement is performed.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature and
        split values for each branching step and each tree in the forest.
    n_jobs : int, default=None
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    verbose : int, default=0
        Controls the verbosity of the tree building process.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.isof import SubIsolationForest
    >>> model = SubIsolationForest(window_size=5, n_estimators=10, random_state=42)
    >>> x = pd.DataFrame([0, 0.5, 100, 0.1, 0, 0, 0, 100, 0, 0, 0.3, -1, 0, 100, 0.2])
    >>> model.fit_transform(x)
       labels
    0       0
    1       0
    2       1
    3       0
    4       0
    5       0
    6       0
    7       1
    8       0
    9       0
    10      0
    11      0
    12      0
    13      1
    14      0

    See Also
    --------
    SubLOF : Windowed Local Outlier Factor detector.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["rupeshca007"],
        "maintainers": ["rupeshca007"],
        # estimator type
        # --------------
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": True,
        "fit_is_empty": False,
        # CI and test flags
        # -----------------
        "tests:core": True,
    }

    def __init__(
        self,
        window_size,
        *,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.models_ = None
        super().__init__()

    def _fit(self, X, y=None):
        """Fit an IsolationForest model per window.

        Parameters
        ----------
        X : pd.DataFrame
            Training time series data to fit model to.
        y : ignored

        Returns
        -------
        self : reference to self.
        """
        model_params = {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }

        intervals = self._split_into_intervals(X.index, self.window_size)
        self.models_ = {
            interval: IsolationForest(**model_params) for interval in intervals
        }

        for interval, model in self.models_.items():
            mask = (X.index >= interval.left) & (X.index < interval.right)
            X_win = X.loc[mask]
            if len(X_win) > 0:
                model.fit(X_win)

        return self

    @staticmethod
    def _split_into_intervals(x, interval_size):
        """Split the index range of ``x`` into equally sized intervals.

        Parameters
        ----------
        x : pd.Index
            The time series index to partition.
        interval_size : int, float, or datetime.timedelta
            Width of each interval.

        Returns
        -------
        pd.IntervalIndex
            Non-overlapping, left-closed intervals covering the full index range.
        """
        from sktime.utils.validation.series import is_integer_index

        x_max = x.max()
        x_min = x.min()
        x_span = x_max - x_min

        if isinstance(interval_size, int) and not is_integer_index(x):
            interval_size = x.freq * interval_size
        n_intervals = math.floor(x_span / interval_size) + 1

        if x_max >= x_min + (n_intervals - 1) * interval_size:
            n_intervals += 1

        breaks = [x_min + interval_size * i for i in range(n_intervals)]
        interval_range = pd.IntervalIndex.from_breaks(breaks, closed="left")
        return interval_range

    def _predict(self, X):
        """Detect anomalies on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to evaluate.

        Returns
        -------
        y_pred : pd.Series
            Sparse anomaly indicator.  Values are integer ``iloc`` indices of
            anomalous observations in ``X``.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X = X.copy()
        X["__id"] = pd.RangeIndex(len(X))

        y_all = []
        for interval, model in self.models_.items():
            X_subset = X.loc[
                (X.index >= interval.left) & (X.index < interval.right)
            ]

            if len(X_subset) == 0:
                continue

            # IsolationForest returns -1 for anomalies, +1 for normal
            y_subset = model.predict(X_subset.iloc[:, :-1])  # exclude __id col
            anomaly_indexes = (
                np.where(y_subset == -1)[0] + X_subset["__id"].iloc[0]
            )
            y_all.append(pd.Series(anomaly_indexes))

        if len(y_all) == 0:
            return self._empty_sparse()

        y_pred = pd.concat(y_all, ignore_index=True).reset_index(drop=True)
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.  If no special parameters are
            defined for a value, will return the ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each ``dict`` is passed as ``**params`` to the constructor.
        """
        params0 = {
            "window_size": datetime.timedelta(days=25),
            "n_estimators": 10,
            "random_state": 42,
        }
        params1 = {
            "window_size": 3,
            "n_estimators": 5,
            "contamination": 0.1,
            "bootstrap": True,
            "random_state": 0,
        }
        return [params0, params1]

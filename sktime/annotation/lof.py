"""Windowed local outlier factor."""

__author__ = ["Alex-JG3"]

import datetime
import math

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from sktime.annotation.base import BaseSeriesAnnotator


class SubLOF(BaseSeriesAnnotator):
    """Timeseries version of local outlier factor.

    The LOF models are fit to windows timeseries data.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

    window_size : int, float, timedelta
        Size of the non-overlapping windows on which the LOF models are fit.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf is size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    p : float, default=2
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the scores of the samples.

        - if 'auto', the threshold is determined as in the
          original paper,
        - if a float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    novelty : bool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set; and note that the
        results obtained this way may differ from the standard LOF results.

        .. versionadded:: 0.20

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.annotation.lof import SubLOF
    >>> model = SubLOF(3, window_size=5)
    >>> x = pd.DataFrame([0, 0.5, 100, 0.1, 0, 0, 0, 100, 0, 0, 0.3, -1, 0, 100, 0.2])
    >>> model.fit_transform(x)
    0     0
    1     0
    2     1
    3     0
    4     0
    5     0
    6     0
    7     1
    8     0
    9     0
    10    0
    11    0
    12    0
    13    1
    14    0
    dtype: int64
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "univariate-only": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        n_neighbors,
        window_size,
        *,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination="auto",
        novelty=False,
        n_jobs=None,
    ):
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.models = None
        super().__init__()

    def _fit(self, X, Y=None):
        """Fit the LOF model to ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to (time series).
        """
        model_params = {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "metric": self.metric,
            "p": self.p,
            "metric_params": self.metric_params,
            "contamination": self.contamination,
            "novelty": self.novelty,
            "n_jobs": self.n_jobs,
        }
        if isinstance(X, pd.Series):
            X = X.to_frame()

        intervals = self._split_into_intervals(X.index, self.window_size)
        self.models = {
            interval: LocalOutlierFactor(**model_params) for interval in intervals
        }

        for interval, model in self.models.items():
            mask = (X.index >= interval.left) & (X.index < interval.right)
            model.fit(X.loc[mask])

    @staticmethod
    def _split_into_intervals(x, interval_size):
        """Split the range of ``x`` into equally sized intervals."""
        x_max = x.max()
        x_min = x.min()
        n_intervals = math.floor((x_max - x_min) / interval_size) + 1

        if x_max >= x_min + (n_intervals - 1) * interval_size:
            n_intervals += 1

        breaks = [x_min + interval_size * i for i in range(n_intervals)]
        interval_range = pd.IntervalIndex.from_breaks(breaks, closed="left")
        return interval_range

    def _predict(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to annotate (time series).

        Returns
        -------
        Y : pd.Series or an IntervalSeries
            Change points in sequence X.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()

        y_all = []
        for interval, model in self.models.items():
            X_subset = X.loc[(X.index >= interval.left) & (X.index < interval.right)]

            if len(X_subset) == 0:
                continue

            y_subset = model.predict(X_subset)
            anomaly_indexes = X_subset.index[y_subset == -1]
            y_all.append(pd.Series(anomaly_indexes))

        if len(y_all) == 0:
            return pd.Series()

        return pd.concat(y_all, ignore_index=True).reset_index(drop=True)

    def fit_predict(self, X, Y=None):
        """Fit to data, then predict it.

        Fits model to X and Y with given annotation parameters
        and returns the annotations made by the model.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Data to be transformed
        Y : pd.Series or np.ndarray, optional (default=None)
            Unused, defaults to none.

        Returns
        -------
        self : pd.Series
            Series containing the locations of the anomalies in X.
        """
        model_params = {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "metric": self.metric,
            "p": self.p,
            "metric_params": self.metric_params,
            "contamination": self.contamination,
            "novelty": self.novelty,
            "n_jobs": self.n_jobs,
        }
        if isinstance(X, pd.Series):
            X = X.to_frame()

        intervals = self._split_into_intervals(X.index, self.window_size)
        self.models = {
            interval: LocalOutlierFactor(**model_params) for interval in intervals
        }

        y_all = []
        for interval, model in self.models.items():
            X_subset = X.loc[(X.index >= interval.left) & (X.index < interval.right)]

            if len(X_subset) == 0:
                continue

            y_subset = model.fit_predict(X_subset)
            anomaly_indexes = X_subset.index[y_subset == -1].tolist()
            y_all += anomaly_indexes

        if len(y_all) == 0:
            return pd.Series()

        return pd.Series(y_all)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params = {
            "n_neighbors": 5,
            "window_size": datetime.timedelta(days=25),
            "novelty": True,
        }
        return params

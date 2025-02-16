"""Cluster tuning estimator for time series clustering.

Purpose:
    Automatically finds the optimal number of clusters for a given clustering algorithm
    by evaluating different cluster sizes using a specified metric.

This class follows the sktime parameter estimator template.

Mandatory implements:
    fitting                     - _fit(self, X)
    fitted parameter inspection - _get_fitted_params()

Optional implements:
    updating                              - _update(self, X)
    data conversion and capabilities tags - _tags

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()

"""

from sktime.base import BaseObject
from sktime.param_est.base import BaseParamFitter
from sktime.distances import pairwise_distance
import numpy as np
import logging

log = logging.getLogger()
console = logging.StreamHandler()
log.addHandler(console)


class BaseClusterMetric(BaseObject):
    """Base class for cluster evaluation metrics."""

    def evaluate(self, X, labels):
        """
        Evaluate the quality of clustering performed on data X for given labels.

        Parameters
        ----------
        X : panel-like object
            The input time series data.
        labels : array-like of int
            Cluster labels for each time series in X.

        Returns
        -------
        score : float
            The computed metric score.
        """
        raise NotImplementedError()


class TimeSeriesSilhouetteScore(BaseClusterMetric):
    """
    Silhouette score for time series clustering.

    This implementation computes the silhouette score by first calculating the
    pairwise distance matrix using a specified time series distance metric.
    """

    def __init__(self, metric="euclidean", **metric_params):
        """
        Initialize the silhouette score metric.

        Parameters
        ----------
        metric : str, default="euclidean"
            The distance metric to use.
        **metric_params : dict
            Additional keyword arguments to pass to the distance function.
        """
        self.metric = metric
        self.metric_params = metric_params.copy()

    def evaluate(self, X, labels):
        """
        Compute the silhouette score for time series clustering.

        Parameters
        ----------
        X : panel-like object
            The input time series data.
            It must be in the format expected by pairwise_distance.
        labels : array-like of int
            Cluster labels for each time series in X.

        Returns
        -------
        score : float
            The mean silhouette score over all time series.
        """

        distance_matrix = pairwise_distance(X, metric=self.metric, **self.metric_params)
        n = len(labels)
        unique_labels = np.unique(labels)

        # If there's only one cluster, the silhouette score is not defined.
        if len(unique_labels) < 2:
            return 0.0

        silhouette_values = np.zeros(n)

        for i in range(n):
            same_cluster = np.where(labels == labels[i])[0]
            same_cluster = same_cluster[same_cluster != i]  # Exclude self
            a = (
                np.mean(distance_matrix[i, same_cluster])
                if same_cluster.size > 0
                else 0.0
            )

            b = np.inf
            for label in unique_labels:
                if label == labels[i]:
                    continue
                other_cluster = np.where(labels == label)[0]
                if other_cluster.size > 0:
                    b = min(b, np.mean(distance_matrix[i, other_cluster]))

            silhouette_values[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

        return np.mean(silhouette_values)


class ClusterSupportDetection(BaseParamFitter):
    """Custom parameter fitter for clustering optimization.

    Automatically determines the optimal number of clusters using a specified metric.

    Parameters
    ----------
    estimator : clustering model instance
        The clustering model to tune.
    param_range : list or range
        The range of cluster numbers to evaluate.
    metric : function or object with an evaluate method, default=None
        Function to evaluate clustering quality. If None, uses the elbow method.
    metric_params : dict, optional
        Additional parameters for the metric function.
    sample_size : int, optional
        Number of samples to use if dataset is large.
    verbose : int, default=0
        Verbosity level (0 for silent, 1 for progress updates).
    random_state : int or None, default=None
        Random seed for reproducibility. If provided and the estimator supports
        a `random_state` parameter, it will be set accordingly.
    direction : str, default="max"
        Optimization direction. Use "max" if a higher metric score is better
        or "min" if a lower metric score is better.
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "scitype:X": "Panel",
        "capability:missing_values": False,
        "capability:multivariate": False,
        "authors": ["TomatoChocolate12"],
    }

    def __init__(
        self,
        estimator,
        param_range,
        metric=None,
        metric_params=None,
        sample_size=None,
        verbose=0,
        random_state=None,
        direction="max",
    ):
        self.estimator = estimator
        self.param_range = param_range
        self.metric = metric
        # Ensure metric_params is not modified after initialization.
        self.metric_params = metric_params.copy() if metric_params is not None else {}
        self.sample_size = sample_size
        self.verbose = verbose
        self.random_state = random_state
        self.direction = direction

        super().__init__()

    def _fit(self, X, y=None):
        """Return the optimal number of clusters by evaluating different values."""
        best_score = -np.inf
        best_param = None
        scores = {}
        inertia_values = []

        for param in self.param_range:
            if self.verbose:
                log.info(f"Evaluating {param} clusters...")

            params = {"n_clusters": param}
            # add the random state only if the estimator supports it
            if self.random_state is not None:
                if "random_state" in self.estimator.get_params():
                    params["random_state"] = self.random_state

            estimator = self.estimator.set_params(n_clusters=param)
            labels = estimator.fit_predict(X)

            if self.metric:
                # If the metric has an 'evaluate' method, use it.
                if hasattr(self.metric, "evaluate"):
                    score = self.metric.evaluate(X, labels)
                else:
                    score = self.metric(X, labels, **self.metric_params)
                scores[param] = score
            else:
                inertia = estimator.inertia_
                inertia_values.append(inertia)
                scores[param] = -inertia  # Minimizing inertia

        if not self.metric:
            # Find the elbow point
            best_param = self._find_elbow_point(self.param_range, inertia_values)
            best_score = -inertia_values[best_param - self.param_range[0]]
        else:
            if self.direction == "max":
                best_param = max(scores, key=scores.get)
            elif self.direction == "min":
                best_param = min(scores, key=scores.get)
            else:
                raise ValueError(
                    f"Invalid direction: {self.direction} should be either 'max' or 'min'."
                )
            best_score = scores[best_param]

        self.best_param_ = best_param
        self.best_score_ = best_score
        self.scores_ = scores

        if self.verbose:
            print(f"Best parameter: {best_param} with score: {best_score}")

        return self

    def _find_elbow_point(self, param_range, inertia_values):
        """Return the elbow point by detecting when the slope drops below a threshold."""
        diffs = np.diff(inertia_values)
        threshold = self.metric_params.get("elbow_threshold", 0.1)
        for i, diff in enumerate(diffs):
            if abs(diff) < threshold:
                return param_range[i]
        return param_range[-1]

    def _get_fitted_params(self):
        """Return the best parameter found."""
        return {"best_n_clusters": self.best_param_}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""

        from sktime.clustering.k_means import TimeSeriesKMeans
        from sktime.clustering.k_medoids import TimeSeriesKMedoids
        from sktime.clustering.k_shapes import TimeSeriesKShapes

        params = {
            "estimator": TimeSeriesKMeans(),
            "param_range": range(2, 10),
            "metric": TimeSeriesSilhouetteScore(metric="dtw"),
            "metric_params": {},
            "random_state": 42,
            "direction": "max",
        }

        params2 = {
            "estimator": TimeSeriesKMedoids(),
            "param_range": range(2, 10),
            "metric": None,
            "metric_params": {"elbow_threshold": 0.05},
        }

        params3 = {
            "estimator": TimeSeriesKShapes(),
            "param_range": range(2, 10),
            "metric": None,
            "metric_params": {},
            "random_state": 1,
            "direction": "max",
        }
        return [params, params2, params3]

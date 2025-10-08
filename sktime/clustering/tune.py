"""Cluster tuning estimator for time series clustering.

Purpose:
    Automatically finds the optimal number of clusters for a given clustering algorithm
    by evaluating different cluster sizes using a specified metric.

This class follows the sktime parameter estimator template.

Mandatory implements:
    fitting                     - _fit_predict(self, X)
    fitted parameter inspection - _get_fitted_params()

Optional implements:
    updating                              - _update(self, X)
    data conversion and capabilities tags - _tags

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""

import logging

import numpy as np

from sktime.param_est.base import BaseParamFitter
from sktime.performance_metrics._clustering_metrics import TimeSeriesSilhouetteScore

log = logging.getLogger()
console = logging.StreamHandler()
log.addHandler(console)


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
        sample_size=None,
        verbose=0,
        random_state=None,
        direction="max",
    ):
        self.estimator = estimator
        self.param_range = param_range
        self.metric = metric
        self.sample_size = sample_size
        self.verbose = verbose
        self.random_state = random_state
        self.direction = direction
        self.best_n_clusters_ = None
        self.best_score_ = None
        self.scores_ = None

        if hasattr(estimator, "predict"):
            self._tags["capability:predict"] = True
        if hasattr(estimator, "predict_proba"):
            self._tags["capability:predict_proba"] = True

        super().__init__()

    def _fit(self, X, y=None):
        """Return the optimal number of clusters by evaluating different values."""
        best_score = -np.inf
        best_param = None
        scores = {}
        inertia_values = []

        for param in self.param_range:
            if self.verbose:
                log.info("Evaluating %d clusters...", param)

            params = {"n_clusters": param}
            if self.random_state is not None:
                if "random_state" in self.estimator.get_params():
                    params["random_state"] = self.random_state

            estimator = self.estimator.set_params(n_clusters=param)
            labels = estimator.fit_predict(X)

            if self.metric:
                if hasattr(self.metric, "evaluate"):
                    score = self.metric.evaluate(X, labels)
                else:
                    score = self.metric(X, labels)
                scores[param] = score
            else:
                inertia = estimator.inertia_
                inertia_values.append(inertia)
                scores[param] = -inertia

        if not self.metric:
            best_param = self._find_elbow_point(self.param_range, inertia_values)
            best_score = -inertia_values[best_param - self.param_range[0]]
        else:
            if self.direction == "max":
                best_param = max(scores, key=scores.get)
            elif self.direction == "min":
                best_param = min(scores, key=scores.get)
            else:
                raise ValueError(
                    f"Invalid argument: {self.direction} must be either 'max' or 'min'."
                )
            best_score = scores[best_param]

        self.best_n_clusters_ = best_param
        self.best_score_ = best_score
        self.scores_ = scores

        if self.verbose:
            print("Best parameter: %d with score: %f", best_param, best_score)

        return self

    def _find_elbow_point(self, range_, inertia_values):
        """Return the elbow point by detecting the point of maximum curvature."""
        inertia_values = np.array(inertia_values)
        x = np.array(range_)
        dy_dx = np.diff(inertia_values) / np.diff(x)
        d2y_dx2 = np.diff(dy_dx) / np.diff(x[:-1])
        curvature = np.abs(d2y_dx2) / (1 + dy_dx[:-1] ** 2) ** (3 / 2)
        max_curvature_idx = np.argmax(curvature)
        return range_[max_curvature_idx + 1]

    def predict(self, X):
        """Predict clusters using the best found parameters."""
        if not hasattr(self.estimator, "predict"):
            raise NotImplementedError("Inner estimator does not support predict.")
        self.estimator.set_params(n_clusters=self.best_n_clusters_)
        return self.estimator.fit_predict(X)

    def predict_proba(self, X):
        """Predict cluster probabilities if supported."""
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError("Inner estimator does not support predict_proba.")
        self.estimator.set_params(n_clusters=self.best_n_clusters_)
        return self.estimator.fit(X).predict_proba(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.clustering.kvisibility import TimeSeriesKvisibility

        params = {
            "estimator": TimeSeriesKvisibility(),
            "param_range": range(2, 10),
            "metric": None,
            "verbose": 0,
            "random_state": 42,
        }

        params2 = {
            "estimator": TimeSeriesKvisibility(),
            "param_range": range(5, 10),
            "metric": TimeSeriesSilhouetteScore(),
            "verbose": 1,
        }

        params3 = {
            "estimator": TimeSeriesKvisibility(),
            "param_range": range(2, 7),
            "metric": TimeSeriesSilhouetteScore(),
            "verbose": 1,
            "random_state": 1,
        }

        return [params, params2, params3]

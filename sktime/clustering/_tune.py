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

from sktime.param_est.base import BaseParamFitter
from sklearn.metrics import silhouette_score
import numpy as np

class ClusterSupportDetection(BaseParamFitter):
    """Custom parameter fitter for clustering optimization.
    
    Automatically determines the optimal number of clusters using a specified metric.

    Hyper-parameters
    ----------------
    estimator : clustering model instance
        The clustering model to tune, must have a `fit_predict` method.
    param_range : list or range
        The range of cluster numbers to evaluate.
    metric : function, default=None
        Function to evaluate clustering quality. If None, uses the elbow method.
    metric_params : dict, optional
        Additional parameters for the metric function.
    sample_size : int, optional
        Number of samples to use if dataset is large.
    verbose : int, default=0
        Verbosity level (0 for silent, 1 for progress updates).
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
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
            verbose=0
    ):
        self.estimator = estimator
        self.param_range = param_range
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else {}
        self.sample_size = sample_size
        self.verbose = verbose
        
        super().__init__()

    def _fit(self, X, y=None):
        """Finds the optimal number of clusters by evaluating different values."""
        best_score = -np.inf
        best_param = None
        scores = {}
        inertia_values = []
        
        for param in self.param_range:
            if self.verbose:
                print(f"Evaluating {param} clusters...")
            
            estimator = self.estimator.set_params(n_clusters=param)
            labels = estimator.fit_predict(X)
            
            if self.metric:
                score = self.metric(X, labels, **self.metric_params)
                scores[param] = score
            else:
                inertia = estimator.inertia_
                inertia_values.append(inertia)
                scores[param] = -inertia  # Minimizing inertia
        
        if not self.metric:
            # Find the elbow point
            best_param = self._find_elbow_point(inertia_values)
            best_score = -inertia_values[best_param - self.param_range[0]]
        else:
            best_param = max(scores, key=scores.get)
            best_score = scores[best_param]
        
        self.best_param_ = best_param
        self.best_score_ = best_score
        self.scores_ = scores
        
        if self.verbose:
            print(f"Best parameter: {best_param} with score: {best_score}")
        
        return self
    
    def _find_elbow_point(self, param_range, inertia_values):
        """Finds the elbow point by detecting when the slope drops below a threshold."""
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
            "metric": silhouette_score,
            "metric_params": None
        }

        params2 = {
            "estimator": TimeSeriesKMedoids(),
            "param_range": range(2, 10),
            "metric": silhouette_score,
            "metric_params": {"elbow_threshold": 0.05},
        }

        params3 = {
            "estimator": TimeSeriesKShapes(),
            "param_range": range(2, 10),
            "metric": None,
            "metric_params": None,
        }
        return [params, params2, params3]

import numpy as np

from sktime.transformers.tree import ShapeletTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils import check_array


class ShapeletForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_leaf=2,
                 n_shapelets=10,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 bootstrap=True,
                 n_jobs=None,
                 dim_to_use = 0,
                 seed=0):
        """A shapelet forest classifier
        """
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.seed = seed
        self.dim_to_use = dim_to_use

    def predict(self, X, check_input=True):
        X = np.array([np.asarray(x) for x in X.iloc[:, self.dim_to_use]])
        return self.classes_[np.argmax(
            self.predict_proba(X, check_input=check_input), axis=1)]

    def predict_proba(self, X, check_input=True):
        
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

        if self.n_dims_ > 1 and X.ndim != 3:
            raise ValueError("illegal input dimensions X.ndim != 3")

        if X.shape[-1] != self.n_timestep_:
            raise ValueError("illegal input shape ({} != {})".format(
                X.shape[-1], self.n_timestep_))

        if X.ndim > 2 and X.shape[1] != self.n_dims_:
            raise ValueError("illegal input shape ({} != {}".format(
                X.shape[1], self.n_dims))

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        X = X.reshape(X.shape[0], self.n_dims_ * self.n_timestep_)
        return self.bagging_classifier_.predict_proba(X)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """
        X = np.array([np.asarray(x) for x in X.iloc[:, self.dim_to_use]])
        seed = check_random_state(self.seed)
        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        shapelet_tree_classifier = ShapeletTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_shapelets=self.n_shapelets,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=seed,
        )
#        print(shapelet_tree_classifier)

        if n_dims > 1:
            shapelet_tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=shapelet_tree_classifier,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.seed,
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)
        self.bagging_classifier_.fit(X, y, sample_weight=sample_weight)

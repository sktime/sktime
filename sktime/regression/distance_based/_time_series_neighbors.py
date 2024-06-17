"""KNN time series regression.

This class is a KNN regressor which supports time series distance measures. The class
has hardcoded string references to numba based distances in sktime.distances. It can
also be used with callables, or sktime (pairwise transformer) estimators.

This is a direct wrap or sklearn KNeighbors, with added functionality that allows time
series distances to be passed, and the sktime time series regressor interface.
"""

__author__ = ["fkiraly"]
__all__ = ["KNeighborsTimeSeriesRegressor"]

from sklearn.neighbors import KNeighborsRegressor

from sktime.base._panel.knn import _BaseKnnTimeSeriesEstimator
from sktime.distances import pairwise_distance
from sktime.dists_kernels.base.adapters._sklearn import _SklearnDistMixin
from sktime.regression.base import BaseRegressor

# add new distance string codes here
DISTANCES_SUPPORTED = [
    "euclidean",
    # Euclidean will default to the base class distance
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "lcss",
    "edr",
    "erp",
    "msm",
]


class KNeighborsTimeSeriesRegressor(
    _SklearnDistMixin, _BaseKnnTimeSeriesEstimator, BaseRegressor
):
    """KNN Time Series Regressor.

    An adapted version of the scikit-learn KNeighborsRegressor for time series data.

    This class is a KNN regressor which supports time series distance measures.
    It has hardcoded string references to numba based distances in sktime.distances,
    and can also be used with callables, or sktime (pairwise transformer) estimators.

    Parameters
    ----------
    n_neighbors : int, set k for knn (default =1)
    weights : string or callable function, optional. default = 'uniform'
        mechanism for weighting a vot
        one of: 'uniform', 'distance', or a callable function
    algorithm : str, optional. default = 'brute'
        search method for neighbours
        one of {'auto', 'ball_tree', 'brute', 'brute_incr'}

        * 'brute' precomputes the distance matrix and applies
          ``sklearn`` ``KNeighborsRegressor`` directly.
          This algorithm is not memory efficient as it scales with the size
          of the distance matrix, but may be more runtime efficient.
        * 'brute_incr' passes the distance to ``sklearn`` ``KNeighborsRegressor``,
          with ``algorithm='brute'``. This is useful for large datasets,
          for memory efficiency, as the distance is used incrementally,
          without precomputation. However, this may be less runtime efficient.
        * 'ball_tree' uses a ball tree to find the nearest neighbors,
          using ``KNeighborsRegressor`` from ``sklearn``.
          May be more runtime and memory efficient on mid-to-large datasets,
          however, the distance computation may be slower.

    distance : str or callable, optional. default ='dtw'
        distance measure between time series
        if str, must be one of the following strings:
            'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw',
            'lcss', 'edr', 'erp', 'msm'
        this will substitute a hard-coded distance metric from sktime.distances
        When mpdist is used, the subsequence length (parameter m) must be set
            Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
                                metric='mpdist', metric_params={'m':30})
        if callable, must be of signature (X: Panel, X2: Panel) -> np.ndarray
            output must be mxn array if X is Panel of m Series, X2 of n Series
            if distance_mtype is not set, must be able to take
                X, X2 which are pd_multiindex and numpy3D mtype
        can be pairwise panel transformer inheriting from BasePairwiseTransformerPanel
    distance_params : dict, optional. default = None.
        dictionary for metric parameters , in case that distane is a str
    distance_mtype : str, or list of str optional. default = None.
        mtype that distance expects for X and X2, if a callable
            only set this if distance is not BasePairwiseTransformerPanel descendant
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree. This can affect the
        speed of the construction and query, as well as the memory required to store
        the tree. The optimal value depends on the nature of the problem.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> regressor = KNeighborsTimeSeriesRegressor()
    >>> regressor.fit(X_train, y_train)
    KNeighborsTimeSeriesRegressor(...)
    >>> y_pred = regressor.predict(X_test)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_mtype": ["pd-multiindex", "numpy3D"],
    }

    def __init__(
        self,
        n_neighbors=1,
        weights="uniform",
        algorithm="brute",
        distance="dtw",
        distance_params=None,
        distance_mtype=None,
        leaf_size=30,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.distance = distance
        self.distance_params = distance_params
        self.distance_mtype = distance_mtype
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        # translate distance strings into distance callables
        if isinstance(distance, str) and distance not in DISTANCES_SUPPORTED:
            raise ValueError(
                f"Unrecognised distance measure string: {distance}. "
                f"Allowed values for string codes are: {DISTANCES_SUPPORTED}. "
                "Alternatively, pass a callable distance measure into the constructor."
            )

        self.knn_estimator_ = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric="precomputed",
            metric_params=distance_params,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )
        self.weights = weights

        super().__init__()

        # the distances in sktime.distances want numpy3D
        #   otherwise all Panel formats are ok
        if isinstance(self.distance, str):
            self.set_tags(X_inner_mtype="numpy3D")
            self.set_tags(**{"capability:unequal_length": False})
            self.set_tags(**{"capability:missing_values": False})
        elif distance_mtype is not None:
            self.set_tags(X_inner_mtype=distance_mtype)

        from sktime.dists_kernels import BasePairwiseTransformerPanel

        # inherit capability tags from distance, if it is an estimator
        if isinstance(distance, BasePairwiseTransformerPanel):
            inherit_tags = [
                "capability:missing_values",
                "capability:unequal_length",
                "capability:multivariate",
            ]
            self.clone_tags(distance, inherit_tags)

    def _distance(self, X, X2):
        """Compute distance - unified interface to str code and callable."""
        distance = self.distance
        distance_params = self.distance_params
        if distance_params is None:
            distance_params = {}

        if isinstance(distance, str):
            return pairwise_distance(X, X2, distance, **distance_params)
        else:
            return distance(X, X2)

    def _fit_dist(self, X, y):
        """Fit the model using adapted distance metric."""
        # sklearn wants distance callabel element-wise,
        # numpy1D x numpy1D -> float
        # sktime distance classes are Panel x Panel -> numpy2D
        # and the numba distances are numpy3D x numpy3D -> numpy2D
        # so we need to wrap the sktime distances
        if isinstance(self.distance, str):
            # numba distances
            metric = self._one_element_distance_npdist
        else:
            # sktime distance classes
            metric = self._one_element_distance_sktime_dist

        algorithm = self.algorithm
        if algorithm == "brute_incr":
            algorithm = "brute"

        self.knn_estimator_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            algorithm=algorithm,
            metric=metric,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )

        X = self._convert_X_to_sklearn(X)
        self.knn_estimator_.fit(X, y)
        return self

    def _fit_precomp(self, X, y):
        """Fit the model using precomputed distance matrix."""
        # store full data as indexed X
        self._X = X

        dist_mat = self._distance(X, X)

        self.knn_estimator_.fit(dist_mat, y)

        return self

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        param1 = {
            "n_neighbors": 1,
            "weights": "uniform",
            "algorithm": "auto",
            "distance": "euclidean",
            "distance_params": None,
            "n_jobs": None,
        }
        param2 = {
            "n_neighbors": 3,
            "weights": "distance",
            "algorithm": "ball_tree",
            "distance": "dtw",
            "distance_params": {"window": 0.5},
            "n_jobs": -1,
        }
        return [param1, param2]

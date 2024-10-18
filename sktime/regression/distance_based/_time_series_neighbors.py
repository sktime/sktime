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


class KNeighborsTimeSeriesRegressor(_BaseKnnTimeSeriesEstimator, BaseRegressor):
    """K-nearest neighbours Time Series Regressor.

    An adapted version of the ``scikit-learn`` ``KNeighborsRegressor``,
    adapted for time series data.

    Time series distances are passed as the ``distance argument``, which can be:

    * a string. This will substitute a hard-coded distance metric
      from ``sktime.distances``. These default distances are intended to be
      performant, but cannot deal with unequal length or multivariate series.
    * a ``sktime`` pairwise transformer.
      These are available in ``sktime.dists_kernels``, and can be discovered
      via ``registry.all_estimators`` by searching for
      ``pairwise-transformer`` type.and are composable
      first class citizens in the ``sktime`` framework.
      Distances dealing with unequal length or multivariate series are available,
      these can be discovered via ``capability:unequal_length`` and
      ``capability:multivariate`` tags.
    * a callable. The exact signature for callables is described below.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.

    weights : str or callable, optional (default: 'uniform')
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this
          case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of
          distances, and returns an array of the same
          shape containing the weights.

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


    distance : str, sktime pairwise transformer, or callable, optional. default ='dtw'
        distance measure between time series

        * if str, must be one of the following strings:
          'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw',
          'lcss', 'edr', 'erp', 'msm', 'twe'
          this will substitute a hard-coded distance metric from ``sktime.distances``
        * if ``sktime`` pairwise transformer,
          must implement the ``pairwise-transformer`` interface.
          ``sktime`` transformers are available in ``sktime.dists_kernels``,
          and discoverable via ``registry.all_estimators`` by searching for
          ``pairwise-transformer`` type.
        * if non-class callable, parameters can be passed via distance_params
          Example: knn_dtw = KNeighborsTimeSeriesClassifier(
          distance='dtw', distance_params={'epsilon':0.1})
        * if any callable, must be of signature ``(X: Panel, X2: Panel) -> np.ndarray``.
          The output must be mxn array if X is Panel of m Series, X2 of n Series;
          if ``distance_mtype`` is not set, must be able to take
          ``X``, ``X2`` which are of ``pd_multiindex`` and ``numpy3D`` mtype

    distance_params : dict, optional. default = None.
        dictionary for metric parameters , in case that distance is a str
    distance_mtype : str, or list of str optional. default = None.
        mtype that distance expects for X and X2, if a callable
            only set this if distance is not BasePairwiseTransformerPanel descendant
    pass_train_distances : bool, optional, default = False.
        Whether distances between training points are computed and passed to sklearn.
        Passing is superfluous for algorithm='brute', but may have impact otherwise.

    leaf_size : int, optional, default=30
        Leaf size passed to ``BallTree`` or ``KDTree``.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    n_jobs : int, optional,  default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        Does not affect the ``fit`` method.

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
        pass_train_distances=False,
        leaf_size=30,
        n_jobs=None,
    ):
        self._knn_cls = KNeighborsRegressor

        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            distance=distance,
            distance_params=distance_params,
            distance_mtype=distance_mtype,
            pass_train_distances=pass_train_distances,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )

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
        params0 = {}
        params1 = {
            "n_neighbors": 1,
            "weights": "uniform",
            "algorithm": "auto",
            "distance": "euclidean",
            "distance_params": None,
            "n_jobs": None,
        }
        params2 = {
            "n_neighbors": 3,
            "weights": "distance",
            "algorithm": "ball_tree",
            "distance": "dtw",
            "distance_params": {"window": 0.5},
            "n_jobs": -1,
        }

        # testing that callables/classes can be passed
        from sktime.dists_kernels.compose_tab_to_panel import AggrDist

        dist = AggrDist.create_test_instance()
        params3 = {"distance": dist}

        params4 = {
            "algorithm": "brute_incr",
            "distance": "dtw",
            "distance_params": {"epsilon": 0.1},
        }
        params5 = {"algorithm": "ball_tree", "distance": dist}

        params = [params0, params1, params2, params3, params4, params5]
        return params

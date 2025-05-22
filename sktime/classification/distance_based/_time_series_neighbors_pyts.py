"""K-nearest neighbors time series classifier, from pyts."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["KNeighborsTimeSeriesClassifierPyts"]

from sktime.base.adapters._pyts import _PytsAdapter
from sktime.classification.base import BaseClassifier


class KNeighborsTimeSeriesClassifierPyts(_PytsAdapter, BaseClassifier):
    """K-nearest neighbors time series classifier, from ``pyts``.

    Direct interface to ``pyts.classification.KNeighborsClassifier``,
    author of the interfaced class is ``johannfaouzi``.

    Parameters
    ----------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use.

    weights : str or callable, optional (default = 'uniform')
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors. Ignored ff ``metric``
        is either 'dtw', 'dtw_sakoechiba', 'dtw_itakura', 'dtw_multiscale',
        'dtw_fast' or 'boss' ('brute' will be used).

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or DistanceMetric object (default = 'minkowski')
        The distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class from
        scikit-learn for a list of available metrics.
        For Dynamic Time Warping, the available metrics are 'dtw',
        'dtw_sakoechiba', 'dtw_itakura', 'dtw_multiscale' and 'dtw_fast'.
        For BOSS metric, one can use 'boss'.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``n_jobs=-1``, then the number of jobs is set to the number of CPU
        cores. Doesn't affect :meth:`fit` method.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    Examples
    --------
    >>> import sktime.classification.distance_based as clf_db  # doctest: +SKIP
    >>> from clf_db import KNeighborsTimeSeriesClassifierPyts  # doctest: +SKIP
    >>> from sktime.datasets import load_unit_test  # doctest: +SKIP
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test")  # doctest: +SKIP
    >>> clf = KNeighborsTimeSeriesClassifierPyts(n_neighbors=1,
    ...     weights="uniform",
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     p=2,
    ...     metric="minkowski",
    ...     metric_params=None,
    ...     n_jobs=1,
    ... )  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    KNeighborsTimeSeriesClassifierPyts(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["johannfaouzi", "fkiraly"],  # johannfaouzi is author of upstream
        "python_dependencies": "pyts",
        # estimator type
        # --------------
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "classifier_type": "distance",
    }

    # defines the name of the attribute containing the pyts estimator
    _estimator_attr = "_pyts_rocket"

    def _get_pyts_class(self):
        """Get pyts class.

        should import and return pyts class
        """
        from pyts.classification import KNeighborsClassifier

        return KNeighborsClassifier

    def __init__(
        self,
        n_neighbors=1,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=1,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        super().__init__()

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "n_neighbors": 3,
            "weights": "distance",
            "metric": "dtw_fast",
        }
        return [params1, params2]

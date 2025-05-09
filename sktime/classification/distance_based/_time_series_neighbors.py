"""KNN time series classification.

This class is a KNN classifier which supports time series distance measures. The class
has hardcoded string references to numba based distances in sktime.distances. It can
also be used with callables, or sktime (pairwise transformer) estimators.

This is a direct wrap or sklearn KNeighbors, with added functionality that allows time
series distances to be passed, and the sktime time series classifier interface.

todo: add a utility method to set keyword args for distance measure parameters. (e.g.
handle the parameter name(s) that are passed as metric_params automatically, depending
on what distance measure is used in the classifier (e.g. know that it is w for dtw, c
for msm, etc.). Also allow long-format specification for non-standard/user-defined
measures e.g. set_distance_params(measure_type=None, param_values_to_set=None,
param_names=None)
"""

__author__ = ["fkiraly", "jasonlines", "TonyBagnall", "chrisholder"]
__all__ = ["KNeighborsTimeSeriesClassifier"]

from sklearn.neighbors import KNeighborsClassifier

from sktime.base._panel.knn import _BaseKnnTimeSeriesEstimator
from sktime.classification.base import BaseClassifier


class KNeighborsTimeSeriesClassifier(_BaseKnnTimeSeriesEstimator, BaseClassifier):
    """K-nearest neighbours Time Series Classifier.

    An adapted version of the ``scikit-learn`` ``KNeighborsClassifier``,
    adapted for time series data.

    This class is a KNN classifier which supports time series distance measures.

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
    n_neighbors : int, optional, default = 1
        k in "k nearest neighbours"

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.

        Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    algorithm : str, optional. default = 'brute'
        search method for neighbours
        one of {'ball_tree', 'brute', 'brute_incr'}

        * 'brute' precomputes the distance matrix and applies
          ``sklearn`` ``KNeighborsClassifier`` directly.
          This algorithm is not memory efficient as it scales with the size
          of the distance matrix, but may be more runtime efficient.
        * 'brute_incr' passes the distance to ``sklearn`` ``KNeighborsClassifier``,
          with ``algorithm='brute'``. This is useful for large datasets,
          for memory efficiency, as the distance is used incrementally,
          without precomputation. However, this may be less runtime efficient.
        * 'ball_tree' uses a ball tree to find the nearest neighbors,
          using ``KNeighborsClassifier`` from ``sklearn``.
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

    distance_params : dict, optional, default = None.
        dictionary for distance parameters, in case that distance is a str or callable
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
    >>> from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> classifier = KNeighborsTimeSeriesClassifier(distance="euclidean")
    >>> classifier.fit(X_train, y_train)
    KNeighborsTimeSeriesClassifier(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["jasonlines", "TonyBagnall", "chrisholder", "fkiraly"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "X_inner_mtype": ["pd-multiindex", "numpy3D"],
        "classifier_type": "distance",
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
        self._knn_cls = KNeighborsClassifier

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

    def _predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : sktime-compatible Panel data, of mtype X_inner_mtype, with n_samples series
            data to predict class labels for

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        if self.algorithm == "brute":
            return self._predict_proba_precomp(X)
        else:
            return self._predict_proba_dist(X)

    def _predict_proba_dist(self, X):
        """Predict (proba) using adapted distance metric."""
        X = self._dist_adapt._convert_X_to_sklearn(X)
        y_pred = self.knn_estimator_.predict_proba(X)
        return y_pred

    def _predict_proba_precomp(self, X):
        """Predict (proba) using precomputed distance matrix."""
        # self._X should be the stored _X
        dist_mat = self._dist_adapt._distance(X, self._X)
        y_pred = self.knn_estimator_.predict_proba(dist_mat)
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        # non-default distance and algorithm
        params0 = {"distance": "euclidean"}

        # testing distance_params
        params1 = {"distance": "dtw", "distance_params": {"epsilon": 0.1}}

        # testing that callables/classes can be passed
        from sktime.dists_kernels.compose_tab_to_panel import AggrDist

        dist = AggrDist.create_test_instance()
        params2 = {"distance": dist}

        params3 = {"algorithm": "ball_tree"}
        # params5 = {"algorithm": "kd_tree", "distance": "euclidean"}
        params4 = {
            "algorithm": "brute_incr",
            "distance": "dtw",
            "distance_params": {"epsilon": 0.1},
        }
        params5 = {"algorithm": "ball_tree", "distance": dist}

        params = [params0, params1, params2, params3, params4, params5]
        return params

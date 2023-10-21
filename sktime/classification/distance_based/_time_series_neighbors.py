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

__author__ = ["jasonlines", "TonyBagnall", "chrisholder", "fkiraly"]
__all__ = ["KNeighborsTimeSeriesClassifier"]

from inspect import signature

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sktime.classification.base import BaseClassifier
from sktime.datatypes import check_is_mtype
from sktime.distances import pairwise_distance

# add new distance string codes here
DISTANCES_SUPPORTED = [
    "euclidean",
    # Euclidean will default to the base class distance
    "squared",
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "lcss",
    "edr",
    "erp",
    "msm",
    "twe",
]


class KNeighborsTimeSeriesClassifier(BaseClassifier):
    """KNN Time Series Classifier.

    An adapted version of the scikit-learn KNeighborsClassifier for time series data.

    This class is a KNN classifier which supports time series distance measures.
    It has hardcoded string references to numba based distances in sktime.distances,
    and can also be used with callables, or sktime (pairwise transformer) estimators.

    Parameters
    ----------
    n_neighbors : int, set k for knn (default =1)
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:
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
        one of {'auto', 'ball_tree', 'kd_tree', 'brute'}
    distance : str or callable, optional. default ='dtw'
        distance measure between time series
        if str, must be one of the following strings:
            'euclidean', 'squared', 'dtw', 'ddtw', 'wdtw', 'wddtw',
            'lcss', 'edr', 'erp', 'msm', 'twe'
        this will substitute a hard-coded distance metric from sktime.distances
        If non-class callable, parameters can be passed via distance_params
            Example: knn_dtw = KNeighborsTimeSeriesClassifier(
                                    distance='dtw', distance_params={'epsilon':0.1})
        if any callable, must be of signature (X: Panel, X2: Panel) -> np.ndarray
            output must be mxn array if X is Panel of m Series, X2 of n Series
            if distance_mtype is not set, must be able to take
                X, X2 which are pd_multiindex and numpy3D mtype
        can be pairwise panel transformer inheriting from BasePairwiseTransformerPanel
    distance_params : dict, optional. default = None.
        dictionary for distance parameters, in case that distance is a str or callable
    distance_mtype : str, or list of str optional. default = None.
        mtype that distance expects for X and X2, if a callable
            only set this if distance is not BasePairwiseTransformerPanel descendant
    pass_train_distances : bool, optional, default = False.
        Whether distances between training points are computed and passed to sklearn.
        Passing is superfluous for algorithm='brute', but may have impact otherwise.
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

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
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.distance = distance
        self.distance_params = distance_params
        self.distance_mtype = distance_mtype
        self.pass_train_distances = pass_train_distances
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

        super().__init__()

        # input check for supported distance strings
        if isinstance(distance, str) and distance not in DISTANCES_SUPPORTED:
            raise ValueError(
                f"Unrecognised distance measure string: {distance}. "
                f"Allowed values for string codes are: {DISTANCES_SUPPORTED}. "
                "Alternatively, pass a callable distance measure into the constructor."
            )

        self.knn_estimator_ = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric="precomputed",
            metric_params=distance_params,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
            weights=weights,
        )

        # the distances in sktime.distances want numpy3D
        #   otherwise all Panel formats are ok
        if isinstance(distance, str):
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

    def _distance(self, X, X2=None):
        """Compute distance - unified interface to str code and callable."""
        distance = self.distance
        distance_params = self.distance_params
        if distance_params is None:
            distance_params = {}

        if isinstance(distance, str):
            return pairwise_distance(X, X2, distance, **distance_params)
        else:
            if X2 is not None:
                return distance(X, X2, **distance_params)
            # if X2 is None, check if distance allows None X2 to mean "X2=X"
            else:
                sig = signature(distance).parameters
                X2_sig = sig[list(sig.keys())[1]]
                if X2_sig.default is not None:
                    return distance(X, X2, **distance_params)
                else:
                    return distance(X, **distance_params)

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : sktime compatible Panel data container, of mtype X_inner_mtype,
            with n time series to fit the estimator to
        y : {array-like, sparse matrix}
            Target values of shape = [n]
        """
        # store full data as indexed X
        self._X = X

        if self.pass_train_distances:
            dist_mat = self._distance(X)
        else:
            # if we do not want/need to pass train-train distances,
            #   we still need to pass a zeros matrix, this means "do not consider"
            # citing the sklearn KNeighborsClassifier docs on distance matrix input:
            # "X may be a sparse graph, in which case only "nonzero" elements
            #   may be considered neighbors."
            X_inner_mtype = self.get_tag("X_inner_mtype")
            _, _, X_meta = check_is_mtype(X, X_inner_mtype, return_metadata=True)
            n = X_meta["n_instances"]
            dist_mat = np.zeros([n, n], dtype="float")

        self.knn_estimator_.fit(dist_mat, y)

        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : sktime-compatible data format, Panel or Series, with n_samples series
        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        # self._X should be the stored _X
        dist_mat = self._distance(X, self._X)

        result = self.knn_estimator_.kneighbors(
            dist_mat, n_neighbors=n_neighbors, return_distance=return_distance
        )

        # result is either dist, or (dist, ind) pair, depending on return_distance
        return result

    def _predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : sktime-compatible Panel data, of mtype X_inner_mtype, with n_samples series
            data to predict class labels for

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        # self._X should be the stored _X
        dist_mat = self._distance(X, self._X)

        y_pred = self.knn_estimator_.predict(dist_mat)

        return y_pred

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
        # self._X should be the stored _X
        dist_mat = self._distance(X, self._X)

        y_pred = self.knn_estimator_.predict_proba(dist_mat)

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        # non-default distance and algorithm
        params1 = {"distance": "euclidean"}

        # testing distance_params
        params2 = {"distance": "dtw", "distance_params": {"epsilon": 0.1}}

        # testing that callables/classes can be passed
        from sktime.dists_kernels.compose_tab_to_panel import AggrDist

        dist = AggrDist.create_test_instance()
        params3 = {"distance": dist}

        return [params1, params2, params3]

# -*- coding: utf-8 -*-
"""KNN time series regression.

This class is a KNN regressor which supports time series distance measures.
The class has hardcoded string references to numba based distances in sktime.distances.
It can also be used with callables, or sktime (pairwise transformer) estimators.

This is a direct wrap or sklearn KNeighbors, with added functionality that allows
time series distances to be passed, and the sktime time series regressor interface.
"""

__author__ = ["fkiraly"]
__all__ = ["KNeighborsTimeSeriesRegressor"]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _check_weights

from sktime.distances import pairwise_distance
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


class KNeighborsTimeSeriesRegressor(BaseRegressor):
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
        one of {'auto’, 'ball_tree', 'kd_tree', 'brute'}
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
        "capability:multivariate": True,
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
                "Alternatively, pass a callable distance measure into the constuctor."
            )

        self.knn_estimator_ = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric="precomputed",
            metric_params=distance_params,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )
        self.weights = _check_weights(weights)

        super(KNeighborsTimeSeriesRegressor, self).__init__()

        # the distances in sktime.distances want numpy3D
        #   otherwise all Panel formats are ok
        if isinstance(self.distance, str):
            self.set_tags(X_inner_mtype="numpy3D")
        elif distance_mtype is not None:
            self.set_tags(X_inner_mtype=distance_mtype)

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

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : sktime-compatible Panel data format, with n_samples series
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]
        """
        # store full data as indexed X
        self._X = X

        dist_mat = self._distance(X, X)

        self.knn_estimator_.fit(dist_mat, y)

        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : sktime-compatible data format, Panel or Series, with n_samples series
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]
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
        # self._X should be the stored _X
        dist_mat = self._distance(X, self._X)

        neigh_ind = self.knn_estimator_.kneighbors(
            dist_mat, n_neighbors=n_neighbors, return_distance=return_distance
        )

        return neigh_ind

    def _predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : sktime-compatible Panel data format, with n_samples series

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        # self._X should be the stored _X
        dist_mat = self._distance(X, self._X)

        y_pred = self.knn_estimator_.predict(dist_mat)

        return y_pred

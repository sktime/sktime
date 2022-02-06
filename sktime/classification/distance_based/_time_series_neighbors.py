# -*- coding: utf-8 -*-
"""KNN time series classification.

This class is a KNN classifier which supports time series distance measures.
The class has hardcoded string references to numba based distances in sktime.distances.
It can also be used with callables, or sktime (pairwise transformer) estimators.

This is a direct wrap or sklearn KNeighbors, with added functionality that allows
time series distances to be passed, and the sktime time series classifier interface.

todo: add a utility method to set keyword args for distance measure parameters.
(e.g.  handle the parameter name(s) that are passed as metric_params automatically,
depending on what distance measure is used in the classifier (e.g. know that it is w
for dtw, c for msm, etc.). Also allow long-format specification for
non-standard/user-defined measures e.g. set_distance_params(measure_type=None,
param_values_to_set=None,
param_names=None)
"""

__author__ = ["jasonlines", "TonyBagnall", "chrisholder", "fkiraly"]
__all__ = ["KNeighborsTimeSeriesClassifier"]

import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import _check_weights

from sktime.classification.base import BaseClassifier
from sktime.distances import (
    ddtw_distance,
    dtw_distance,
    erp_distance,
    euclidean_distance,
    lcss_distance,
    wddtw_distance,
    wdtw_distance,
)
from sktime.distances.mpdist import mpdist

# add new distance string codes here
DISTANCE_DICT = {
    "euclidean": euclidean_distance,
    # Euclidean will default to the base class distance
    "dtw": dtw_distance,
    "ddtw": ddtw_distance,
    "wdtw": wdtw_distance,
    "wddtw": wddtw_distance,
    "lcss": lcss_distance,
    "erp": erp_distance,
    "mpdist": mpdist,
}


class KNeighborsTimeSeriesClassifier(BaseClassifier):
    """KNN Time Series Classifier.

    An adapted version of the scikit-learn KNeighborsClassifier for time series data.

    This class is a KNN classifier which supports time series distance measures.
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
        if str, one of {'euclidean', 'dtw', 'dtwcv', 'ddtw', 'wdtw', 'wddtw', 'lcss',
                'erp', 'mpdist'}
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

    Examples
    --------
    >>> from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> classifier = KNeighborsTimeSeriesClassifier()
    >>> classifier.fit(X_train, y_train)
    KNeighborsTimeSeriesClassifier(...)
    >>> y_pred = classifier.predict(X_test)
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
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.distance = distance
        self.distance_params = distance_params
        self.distance_mtype = distance_mtype

        self._cv_for_params = False

        # the distances in sktime.distances want numpy3D
        #   otherwise all Panel formats are ok
        if isinstance(distance, str):
            self.set_tags(X_inner_mtype="numpy3D")
        elif distance_mtype is not None:
            self.set_tags(X_inner_mtype=distance_mtype)

        # translate distance strings into distance callables
        if distance in DISTANCE_DICT.keys():
            distance = DISTANCE_DICT[distance]
        elif distance == "dtwcv":  # special case to force loocv grid search
            # cv in training
            if distance_params is not None:
                warnings.warn(
                    "Warning: measure parameters have been specified for "
                    "dtwcv. "
                    "These will be ignored and parameter values will be "
                    "found using LOOCV."
                )
            distance = dtw_distance
            self._cv_for_params = True
            self._param_matrix = {
                "distance_params": [{"w": x / 100} for x in range(0, 100)]
            }
        elif isinstance(distance, str):
            allowed_vals = list(DISTANCE_DICT.keys()) + ["dtwcv"]
            raise ValueError(
                f"Unrecognised distance measure string: {distance}. "
                f"Allowed values for string codes are: {allowed_vals}. "
                "Alternatively, pass a callable distance measure into the constuctor."
            )

        self._distance = distance

        self.knn_estimator_ = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric="precomputed",
            metric_params=distance_params,
            **kwargs,
        )
        self.weights = _check_weights(weights)

        super(KNeighborsTimeSeriesClassifier, self).__init__()

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

    def _predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : sktime-compatible Panel data format, with n_samples series

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

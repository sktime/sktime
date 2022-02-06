# -*- coding: utf-8 -*-
"""KNN time series classification.

 Built on sklearn KNeighborsClassifier, this class supports a range of distance
 measure specifically for time series. These distance functions are defined in cython
 in sktime.distances.elastic_cython. Python versions are in sktime.distances.elastic
 but these are orders of magnitude slower.

Please note that many aspects of this class are taken from scikit-learn's
KNeighborsTimeSeriesClassifier class with necessary changes to enable use with time
series classification data and distance measures.

todo: add a utility method to set keyword args for distance measure parameters.
(e.g.  handle the parameter name(s) that are passed as metric_params automatically,
depending on what distance measure is used in the classifier (e.g. know that it is w
for dtw, c for msm, etc.). Also allow long-format specification for
non-standard/user-defined measures e.g. set_distance_params(measure_type=None,
param_values_to_set=None,
param_names=None)
"""

__author__ = ["jasonlines", "TonyBagnall"]
__all__ = ["KNeighborsTimeSeriesClassifier"]

import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import _check_weights

from sktime.classification.base import BaseClassifier
from sktime.distances.elastic import euclidean_distance
from sktime.distances import (
    ddtw_distance,
    dtw_distance,
    erp_distance,
    lcss_distance,
    wddtw_distance,
    wdtw_distance,
)
from sktime.distances.mpdist import mpdist


class KNeighborsTimeSeriesClassifier(BaseClassifier):
    """KNN Time Series Classifier.

    An adapted version of the scikit-learn KNeighborsClassifier to work with
    time series data.

    Necessary changes required for time series data:
        -   calls to X.shape in kneighbors, predict and predict_proba.
            In the base class, these methods contain:
                n_samples, _ = X.shape
            This however assumes that data must be 2d (a set of multivariate
            time series is 3d). Therefore these methods
            needed to be overridden to change this call to the following to
            support 3d data:
                n_samples = X.shape[0]
        -   check array has been disabled. This method allows nd data via an
        argument in the method header. However, there
            seems to be no way to set this in the classifier and allow it to
            propagate down to the method. Therefore, this
            method has been temporarily disabled (and then re-enabled). It
            is unclear how to fix this issue without either
            writing a new classifier from scratch or changing the
            scikit-learn implementation. TO-DO: find permanent
            resolution to this issue (raise as an issue on sklearn GitHub?)


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
        if str, one of {'dtw','ddtw', 'wdtw','lcss','erp','msm','twe'}: default ='dtw'
            this will substitute a hard-coded distance metric from sktime.distances
        if callable, must be of signature (X: Panel, X2: Panel) -> np.ndarray
            output must be mxn array if X is Panel of m Series, X2 of n Series
        can be pairwise panel transformer inheriting from BasePairwiseTransformerPanel
    distance_params : dict, optional. default = None.
        dictionary for metric parameters , in case that distane is a str

    Examples
    --------
    >>> from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> from sktime.datasets import load_basic_motions
    >>> X_train, y_train = load_basic_motions(return_X_y=True, split="train")
    >>> X_test, y_test = load_basic_motions(return_X_y=True, split="test")
    >>> classifier = KNeighborsTimeSeriesClassifier()
    >>> classifier.fit(X_train, y_train)
    KNeighborsTimeSeriesClassifier(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_neighbors=1,
        weights="uniform",
        distance="dtw",
        distance_params=None,
        **kwargs
    ):
        self._cv_for_params = False
        self.distance = distance
        self.distance_params = distance_params

        if distance == "euclidean":  # Euclidean will default to the base class distance
            distance = euclidean_distance
        elif distance == "dtw":
            distance = dtw_distance
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
        elif distance == "ddtw":
            distance = ddtw_distance
        elif distance == "wdtw":
            distance = wdtw_distance
        elif distance == "wddtw":
            distance = wddtw_distance
        elif distance == "lcss":
            distance = lcss_distance
        elif distance == "erp":
            distance = erp_distance
        elif distance == "mpdist":
            distance = mpdist
            # When mpdist is used, the subsequence length (parameter m) must be set
            # Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
            # metric='mpdist', metric_params={'m':30})
        else:
            if type(distance) is str:
                raise ValueError(
                    "Unrecognised distance measure: " + distance + ". Allowed values "
                    "are names from [euclidean,dtw,ddtw,wdtw,wddtw,lcss,erp,msm] or "
                    "please pass a callable distance measure into the constuctor"
                )

        self.knn_estimator_ = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm="brute",
            metric="precomputed",
            metric_params=distance_params,
            **kwargs
        )
        self.weights = _check_weights(weights)

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : sktime-compatible data format, Panel or Series, with n_samples series
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]
        """
        # store full data as indexed X
        self._X = X

        dist_mat = self.distance(X)

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
        dist_mat = self.distance(X, self._X)

        neigh_ind = self.knn_estimator_.kneighbors(
            dist_mat, n_neighbors=n_neighbors, return_distance=return_distance
        )

        return neigh_ind

    def _predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : sktime-compatible data format, Panel or Series, with n_samples series
        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        # self._X should be the stored _X
        dist_mat = self.distance(X, self._X)

        y_pred = self.knn_estimator_.predict(dist_mat)

        return y_pred

    def _predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : sktime-compatible data format, Panel or Series, with n_samples series

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # self._X should be the stored _X
        dist_mat = self.distance(X, self._X)

        y_pred = self.knn_estimator_.predict_proba(dist_mat)

        return y_pred

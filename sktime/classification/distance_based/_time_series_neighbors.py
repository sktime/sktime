# -*- coding: utf-8 -*-
""" KNN time series classification built on sklearn KNeighborsClassifier

"""

__author__ = "Jason Lines"
__all__ = ["KNeighborsTimeSeriesClassifier"]

import warnings
from distutils.version import LooseVersion
from functools import partial

import numpy as np
from joblib import Parallel
from joblib import delayed
from joblib import effective_n_jobs
from scipy import stats
from scipy.sparse import issparse
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import pairwise_distances_chunked
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
from sklearn.neighbors._base import _check_weights
from sklearn.neighbors._base import _get_weights
from sklearn.utils import gen_even_slices
from sklearn.utils._joblib import __version__ as joblib_version
from sklearn.utils.extmath import weighted_mode
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sktime.distances.elastic_cython import ddtw_distance
from sktime.distances.elastic_cython import dtw_distance
from sktime.distances.elastic_cython import erp_distance
from sktime.distances.elastic_cython import lcss_distance
from sktime.distances.elastic_cython import msm_distance
from sktime.distances.elastic_cython import twe_distance
from sktime.distances.elastic_cython import wddtw_distance
from sktime.distances.elastic_cython import wdtw_distance

from sktime.classification.base import BaseClassifier
from sktime.distances.mpdist import mpdist
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y

"""
Please note that many aspects of this class are taken from scikit-learn's
KNeighborsTimeSeriesClassifier
class with necessary changes to enable use with time series classification
data and distance measures.

TO-DO: add a utility method to set keyword args for distance measure
parameters (e.g. handle the parameter
name(s) that are passed as metric_params automatically, depending on what
distance measure is used in the
classifier (e.g. know that it is w for dtw, c for msm, etc.). Also allow
long-format specification for
non-standard/user-defined measures
e.g. set_distance_params(measure_type=None, param_values_to_set=None,
param_names=None)
"""


class KNeighborsTimeSeriesClassifier(_KNeighborsClassifier, BaseClassifier):
    """
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
    n_neighbors     : int, set k for knn (default =1)
    weights         : mechanism for weighting a vote: 'uniform', 'distance'
    or a callable function: default ==' uniform'
    algorithm       : search method for neighbours {‘auto’, ‘ball_tree’,
    ‘kd_tree’, ‘brute’}: default = 'brute'
    metric          : distance measure for time series: {'dtw','ddtw',
    'wdtw','lcss','erp','msm','twe'}: default ='dtw'
    metric_params   : dictionary for metric parameters: default = None

    """

    def __init__(
        self,
        n_neighbors=1,
        weights="uniform",
        algorithm="brute",
        metric="dtw",
        metric_params=None,
        **kwargs
    ):

        self._cv_for_params = False

        if metric == "dtw":
            metric = dtw_distance
        elif metric == "dtwcv":  # special case to force loocv grid search
            # cv in training
            if metric_params is not None:
                warnings.warn(
                    "Warning: measure parameters have been specified for "
                    "dtwcv. "
                    "These will be ignored and parameter values will be "
                    "found using LOOCV."
                )
            metric = dtw_distance
            self._cv_for_params = True
            self._param_matrix = {
                "metric_params": [{"w": x / 100} for x in range(0, 100)]
            }
        elif metric == "ddtw":
            metric = ddtw_distance
        elif metric == "wdtw":
            metric = wdtw_distance
        elif metric == "wddtw":
            metric = wddtw_distance
        elif metric == "lcss":
            metric = lcss_distance
        elif metric == "erp":
            metric = erp_distance
        elif metric == "msm":
            metric = msm_distance
        elif metric == "twe":
            metric = twe_distance
        elif metric == "mpdist":
            metric = mpdist
        # When mpdist is used, the subsequence length (parameter m) must be set
        # Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
        # metric='mpdist', metric_params={'m':30})
        else:
            if type(metric) is str:
                raise ValueError(
                    "Unrecognised distance measure: " + metric + ". Allowed "
                    "values are "
                    "names from "
                    "[dtw,ddtw,"
                    "wdtw,"
                    "wddtw,"
                    "lcss,erp,"
                    "msm] or "
                    "please "
                    "pass a "
                    "callable "
                    "distance "
                    "measure "
                    "into the "
                    "constuctor "
                    "directly."
                )

        super(KNeighborsTimeSeriesClassifier, self).__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            metric_params=metric_params,
            **kwargs
        )
        self.weights = _check_weights(weights)

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : sktime-format pandas dataframe with shape([n_cases,n_dimensions]),
        or numpy ndarray with shape([n_cases,n_readings,n_dimensions])

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]

        """
        X, y = check_X_y(X, y, enforce_univariate=False, coerce_to_numpy=True)
        y = np.asarray(y)
        check_classification_targets(y)

        # print(X)
        # if internal cv is desired, the relevant flag forces a grid search
        # to evaluate the possible values,
        # find the best, and then set this classifier's params to match
        if self._cv_for_params:
            grid = GridSearchCV(
                estimator=KNeighborsTimeSeriesClassifier(
                    metric=self.metric, n_neighbors=1, algorithm="brute"
                ),
                param_grid=self._param_matrix,
                cv=LeaveOneOut(),
                scoring="accuracy",
            )
            grid.fit(X, y)
            self.metric_params = grid.best_params_["metric_params"]

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            if y.ndim != 1:
                warnings.warn(
                    "A column-vector y was passed when a 1d array "
                    "was expected. Please change the shape of y to "
                    "(n_samples, ), for example using ravel().",
                    DataConversionWarning,
                    stacklevel=2,
                )

            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        self.classes_ = []
        self._y = np.empty(y.shape, dtype=np.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        if hasattr(check_array, "__wrapped__"):
            temp = check_array.__wrapped__.__code__
            check_array.__wrapped__.__code__ = _check_array_ts.__code__
        else:
            temp = check_array.__code__
            check_array.__code__ = _check_array_ts.__code__

        fx = self._fit(X)

        if hasattr(check_array, "__wrapped__"):
            check_array.__wrapped__.__code__ = temp
        else:
            check_array.__code__ = temp

        self._is_fitted = True
        return fx

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : sktime-format pandas dataframe with shape([n_cases,n_dimensions]),
        or numpy ndarray with shape([n_cases,n_readings,n_dimensions])

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
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" % type(n_neighbors)
                )

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse="csr", allow_nd=True)
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1

        train_size = self._fit_X.shape[0]
        if n_neighbors > train_size:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" % (train_size, n_neighbors)
            )
        n_samples = X.shape[0]
        sample_range = np.arange(n_samples)[:, None]

        n_jobs = effective_n_jobs(self.n_jobs)
        if self._fit_method == "brute":

            reduce_func = partial(
                self._kneighbors_reduce_func,
                n_neighbors=n_neighbors,
                return_distance=return_distance,
            )

            # for efficiency, use squared euclidean distances
            kwds = (
                {"squared": True}
                if self.effective_metric_ == "euclidean"
                else self.effective_metric_params_
            )

            result = pairwise_distances_chunked(
                X,
                self._fit_X,
                reduce_func=reduce_func,
                metric=self.effective_metric_,
                n_jobs=n_jobs,
                **kwds
            )

        elif self._fit_method in ["ball_tree", "kd_tree"]:
            if issparse(X):
                raise ValueError(
                    "%s does not work with sparse matrices. Densify the data, "
                    "or set algorithm='brute'" % self._fit_method
                )
            if LooseVersion(joblib_version) < LooseVersion("0.12"):
                # Deal with change of API in joblib
                delayed_query = delayed(self._tree.query, check_pickle=False)
                parallel_kwargs = {"backend": "threading"}
            else:
                delayed_query = delayed(self._tree.query)
                parallel_kwargs = {"prefer": "threads"}
            result = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(X[s], n_neighbors, return_distance)
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
        else:
            raise ValueError("internal: _fit_method not recognized")

        if return_distance:
            dist, neigh_ind = zip(*result)
            result = np.vstack(dist), np.vstack(neigh_ind)
        else:
            result = np.vstack(result)

        if not query_is_train:
            return result
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                dist, neigh_ind = result
            else:
                neigh_ind = result

            sample_mask = neigh_ind != sample_range

            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False

            neigh_ind = np.reshape(neigh_ind[sample_mask], (n_samples, n_neighbors - 1))

            if return_distance:
                dist = np.reshape(dist[sample_mask], (n_samples, n_neighbors - 1))
                return dist, neigh_ind
            return neigh_ind

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n_query,
        n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        self.check_is_fitted()

        if hasattr(check_array, "__wrapped__"):
            temp = check_array.__wrapped__.__code__
            check_array.__wrapped__.__code__ = _check_array_ts.__code__
        else:
            temp = check_array.__code__
            check_array.__code__ = _check_array_ts.__code__

        neigh_dist, neigh_ind = self.kneighbors(X)
        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_samples = X.shape[0]
        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        if hasattr(check_array, "__wrapped__"):
            check_array.__wrapped__.__code__ = temp
        else:
            check_array.__code__ = temp
        return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n_query,
        n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        self.check_is_fitted()

        if hasattr(check_array, "__wrapped__"):
            temp = check_array.__wrapped__.__code__
            check_array.__wrapped__.__code__ = _check_array_ts.__code__
        else:
            temp = check_array.__code__
            check_array.__code__ = _check_array_ts.__code__

        X = check_array(X, accept_sparse="csr")

        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_samples = X.shape[0]

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(X.shape[0])
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_samples, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        if hasattr(check_array, "__wrapped__"):
            check_array.__wrapped__.__code__ = temp
        else:
            check_array.__code__ = temp
        return probabilities


# overwrite sklearn internal checks, this is really hacky
# we now need to replace: check_array.__wrapped__.__code__ since it's
# wrapped by a future warning decorator
def _check_array_ts(array, *args, **kwargs):
    return array

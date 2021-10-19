# -*- coding: utf-8 -*-
"""
Base class template for time series classifier scitype.

    class name: BaseClassifier

Scitype defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__all__ = [
    "BaseClassifier",
    "classifier_list",
]
__author__ = ["mloning", "fkiraly"]

import numpy as np

from sktime.base import BaseEstimator
from sktime.utils.validation.panel import check_X, check_X_y

"""
Main list of classifiers extending this class. For clarity, some utility classifiers,
such as Proximity Stump, are not listed.
"""
classifier_list = [
    # in classification/distance_based
    "ProximityForest",
    # "KNeighborsTimeSeriesClassifier",
    # "ElasticEnsemble",
    # "ShapeDTW",
    # in classification/dictionary_based
    "BOSS",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "WEASEL",
    "MUSE",
    # in classification/interval_based
    "RandomIntervalSpectralForest",
    "TimeSeriesForest",
    "CanonicalIntervalForest",
    # in classification/shapelet_based
    "ShapeletTransformClassifier",
    "ROCKET",
    "MrSEQLClassifier",
]


class BaseClassifier(BaseEstimator):
    """Base time series classifier template class.

    The base classifier specifies the methods and method
    signatures that all forecasters have to implement.

    Specific implementations of these methods is deferred to concrete
    forecasters.
    """

    _tags = {
        "coerce-X-to-numpy": True,
    }

    def __init__(self):
        self._is_fitted = False

        super(BaseClassifier, self).__init__()

    def fit(self, X, y):
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries
        y : array-like, shape =  [n_instances] - the class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)
        coerce_to_pandas = self.get_tag("coerce-X-to-pandas", False)
        X, y = check_X_y(
            X, y, coerce_to_numpy=coerce_to_numpy, coerce_to_pandas=coerce_to_pandas
        )

        self._fit(X, y)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, X):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries

        Returns
        -------
        y : array-like, shape =  [n_instances] - predicted class labels
        """
        coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)
        coerce_to_pandas = self.get_tag("coerce-X-to-pandas", False)
        X = check_X(
            X, coerce_to_numpy=coerce_to_numpy, coerce_to_pandas=coerce_to_pandas
        )
        self.check_is_fitted()

        y = self._predict(X)

        return y

    def predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries

        Returns
        -------
        y : array-like, shape =  [n_instances, n_classes] - predictive pmf
        """
        coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)
        coerce_to_pandas = self.get_tag("coerce-X-to-pandas", False)
        X = check_X(
            X, coerce_to_numpy=coerce_to_numpy, coerce_to_pandas=coerce_to_pandas
        )
        self.check_is_fitted()
        return self._predict_proba(X)

    def score(self, X, y):
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries
        y : array-like, shape =  [n_instances] - predicted class labels

        Returns
        -------
        float, accuracy score of predict(X) vs y
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), normalize=True)

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        core logic

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries
        y : array-like, shape = [n_instances] - the class labels

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X):
        """Predicts labels for sequences in X.

        core logic

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries

        Returns
        -------
        y : array-like, shape =  [n_instances] - predicted class labels
        """
        distributions = self.predict_proba(X)
        predictions = []
        for instance_index in range(0, X.shape[0]):
            distribution = distributions[instance_index]
            prediction = np.argmax(distribution)
            predictions.append(prediction)
        y = self.label_encoder.inverse_transform(predictions)

        return y

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries

        Returns
        -------
        y : array-like, shape =  [n_instances, n_classes] - predictive pmf
        """
        raise NotImplementedError("abstract method")

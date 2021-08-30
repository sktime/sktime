# -*- coding: utf-8 -*-
""" RandOm Convolutional KErnel Transform (ROCKET)
"""

__author__ = ["RavenRudi"]
__all__ = ["ROCKETRegressor"]

import numpy as np
from sklearn.linear_model import RidgeCV

from sktime.utils.validation.panel import check_X, check_X_y
from sktime.regression.base import BaseRegressor
from sktime.series_as_features.base.estimators.shapelet_based._rocket_estimator import (
    BaseROCKETEstimator,
)


class ROCKETRegressor(BaseROCKETEstimator, BaseRegressor):
    """
    Regressor wrapped for the ROCKET transformer using RidgeCV as the
    base regressor.

    Parameters
    ----------
    num_kernels             : int, number of kernels for ROCKET transform
    (default=10,000)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    classifier              : ROCKET classifier
    n_classes               : extracted from the data

    Notes
    -----
    @article{dempster_etal_2019,
      author  = {Dempster, Angus and Petitjean, Francois and Webb,
      Geoffrey I},
      title   = {ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels},
      year    = {2019},
      journal = {arXiv:1910.13051}
    }

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/ROCKETClassifier.java

    """

    @property
    def base_estimator(self):
        return RidgeCV(alphas=np.logspace(-3, 3, 10), normalize=True)


def fit(self, X, y):
    """Fit regressor to training data.

    Parameters
    ----------
    X : pd.DataFrame, optional (default=None)
        Exogeneous data
    y : pd.Series, pd.DataFrame, or np.array
        Target time series to which to fit the regressor.

    Returns
    -------
    self :
        Reference to self.
    """
    coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)

    X, y = check_X_y(X, y, coerce_to_numpy=coerce_to_numpy)

    self._fit(X, y)

    # this should happen last
    self._is_fitted = True


def predict(self, X):
    """Predict time series.

    Parameters
    ----------
    X : pd.DataFrame, shape=[n_obs, n_vars]
        A2-d dataframe of exogenous variables.

    Returns
    -------
    y_pred : pd.Series
        Regression predictions.
    """
    coerce_to_numpy = self.get_tag("coerce-X-to-numpy", False)

    X = check_X(X, coerce_to_numpy=coerce_to_numpy)
    self.check_is_fitted()

    y = self._predict(X)

    return y

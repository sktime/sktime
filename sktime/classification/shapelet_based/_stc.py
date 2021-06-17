# -*- coding: utf-8 -*-
"""Shapelet Transform Classifier.

Wrapper implementation of a shapelet transform classifier pipeline that simply
performs a (configurable) shapelet transform then builds (by default) a random
forest. This is a stripped down version for basic usage.
"""

__author__ = "Tony Bagnall"
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import class_distribution
from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.shapelets import ContractedShapeletTransform
from sktime.utils.validation.panel import check_X, check_X_y


class ShapeletTransformClassifier(BaseClassifier):
    """Shapelet Transform Classifier.

    Basic implementation along the lines of [1,2]

    Parameters
    ----------
    transform_contract_in_mins : int, search time for shapelets, optional
    (default = 300)
    n_estimators               :       500,
    random_state               :  int, seed for random, optional (default = none)

    Attributes
    ----------
    TO DO

    Notes
    -----
    ..[1] Jon Hills et al., "Classification of time series by
    shapelet transformation",
        Data Mining and Knowledge Discovery, 28(4), 851--881, 2014
    https://link.springer.com/article/10.1007/s10618-013-0322-1
    ..[2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform
    for Multiclass Time Series Classification",
    Transactions on Large-Scale Data and Knowledge Centered
      Systems, 32, 2017
    https://link.springer.com/chapter/10.1007/978-3-319-22729-0_20
    Java Version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/classifiers/shapelet_based/ShapeletTransformClassifier.java
    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self, transform_contract_in_mins=60, n_estimators=500, random_state=None
    ):
        self.transform_contract_in_mins = transform_contract_in_mins
        self.n_estimators = n_estimators
        self.random_state = random_state

        #        self.shapelet_transform=ContractedShapeletTransform(
        #        time_limit_in_mins=self.time_contract_in_mins, verbose=shouty)
        #        self.classifier=RandomForestClassifier(
        #        n_estimators=self.n_estimators,criterion="entropy")
        #        self.st_X=None;
        super(ShapeletTransformClassifier, self).__init__()

    def fit(self, X, y):
        """Perform a shapelet transform then builds a random forest.

        Contract default for ST is 5 hours
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification. RISE has no bespoke method for multivariate
            classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True)

        # if y is a pd.series then convert to array.
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # generate pipeline in fit so that random state can be propagated properly.
        self.classifier_ = Pipeline(
            [
                (
                    "st",
                    ContractedShapeletTransform(
                        time_contract_in_mins=self.transform_contract_in_mins,
                        verbose=False,
                        random_state=self.random_state,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators, random_state=self.random_state
                    ),
                ),
            ]
        )

        self.n_classes_ = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        self.classifier_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Find predictions for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        X = check_X(X, enforce_univariate=True)
        self.check_is_fitted()

        return self.classifier_.predict(X)

    def predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        X = check_X(X, enforce_univariate=True)
        self.check_is_fitted()

        return self.classifier_.predict_proba(X)

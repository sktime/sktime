""" ROCKET Classifier
wrapper implementation of the ROCKET classifier that uses

"""

__author__ = ["Angus Dempster", "Tony Bagnall"]
__all__ = ["RocketClassifier"]

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import class_distribution
from sktime.classification.base import BaseClassifier
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformers.panel.rocket import Rocket
from sklearn.pipeline import Pipeline
from sktime.utils.validation.panel import check_X, check_X_y


class RocketClassifier(BaseClassifier):
    """Rocket Classifier
        Basic implementation along the lines of

    """

    def __init__(self, random_state=None):
        self.random_state = random_state
        self._classifier = None
        self._num_classes = 0
        self.classes_ = 0

        super(RocketClassifier, self).__init__()

    def fit(self, X, y):
        """Perform a rocket transform then builds a ridge classifier.
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
        X, y = check_X_y(X, y, enforce_univariate=False)

        # if y is a pd.series then convert to array.
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # generate pipeline in fit so that random state can be propagated properly.
        self._classifier = Pipeline(
            [("rocket", Rocket(random_state=self.random_state),),
             ("ridge", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),),
            ]
        )

        self._num_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        self._classifier.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samps, num_atts] or a
        data frame.
        If a Pandas data frame is passed,

        Returns
        -------
        output : array of shape = [n_samples]
        """
        X = check_X(X, enforce_univariate=False)
        self.check_is_fitted()

        return self._classifier.predict(X)

    def predict_proba(self, X):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,

        Returns
        -------
        output : array of shape = [n_samples, num_classes] of probabilities
        """
        X = check_X(X, enforce_univariate=False)
        self.check_is_fitted()

        return self._classifier.predict_proba(X)

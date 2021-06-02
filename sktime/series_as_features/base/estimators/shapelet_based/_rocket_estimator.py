# -*- coding: utf-8 -*-
""" RandOm Convolutional KErnel Transform (ROCKET)
"""

__author__ = "Raven Rudi"
__all__ = ["BaseROCKETEstimator"]

from abc import ABC, abstractmethod
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.utils.multiclass import class_distribution

from sktime.transformations.panel.rocket import Rocket
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class BaseROCKETEstimator(ABC):
    """
    Base class for ROCKET classifier and ROCKET regressor.

    Allows the creation of an ensemble of ROCKET estimators to allow for
    generation of probabilities as the expense of scalability.

    Parameters
    ----------
    num_kernels             : int, number of kernels for ROCKET transform
    (default=10,000)
    ensemble                : boolean, create ensemble of ROCKET's (default=False)
    ensemble_size           : int, size of the ensemble (default=25)
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    classifiers             : array of IndividualTDE classifiers
    weights                 : weight of each classifier in the ensemble
    weight_sum              : sum of all weights
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

    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        num_kernels=10000,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classifier = None

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super().__init__()

    @property
    @abstractmethod
    def base_estimator(self):
        # set in ROCKET classifier and ROCKET regressor
        pass

    def fit(self, X, y):
        """
        Build a pipeline containing the ROCKET transformer and RidgeClassifierCV
        classifier.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        self.classifier = rocket_pipeline = make_pipeline(
            Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            self.base_estimator,
        )
        rocket_pipeline.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        self.check_is_fitted()
        X = check_X(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X)

        dists = np.zeros((X.shape[0], self.n_classes))
        preds = self.classifier.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists

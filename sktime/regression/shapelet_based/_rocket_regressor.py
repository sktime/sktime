# -*- coding: utf-8 -*-
""" RandOm Convolutional KErnel Transform (ROCKET)
"""

__author__ = ["RavenRudi"]
__all__ = ["ROCKETRegressor"]

import numpy as np
from sklearn.linear_model import RidgeCV


from sktime.regression.base import BaseRegressor
from sktime.series_as_features.base.estimators.shapelet_based._rocket_estimator import (
    BaseROCKETEstimator,
)


class ROCKETRegressor(BaseROCKETEstimator, BaseRegressor):
    """
    Regressor wrapped for the ROCKET transformer using RidgeCV as the
    base regressor.
    Allows the creation of an ensemble of ROCKET regressors to allow for
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
    regressors             : array of IndividualTDE regressors
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
    }

    def __init__(
        self,
        num_kernels=10000,
        ensemble=False,
        ensemble_size=25,
        random_state=None,
    ):

        super().__init__(num_kernels, ensemble, ensemble_size, random_state)
        super().__init__()

    def fit(self, X, y):
        """
        Build a single or ensemble of pipelines containing the ROCKET transformer and
        RidgeCV regressor.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        super().fit(RidgeCV(alphas=np.logspace(-3, 3, 10), normalize=True), X, y)
        return self

# -*- coding: utf-8 -*-
""" RandOm Convolutional KErnel Transform (ROCKET)
"""

__author__ = "Matthew Middlehurst"
__all__ = ["ROCKETClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.base import BaseClassifier
from sktime.series_as_features.base.estimators.shapelet_based._rocket_estimator import (
    BaseROCKETEstimator,
)


class ROCKETClassifier(BaseROCKETEstimator, BaseClassifier):
    """
    Classifier wrapped for the ROCKET transformer using RidgeClassifierCV as the
    base classifier.
    Allows the creation of an ensemble of ROCKET classifiers to allow for
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

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/ROCKETClassifier.java

    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
    }

    # Used in BaseROCKETEstimator
    @property
    def base_model(self):
        return RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)

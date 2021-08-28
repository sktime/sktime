# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (ROCKET)."""

__author__ = ["MatthewMiddlehurst", "victordremov", "RavenRudi"]
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
        return RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)

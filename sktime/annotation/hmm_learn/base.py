# -*- coding: utf-8 -*-

"""
Hidden Markov Model based annotation from hmmlearn.

This code provides a base interface template for models
from hmmlearn for using that library for annotation of time series.

Please see the original library
(https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py)
"""

import numpy as np
import pandas as pd

from sktime.annotation.base import BaseSeriesAnnotator

__author__ = ["miraep8"]
__all__ = ["BaseHMMLearn"]


class BaseHMMLearn(BaseSeriesAnnotator):
    """Base class for all HMM wrappers, handles required overlap between packages."""

    _tags = {
        "univariate-only": True,
        "fit_is_empty": True,
        "python_dependencies": "hmmlearn",
    }  # for unit test cases
    _hmm_estimator = None

    def __init__(self):
        super(BaseHMMLearn, self).__init__()

    @staticmethod
    def _fix_input(X):
        """Convert input X into the format needed.

        Parameters
        ----------
        X : arraylike (1D np.ndarray or pd.series), shape = [num_observations]
            Observations to apply labels to.

        Returns
        -------
        X : arraylike (2D np.ndarray), shape = [1, num_observations]
            Observations to apply labels to.
        series: bool - whether or not X was originally a pd.Series
        index: pd.index, the index if X was originally a series object.
        """
        series = isinstance(X, pd.Series)
        index = None
        if series:
            index = X.index
            X = (X.to_numpy()).reshape((-1, 1))
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape((-1, 1))
        return X, series, index

    def _fit(self, X, Y=None):
        """Ensure X is correct type, then fit wrapped estimator.

        Parameters
        ----------
        X : arraylike (1D np.ndarray or pd.series), shape = [num_observations]
            Observations to apply labels to.

        Returns
        -------
        self :
            Reference to self.
        """
        X, _, _ = self._fix_input(X)
        self._hmm_estimator = self._hmm_estimator.fit(X)
        return self

    def _predict(self, X):
        """Ensure the input type is correct, then predict using wrapped estimator.

        Parameters
        ----------
        X : 1D np.array, shape = [num_observations]
            Observations to apply labels to.

        Returns
        -------
        annotated_x : array-like, shape = [num_observations]
            Array of predicted class labels, same size as input.
        """
        X, series, index = self._fix_input(X)
        X_prime = self._hmm_estimator.predict(X)
        if series:
            X_prime = pd.Series(X_prime, index=index)
        return X_prime

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """Interface class which allows users to sample from their HMM."""
        return self._hmm_estimator.sample(n_samples, random_state, currstate)

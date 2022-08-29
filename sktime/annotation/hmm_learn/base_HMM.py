# -*- coding: utf-8 -*-

"""
Hidden Markov Model based annotation from hmmlearn.

This code provides a base interface template for models
from hmmlearn for using that library for annotation of time series.

Please see the original library
(https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py)
"""

import pandas as pd

from sktime.annotation.base import BaseSeriesAnnotator

__author__ = ["miraep8"]
__all__ = ["BaseHMMLearn"]


class BaseHMMLearn(BaseSeriesAnnotator):
    """Base class for all HMM wrappers, handles required overlap between packages."""

    _tags = {"univariate-only": True, "fit_is_empty": True}  # for unit test cases
    _hmm_estimator = None

    def __init__(self):
        super(BaseHMMLearn, self).__init__()

    def _fit(self, X, Y=None):
        series = isinstance(X, pd.Series)
        if series:
            X = (X.to_numpy()).reshape((-1, 1))
        self._hmm_estimator = self._hmm_estimator.fit(X)
        return self

    def _predict(self, X):
        series = isinstance(X, pd.Series)
        if series:
            index = X.index
            X = (X.to_numpy()).reshape((-1, 1))
        X_prime = self._hmm_estimator.predict(X)
        if series:
            X_prime = pd.Series(X_prime, index=index)
        return X_prime

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """Interface class which allows users to sample from their HMM."""
        return self._hmm_estimator.sample(n_samples, random_state, currstate)

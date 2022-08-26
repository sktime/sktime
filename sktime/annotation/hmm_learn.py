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
from sktime.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["miraep8"]
__all__ = ["BaseHMMLearn", "GaussianHMM"]


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


_check_soft_dependencies("hmmlearn.hmm", severity="warning")


class GaussianHMM(BaseHMMLearn):
    """Hidden Markov Model with Gaussian emissions.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.
    covars_ : array
        Covariance parameters for each state.
        The shape depends on :attr:`covariance_type`:
        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    Examples
    --------
    >>> from sktime.annotation.hmm_learn import GaussianHMM
    >>> model = GaussianHMM(algorithm='viterbi', n_components=2)
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        startprob_prior: float = 1.0,
        transmat_prior: float = 1.0,
        means_prior: float = 0,
        means_weight: float = 0,
        covars_prior: float = 1e-2,
        covars_weight: float = 1,
        algorithm: str = "viterbi",
        random_state: float = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "stmc",
        init_params: str = "stmc",
        implementation: str = "log",
    ):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.implementation = implementation
        super(GaussianHMM, self).__init__()

    def _fit(self, X, Y=None):
        # import inside _fit to avoid hard dependency.
        from hmmlearn.hmm import GaussianHMM as _GaussianHMM

        self._hmm_estimator = _GaussianHMM(
            self.n_components,
            self.covariance_type,
            self.min_covar,
            self.startprob_prior,
            self.transmat_prior,
            self.means_prior,
            self.means_weight,
            self.covars_prior,
            self.covars_weight,
            self.algorithm,
            self.random_state,
            self.n_iter,
            self.tol,
            self.verbose,
            self.params,
            self.init_params,
            self.implementation,
        )
        super(GaussianHMM, self)._fit(X, Y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        params = {
            "n_components": 3,
            "covariance_type": "diag",
            "min_covar": 1e-3,
        }

        return params

# -*- coding: utf-8 -*-

"""
Hidden Markov Model based annotation from hmmlearn.

This code provides a base interface template for models
from hmmlearn for using that library for annotation of time series.
"""

from attr import define

from sktime.annotation.base import BaseSeriesAnnotator
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("hmmlearn", severity="warning")

__author__ = ["miraep8"]
__all__ = ["BaseHMMLearn"]


class BaseHMMLearn(BaseSeriesAnnotator):
    """Base class for all HMM wrappers, handles required overlap between packages."""

    _tags = {"univariate-only": True, "fit_is_empty": True}  # for unit test cases
    _hmm_estimator = None

    def __init__(self):
        super(BaseHMMLearn, self).__init__()

    def _fit(self, X, Y=None):
        self._hmm_estimator = self._hmm_estimator.fit(X)
        return self

    def _predict(self, X):
        return self._hmm_estimator.predict(X)

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """Interface class which allows users to sample from their HMM."""
        return self._hmm_estimator.sample(n_samples, random_state, currstate)


@define
class GuassianHMM(BaseHMMLearn):

    n_components: int = 1
    covariance_type: str = "diag"
    min_covar: float = 1e-3
    startprob_prior: float = 1.0
    transmat_prior: float = 1.0
    means_prior: float = 0
    means_weight: float = 0
    covars_prior: float = 1e-2
    covars_weight: float = 1
    algorithm: str = "viterbi"
    random_state: float = None
    n_iter: int = 10
    tol: float = 1e-2
    verbose: bool = False
    params: str = "stmc"
    init_params: str = "stmc"
    implementation: str = "log"

    def __attrs_post_init__(self):

        # import inside method to avoid hard dependency
        import hmmlearn

        self._hmm_estimator = hmmlearn.hmm.GaussianHMM(
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
        super(GuassianHMM, self).__init__()

# -*- coding: utf-8 -*-

"""
Hidden Markov Model with Multinomial emissions.

Please see the original library
(https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py)
"""

from sktime.annotation.hmm_learn import BaseHMMLearn
from sktime.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["klam-data", "pyyim", "mgorlin"]
__all__ = ["MultinomialHMM"]

_check_soft_dependencies("hmmlearn.hmm", severity="warning")


class MultinomialHMM(BaseHMMLearn):
    """Hidden Markov Model with multinomial emissions.

    Parameters
    ----------
    n_components : int
        Number of states.
    n_trials : int or array of int
        Number of trials (when sampling, all samples must have the same
        :attr:`n_trials`).
    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.
    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.
    algorithm : {"viterbi", "map"}, optional
        Decoder algorithm.
    random_state: RandomState or an int seed, optional
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.
    verbose : bool, optional
        Whether per-iteration convergence reports are printed to
        :data:`sys.stderr`.  Convergence can also be diagnosed using the
        :attr:`monitor_` attribute.
    params, init_params : string, optional
        The parameters that get updated during (``params``) or initialized
        before (``init_params``) the training.  Can contain any
        combination of 's' for startprob, 't' for transmat, and 'e' for
        emissionprob.  Defaults to all parameters.
    implementation: string, optional
        Determines if the forward-backward algorithm is implemented with
        logarithms ("log"), or using scaling ("scaling").  The default is
        to use logarithms for backwards compatability.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    emissionprob_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import MultinomialHMM # doctest: +SKIP
    """

    def __init__(
        self,
        n_components: int = 1,
        n_trials: int = None,
        startprob_prio: float = 1.0,
        transmat_prior: float = 1.0,
        algorithm: str = "viterbi",
        random_state: float = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "ste",
        init_params: str = "ste",
        implementation: str = "log",
    ):

        self.n_components = n_components
        self.n_trials = n_trials
        self.startprob_prio = startprob_prio
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.implementation = implementation
        super(MultinomialHMM, self).__init__()

    def _fit(self, X, Y=None):
        # import inside _fit to avoid hard dependency.
        from hmmlearn.hmm import MultinomialHMM as _MultinomialHMM

        self._hmm_estimator = _MultinomialHMM(
            self.n_components,
            self.n_trials,
            self.startprob_prio,
            self.transmat_prior,
            self.algorithm,
            self.random_state,
            self.n_iter,
            self.tol,
            self.verbose,
            self.params,
            self.init_params,
            self.implementation,
        )
        return super(MultinomialHMM, self)._fit(X, Y)

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
            "random_state": 7,
        }

        return params

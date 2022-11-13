# -*- coding: utf-8 -*-

"""
Hidden Markov Model with categorical (discrete) emissions.

Please see the original library
(https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py)
"""

from sktime.annotation.hmm_learn import BaseHMMLearn
from sktime.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["kejsitake"]
__all__ = ["CategoricalHMM"]


_check_soft_dependencies("hmmlearn.hmm", severity="warning")


class CategoricalHMM(BaseHMMLearn):
    """
    Hidden Markov Model with categorical (discrete) emissions.

    Parameters
    ----------
    n_components : int
        Number of states.
    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.
    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.
    emissionprob_prior : array, shape (n_components, n_features), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`emissionprob_`.
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
    >>> from sktime.annotation.hmm_learn import CategoricalHMM # doctest: +SKIP
    >>> from sktime.annotation.datagen import piecewise_normal # doctest: +SKIP
    >>> data = piecewise_normal( # doctest: +SKIP
    ...    means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ...    ).reshape((-1, 1))
    >>> model = CategoricalHMM(algorithm='viterbi', n_components=2) # doctest: +SKIP
    >>> model = model.fit(data) # doctest: +SKIP
    >>> labeled_data = model.predict(data) # doctest: +SKIP
    """

    def __init__(
        self,
        n_components: int = 1,
        startprob_prior: str = 1.0,
        transmat_prior: float = 1.0,
        *,
        emissionprob_prior: float = 1.0,
        algorithm: str = "viterbi",
        random_state: float = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "ste",
        init_params: str = "ste",
        implementation: str = "log"
    ):
        self.n_components = n_components
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.emissionprob_prior = emissionprob_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.implementation = implementation
        super(CategoricalHMM, self).__init__()

    def _fit(self, X, Y=None):
        """Create a new instance of wrapped hmmlearn estimator.

        Parameters
        ----------
        X : 1D np.array, shape = [num_observations]
            Observations to apply labels to.

        Returns
        -------
        self :
            Reference to self.
        """
        # import inside _fit to avoid hard dependency.
        from hmmlearn.hmm import CategoricalHMM as _CategoricalHMM

        self._hmm_estimator = _CategoricalHMM(
            n_components=self.n_components,
            startprob_prior=self.startprob_prior,
            transmat_prior=self.transmat_prior,
            emissionprob_prior=self.emissionprob_prior,
            algorithm=self.algorithm,
            random_state=self.random_state,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=self.verbose,
            params=self.params,
            init_params=self.init_params,
            implementation=self.implementation,
        )
        return super(CategoricalHMM, self)._fit(X, Y)

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
        }

        return params

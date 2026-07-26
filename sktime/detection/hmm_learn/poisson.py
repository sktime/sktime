"""Hidden Markov Model with Poisson emissions.

Please see the original library
(https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py)
"""

from sktime.detection.hmm_learn import BaseHMMLearn

__author__ = ["klam-data", "pyyim", "mgorlin"]
__all__ = ["PoissonHMM"]


class PoissonHMM(BaseHMMLearn):
    """Hidden Markov Model with Poisson emissions.

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
    lambdas_prior, lambdas_weight : array, shape (n_components,), optional
        The gamma prior on the lambda values using alpha-beta notation,
        respectively. If None, will be set based on the method of
        moments.
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
        combination of 's' for startprob, 't' for transmat, and 'l' for
        lambdas.  Defaults to all parameters.
    implementation: string, optional
        Determines if the forward-backward algorithm is implemented with
        logarithms ("log"), or using scaling ("scaling").  The default is
        to use logarithms for backwards compatibility.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    lambdas_ : array, shape (n_components, n_features)
        The expectation value of the waiting time parameters for each
        feature in a given state.

    Examples
    --------
    >>> from sktime.detection.hmm_learn import PoissonHMM
    >>> from sktime.detection.datagen import piecewise_poisson
    >>> data = piecewise_poisson(
    ...    lambdas=[1, 2, 3], lengths=[2, 4, 8], random_state=7
    ...    ).reshape((-1, 1))
    >>> model = PoissonHMM(n_components=3)
    >>> model = model.fit(data)
    >>> labeled_data = model.predict(data)
    """

    _tags = {
        "distribution_type": "Poisson",
        # Tag to determine test for test_all_annotators
    }

    def __init__(
        self,
        n_components: int = 1,
        startprob_prior: float = 1.0,
        transmat_prior: float = 1.0,
        lambdas_prior: float = 0.0,
        lambdas_weight: float = 0.0,
        algorithm: str = "viterbi",
        random_state: int = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "stl",
        init_params: str = "stl",
        implementation: str = "log",
    ):
        self.n_components = n_components
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.lambdas_prior = lambdas_prior
        self.lambdas_weight = lambdas_weight
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.implementation = implementation
        super().__init__()

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
        from hmmlearn.hmm import PoissonHMM as _PoissonHMM

        self._hmm_estimator = _PoissonHMM(
            self.n_components,
            self.startprob_prior,
            self.transmat_prior,
            self.lambdas_prior,
            self.lambdas_weight,
            self.algorithm,
            self.random_state,
            self.n_iter,
            self.tol,
            self.verbose,
            self.params,
            self.init_params,
            self.implementation,
        )
        return super()._fit(X, Y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params0 = {
            "n_components": 3,
        }
        params1 = {
            "n_components": 5,
            "algorithm": "map",
        }

        return [params0, params1]

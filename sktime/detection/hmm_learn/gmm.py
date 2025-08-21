"""Hidden Markov Model with Gaussian mixture emissions.

Please see the original library
(https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py)
"""

from sktime.detection.hmm_learn import BaseHMMLearn

__author__ = ["miraep8"]
__all__ = ["GMMHMM"]


class GMMHMM(BaseHMMLearn):
    """Hidden Markov Model with Gaussian mixture emissions.

    Parameters
    ----------
    n_components : int
        Number of states in the model.
    n_mix : int
        Number of states in the GMM.
    covariance_type : {"spherical", "diag", "full", "tied"}, optional
        The type of covariance parameters to use:

        * "spherical" --- each state uses a single variance value that
            applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix
            (default).
        * "full" --- each state uses a full (i.e. unrestricted)
            covariance matrix.
        * "tied" --- all mixture components of each state use **the same**
            full covariance matrix (note that this is not the same as for
            ``GaussianHMM``).

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.
    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.
    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.
    weights_prior : array, shape (n_mix, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`weights_`.
    means_prior, means_weight : array, shape (n_mix, ), optional
        Mean and precision of the Normal prior distribution for
        :attr:`means_`.
    covars_prior, covars_weight : array, shape (n_mix, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.
        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.
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
        before (``init_params``) the training.  Can contain any combination
        of 's' for startprob, 't' for transmat, 'm' for means, 'c'
        for covars, and 'w' for GMM mixing weights.  Defaults to all
        parameters.
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
    weights_ : array, shape (n_components, n_mix)
        Mixture weights for each state.
    means_ : array, shape (n_components, n_mix, n_features)
        Mean parameters for each mixture component in each state.
    covars_ : array
        Covariance parameters for each mixture components in each state.
        The shape depends on :attr:`covariance_type`:

        * (n_components, n_mix)                          if "spherical",
        * (n_components, n_mix, n_features)              if "diag",
        * (n_components, n_mix, n_features, n_features)  if "full"
        * (n_components, n_features, n_features)         if "tied".

    Examples
    --------
    >>> from sktime.detection.hmm_learn import GMMHMM
    >>> from sktime.detection.datagen import piecewise_normal
    >>> data = piecewise_normal(
    ...    means=[2, 4, 1], lengths=[10, 35, 40], random_state=7
    ...    ).reshape((-1, 1))
    >>> model = GMMHMM(n_components=3)
    >>> model = model.fit(data)
    >>> labeled_data = model.predict(data)
    """

    def __init__(
        self,
        n_components: int = 1,
        n_mix: int = 1,
        min_covar: float = 1e-3,
        startprob_prior: float = 1.0,
        transmat_prior: float = 1.0,
        weights_prior: float = 1.0,
        means_prior: float = 0.0,
        means_weight: float = 0.0,
        covars_prior=None,
        covars_weight=None,
        algorithm: str = "viterbi",
        covariance_type: str = "diag",
        random_state=None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "stmcw",
        init_params: str = "stmcw",
        implementation: str = "log",
    ):
        self.n_components = n_components
        self.n_mix = n_mix
        self.min_covar = min_covar
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.algorithm = algorithm
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.implementation = implementation
        super().__init__()

    def _fit(self, X, Y=None):
        # import inside _fit to avoid hard dependency.
        from hmmlearn.hmm import GMMHMM as _GMMHMM

        self._hmm_estimator = _GMMHMM(
            self.n_components,
            self.n_mix,
            self.min_covar,
            self.startprob_prior,
            self.transmat_prior,
            self.weights_prior,
            self.means_prior,
            self.means_weight,
            self.covars_prior,
            self.covars_weight,
            self.algorithm,
            self.covariance_type,
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
        params1 = {
            "n_components": 3,
            "covariance_type": "diag",
            "min_covar": 1e-3,
            "random_state": 7,
        }

        params2 = {
            "n_components": 5,
            "covariance_type": "full",
            "min_covar": 1e-6,
            "random_state": 42,
        }

        return [params1, params2]

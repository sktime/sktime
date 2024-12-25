"""Time series SVR, from tslearn."""

from sktime.base.adapters._tslearn import _TslearnAdapter
from sktime.regression.base import BaseRegressor


class TimeSeriesSVRTslearn(_TslearnAdapter, BaseRegressor):
    """Time Series Suppoer Vector Regressor, from tslearn.

    Direct interface to ``tslearn.svm.svm.TimeSeriesSVR``.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='gak')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'gak' or a kernel accepted by ``sklearn.svm.SVR``.
         If none is given, 'gak' will be used. If a callable is given it is
         used to pre-compute the kernel matrix from data matrices; that matrix
         should be an array of shape ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'gak', 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then:

        - for 'gak' kernel, it is computed based on a sampling of the training
          set ``tslearn.metrics.gamma_soft_dtw``
        - for other kernels (eg. 'rbf'), 1/n_features will be used.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    epsilon : float, optional (default=0.1)
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    cache_size :  float, optional (default=200.0)
        Specify the size of the kernel cache (in MB).

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for GAK cross-similarity matrix
        computations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int, default: 0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.

    support_vectors_ : array of shape [n_SV, sz, d]
        Support vectors in tslearn dataset format

    dual_coef_ : array, shape = [1, n_SV]
        Coefficients of the support vector in the decision function.

    coef_ : array, shape = [1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : array, shape = [1]
        Constants in decision function.

    sample_weight : array-like, shape = [n_samples]
        Individual weights for each sample

    svm_estimator_ : sklearn.svm.SVR
        The underlying sklearn estimator
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["rtavenar", "yanncabanes", "fkiraly"],
        # rtavenar and yanncabanes for interfaced estimator
        "python_dependencies": "tslearn",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": False,
    }

    # defines the name of the attribute containing the tslearn estimator
    _estimator_attr = "_tslearn_svr"

    def _get_tslearn_class(self):
        """Get tslearn class.

        should import and return tslearn class
        """
        from tslearn.svm.svm import TimeSeriesSVR as _TimeSeriesSVR

        return _TimeSeriesSVR

    def __init__(
        self,
        C=1.0,
        kernel="gak",
        degree=3,
        gamma="auto",
        coef0=0.0,
        shrinking=True,
        tol=0.001,
        epsilon=0.1,
        cache_size=200,
        n_jobs=None,
        verbose=0,
        max_iter=-1,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter

        super().__init__()

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"cache_size": 50}
        params2 = {
            "C": 0.9,
            "kernel": "poly",
            "degree": 2,
            "gamma": 0.1,
            "coef0": 0.1,
            "tol": 0.01,
            "epsilon": 0.2,
            "shrinking": False,
            "cache_size": 55,
        }
        return [params1, params2]

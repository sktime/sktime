"""Time series SVC, from tslearn."""

from sktime.base.adapters._tslearn import _TslearnAdapter
from sktime.classification.base import BaseClassifier


class TimeSeriesSVCTslearn(_TslearnAdapter, BaseClassifier):
    """Time Series Suppoer Vector Classifier, from tslearn.

    Direct interface to ``tslearn.svm.svm.TimeSeriesSVC``.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='gak')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'gak' or a kernel accepted by ``sklearn.svm.SVC``.
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

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
        Also, probability estimates are not guaranteed to match predict output.
        See our :ref:`dedicated user guide section <kernels-ml>`
        for more details.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional (default=200.0)
        Specify the size of the kernel cache (in MB).

    class_weight : {dict, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

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

    decision_function_shape : 'ovo', 'ovr', default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2).

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.

    n_support_ : array-like, dtype=int32, shape = [n_class]
        Number of support vectors for each class.

    support_vectors_ : list of arrays of shape [n_SV, sz, d]
        List of support vectors in tslearn dataset format, one array per class

    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the section about multi-class classification in the
        SVM section of the User Guide of ``sklearn`` for details.

    coef_ : array, shape = [n_class-1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    svm_estimator_ : sklearn.svm.SVC
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
    _estimator_attr = "_tslearn_svc"

    def _get_tslearn_class(self):
        """Get tslearn class.

        should import and return tslearn class
        """
        from tslearn.svm.svm import TimeSeriesSVC as _TimeSeriesSVC

        return _TimeSeriesSVC

    def __init__(
        self,
        C=1.0,
        kernel="gak",
        degree=3,
        gamma="auto",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        n_jobs=None,
        verbose=0,
        max_iter=-1,
        decision_function_shape="ovr",
        random_state=None,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

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
            "kernel": "rbf",
            "gamma": 0.1,
            "shrinking": False,
            "probability": True,
            "tol": 0.01,
            "cache_size": 55,
            "decision_function_shape": "ovo",
        }
        return [params1, params2]

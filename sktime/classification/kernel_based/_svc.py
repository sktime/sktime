"""Support vector classifier using time series kernels..

Direct wrap of sklearn SVC with added functionality that allows time series kernel to be
passed, and uses the sktime time series classifier interface.
"""

__author__ = ["fkiraly"]
__all__ = ["TimeSeriesSVC"]

from inspect import signature

from sklearn.svm import SVC

from sktime.classification.base import BaseClassifier


class TimeSeriesSVC(BaseClassifier):
    """Support Vector Classifier, for time series kernels.

    An adapted version of the scikit-learn SVC for time series data.

    Any sktime pairwise transformers are supported as kernels,
    including time series kernels and standard kernels on "flattened" time series.

    Caveat: typically, SVC literature assumes kernels to be positive semi-definite.
    However, any pairwise transformer can be passed as kernel, including distances.
    This will still produce classification results, which may or may not be performant.

    Parameters
    ----------
    kernel : pairwise panel transformer or callable, optional, default see below
        pairwise panel transformer inheriting from ``BasePairwiseTransformerPanel``, or
        callable, must be of signature ``(X: Panel, X2: Panel) -> np.ndarray``
        output must be mxn array if ``X`` is Panel of m Series, ``X``2 of n Series
        if ``distance_mtype`` is not set, must be able to take
        ``X``, ``X2`` which are ``pd_multiindex`` and ``numpy3D`` mtype
        default = mean Euclidean kernel, same as ``AggrDist(RBF())``,
        where ``AggrDist`` is from ``sktime`` and ``RBF`` from ``sklearn``
    kernel_params : dict, optional. default = None.
        dictionary for distance parameters, in case that distance is a callable
    kernel_mtype : str, or list of str optional. default = None.
        mtype that ``kernel`` expects for X and X2, if a callable
        only set this if ``kernel`` is not ``BasePairwiseTransformerPanel`` descendant
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling ``fit``, will slow down that method as it internally uses
        5-fold cross-validation, and ``predict_proba`` may be inconsistent with
        ``predict``.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).
    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    decision_function_shape : ``{'ovo', 'ovr'}``, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
        ('ovo') is always used as multi-class strategy. The parameter is
        ignored for binary classification.
    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when ``probability`` is False.
        Pass an int for reproducible output across multiple function calls.

    Examples
    --------
    >>> from sktime.classification.kernel_based import TimeSeriesSVC
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> from sktime.dists_kernels import AggrDist
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>>
    >>> mean_gaussian_tskernel = AggrDist(RBF())
    >>> classifier = TimeSeriesSVC(kernel=mean_gaussian_tskernel)
    >>> classifier.fit(X_train, y_train)
    TimeSeriesSVC(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "fkiraly",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "X_inner_mtype": ["pd-multiindex", "numpy3D"],
        "classifier_type": "kernel",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    DELEGATED_PARAMS = [
        "C",
        "shrinking",
        "probability",
        "tol",
        "cache_size",
        "class_weight",
        "verbose",
        "max_iter",
        "decision_function_shape",
        "break_ties",
        "random_state",
    ]

    def __init__(
        self,
        kernel=None,
        kernel_params=None,
        kernel_mtype=None,
        C=1,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.kernel_mtype = kernel_mtype

        # naive dict comprehension does not work due to context of eval
        delegated_param_dict = {}
        for key in self.DELEGATED_PARAMS:
            delegated_param_dict[key] = eval(key)

        for key, val in delegated_param_dict.items():
            setattr(self, key, val)

        super().__init__()

        self.svc_estimator_ = SVC(kernel="precomputed", **delegated_param_dict)

        if kernel_mtype is not None:
            self.set_tags(X_inner_mtype=kernel_mtype)

        from sktime.dists_kernels import BasePairwiseTransformerPanel

        # inherit capability tags from distance, if it is an estimator
        if isinstance(kernel, BasePairwiseTransformerPanel):
            inherit_tags = [
                "capability:missing_values",
                "capability:unequal_length",
                "capability:multivariate",
            ]
            self.clone_tags(kernel, inherit_tags)

    def _kernel(self, X, X2=None):
        """Compute distance - unified interface to kernel callable."""
        kernel = self.kernel
        kernel_params = self.kernel_params
        if kernel_params is None:
            kernel_params = {}

        if kernel is None:
            from sklearn.gaussian_process.kernels import RBF

            from sktime.dists_kernels.compose_tab_to_panel import AggrDist

            kernel = AggrDist(RBF())

        if X2 is not None:
            return kernel(X, X2, **kernel_params)
        # if X2 is None, check if kernel allows None X2 to mean "X2=X"
        else:
            sig = signature(kernel).parameters
            X2_sig = sig[list(sig.keys())[1]]
            if X2_sig.default is not None:
                return kernel(X, X2, **kernel_params)
            else:
                return kernel(X, **kernel_params)

    def _fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : sktime compatible Panel data container, of mtype X_inner_mtype,
            with n time series to fit the estimator to
        y : {array-like, sparse matrix}
            Target values of shape = [n]
        """
        # store full data as indexed X
        self._X = X

        kernel_mat = self._kernel(X)

        self.svc_estimator_.fit(kernel_mat, y)

        return self

    def _predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : sktime-compatible Panel data, of mtype X_inner_mtype, with n_samples series
            data to predict class labels for

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        # self._X should be the stored _X
        kernel_mat = self._kernel(X, self._X)

        y_pred = self.svc_estimator_.predict(kernel_mat)

        return y_pred

    def _predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : sktime-compatible Panel data, of mtype X_inner_mtype, with n_samples series
            data to predict class labels for

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # self._X should be the stored _X
        kernel_mat = self._kernel(X, self._X)

        y_pred = self.svc_estimator_.predict_proba(kernel_mat)

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        # testing that callables/classes can be passed
        from sktime.dists_kernels.compose_tab_to_panel import FlatDist

        # probability must be True, or predict_proba will not work
        dist1 = FlatDist.create_test_instance()
        params1 = {"kernel": dist1, "probability": True}

        # testing the default kernel
        params2 = {"probability": True}

        return [params1, params2]

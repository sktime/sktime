"""Learning Shapelets classifier, from pyts."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["ShapeletLearningClassifierPyts"]

from sktime.base.adapters._pyts import _PytsAdapter
from sktime.classification.base import BaseClassifier


class ShapeletLearningClassifierPyts(_PytsAdapter, BaseClassifier):
    """Learning Shapelets algorithm, from pyts.

    Direct interface to ``pyts.classification.LearningShapelets`` or
    ``pyts.classification.LearningShapeletsCrossEntropy``,
    author of the interfaced classes is ``johannfaouzi``.

    Dispatches to ``LearningShapelets`` or ``LearningShapeletsCrossEntropy``,
    depending on the value of the ``loss`` parameter.

    This estimator consists of two steps: computing the distances between the
    shapelets and the time series, then carrying out empirical risk minimization
    using these distances as linear features.
    The risk is minimized using gradient descent; for ``loss="softmax"``,
    this is mathematically equivalent to multinomial logistic regression
    on shapelet features.
    This algorithm learns the shapelets as well as
    the coefficients of the classification.

    Parameters
    ----------
    loss : str (default = 'softmax'), "softmax" or "crossentropy"
        Loss function to use.
        If "softmax", the loss function is the softmax function (logistic loss).
        Dispatches to ``LearningShapelets``.
        If "crossentropy", the loss function is the cross-entropy loss.
        Dispatches to ``LearningShapeletsCrossEntropy``.

    n_shapelets_per_size : int or float (default = 0.2)
        Number of shapelets per size. If float, it represents
        a fraction of the number of timestamps and the number
        of shapelets per size is equal to
        ``ceil(n_shapelets_per_size * n_timestamps)``.

    min_shapelet_length : int or float (default = 0.1)
        Minimum length of the shapelets. If float, it represents
        a fraction of the number of timestamps and the minimum
        length of the shapelets per size is equal to
        ``ceil(min_shapelet_length * n_timestamps)``.

    shapelet_scale : int (default = 3)
        The different scales for the lengths of the shapelets.
        The lengths of the shapelets are equal to
        ``min_shapelet_length * np.arange(1, shapelet_scale + 1)``.
        The total number of shapelets (and features)
        is equal to ``n_shapelets_per_size * shapelet_scale``.

    penalty : 'l1' or 'l2' (default = 'l2')
        Used to specify the norm used in the penalization.

    tol : float (default = 1e-3)
        Tolerance for stopping criterion.

    C : float (default = 1000)
        Inverse of regularization strength. It must be a positive float.
        Smaller values specify stronger regularization.

    learning_rate : float (default = 1.)
        Learning rate for gradient descent optimization. It must be a positive
        float. Note that the learning rate will be automatically decreased
        if the loss function is not decreasing.

    max_iter : int (default = 1000)
        Maximum number of iterations for gradient descent algorithm.

    multi_class : {'multinomial', 'ovr', 'ovo'} (default = 'multinomial')
        Strategy for multiclass classification.
        Only used if ``loss="softmax"``. The options are as follows:
        'multinomial' stands for multinomial cross-entropy loss.
        'ovr' stands for one-vs-rest strategy.
        'ovo' stands for one-vs-one strategy.
        Ignored if the classification task is binary.

    alpha : float (default = -100)
        Scaling term in the softmin function. The lower, the more precised
        the soft minimum will be. Default value should be good for
        standardized time series.

    fit_intercept : bool (default = True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float (default = 1.)
        Scaling of the intercept. Only used if ``fit_intercept=True``.

    class_weight : dict, None or 'balanced' (default = None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have unit weight.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    n_jobs : None or int (default = None)
        The number of jobs to use for the computation.
        Only used if ``loss="softmax"`` and ``multi_class`` is "ovr" or "ovo".

    verbose : int (default = 0)
        Controls the verbosity. It must be a non-negative integer.
        If positive, loss at each iteration is printed.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    shapelets_ : array shape = (n_tasks, n_shapelets)
        Learned shapelets. Each element of this array is a learned
        shapelet.

    coef_ : array, shape = (n_tasks, n_shapelets) or (n_classes, n_shapelets)
        Coefficients for each shapelet in the decision function.

    intercept_ : array, shape = (n_tasks,) or (n_classes,)
        Intercepts (a.k.a. biases) added to the decision function.
        If ``fit_intercept=False``, the intercepts are set to zero.

    n_iter_ : array, shape = (n_tasks,)
        Actual number of iterations.

    Notes
    -----
    The number of tasks (n_tasks) depends on the value of ``multi_class``
    and the number of classes. If there are two classes, the number of
    tasks is equal to 1. If there are more than two classes, the number
    of tasks is equal to:

        - 1 if ``multi_class='multinomial'``
        - n_classes if ``multi_class='ovr'``
        - n_classes * (n_classes - 1) / 2 if ``multi_class='ovo'``

    References
    ----------
    .. [1] J. Grabocka, N. Schilling, M. Wistuba and L. Schmidt-Thieme,
           "Learning Time-Series Shapelets". International Conference on Data
           Mining, 14, 392-401 (2014).
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["johannfaouzi", "fkiraly"],  # johannfaouzi is author of upstream
        "python_dependencies": "pyts",
        # estimator type
        # --------------
        "capability:multioutput": False,
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "classifier_type": "shapelet",
    }

    # defines the name of the attribute containing the pyts estimator
    _estimator_attr = "_pyts_ls"

    def _get_pyts_class(self):
        """Get pyts class.

        should import and return pyts class
        """
        if self.loss == "crossentropy":
            from pyts.classification.learning_shapelets import (
                CrossEntropyLearningShapelets,
            )

            return CrossEntropyLearningShapelets
        elif self.loss == "softmax":
            from pyts.classification import LearningShapelets

            return LearningShapelets
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def _get_pyts_object(self):
        """Abstract method to initialize pyts object.

        The default initializes result of _get_pyts_class
        with self.get_params.
        """
        cls = self._get_pyts_class()
        params = self.get_params()
        params_inner = params.copy()
        params_inner.pop("loss")

        if self.loss == "crossentropy":
            params_inner.pop("multi_class")
            params_inner.pop("n_jobs")

        return cls(**params_inner)

    def __init__(
        self,
        loss="softmax",  # "softmax" or "crossentropy
        n_shapelets_per_size=0.2,
        min_shapelet_length=0.1,
        shapelet_scale=3,
        penalty="l2",
        tol=0.001,
        C=1000,
        learning_rate=1.0,
        max_iter=1000,
        multi_class="multinomial",
        alpha=-100,
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight=None,
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        self.loss = loss
        self.n_shapelets_per_size = n_shapelets_per_size
        self.min_shapelet_length = min_shapelet_length
        self.shapelet_scale = shapelet_scale
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

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
        params1 = {}
        params2 = {
            "loss": "softmax",
            "n_shapelets_per_size": 0.3,
            "min_shapelet_length": 0.11,
            "shapelet_scale": 4,
            "penalty": "l1",
            "tol": 0.01,
            "C": 100,
            "learning_rate": 0.9,
            "max_iter": 500,
            "multi_class": "ovr",
            "alpha": -99,
            "fit_intercept": False,
        }
        params3 = {
            "loss": "crossentropy",
            "n_shapelets_per_size": 0.3,
            "min_shapelet_length": 0.11,
            "shapelet_scale": 4,
            "penalty": "l1",
            "tol": 0.01,
            "C": 100,
            "learning_rate": 0.9,
            "max_iter": 500,
            "alpha": -99,
            "fit_intercept": False,
        }
        return [params1, params2, params3]

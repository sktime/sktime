# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""

__author__ = ["MatthewMiddlehurst", "victordremov", "fkiraly"]
__all__ = ["RocketClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from sktime.classification._delegate import _DelegatedClassifier
from sktime.pipeline import make_pipeline
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)


class RocketClassifier(_DelegatedClassifier):
    """Classifier wrapped for the Rocket transformer using RidgeClassifierCV.

    This classifier simply transforms the input data using the Rocket [1]_
    transformer and builds a RidgeClassifierCV estimator using the transformed data.

    Shorthand for the pipeline
    `rocket * StandardScaler(with_mean=False) * RidgeClassifierCV(alphas)`
    where `alphas = MultiRocketMultivariate`, and
    where `rocket` depends on params `rocket_transform`, `use_multivariate` as follows:

        | rocket_transform | `use_multivariate` | rocket (class)          |
        |------------------|--------------------|-------------------------|
        | "rocket"         | any                | Rocket                  |
        | "minirocket"     | True               | MiniRocketMultivariate  |
        | "minirocket"     | False              | MiniRocket              |
        | "multirocket"    | True               | MultiRocketMultivariate |
        | "multirocket"    | False              | MultiRocket             |

    classes are sktime classes, other parameters are passed on to the rocket class.

    To build other classifiers with rocket transformers, use `make_pipeline` or the
    pipeline dunder `*`, and different transformers/classifiers in combination.

    Parameters
    ----------
    num_kernels : int, optional, default=10,000
        The number of kernels the for Rocket transform.
    rocket_transform : str, optional, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket", "minirocket", "multirocket"]
    max_dilations_per_kernel : int, optional, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, optional, default=4
        MultiRocket only. The number of features per kernel.
    use_multivariate : bool, optional, default=True
        whether to use multivariate rocket transforms (True) or univariate ones (False)
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.
    estimator_ : ClassifierPipeline
        RocketClassifier as a ClassifierPipeline, fitted to data internally

    See Also
    --------
    Rocket

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/ROCKETClassifier.java>`_.

    References
    ----------
    .. [1] Dempster, Angus, François Petitjean, and Geoffrey I. Webb. "Rocket:
       exceptionally fast and accurate time series classification using random
       convolutional kernels." Data Mining and Knowledge Discovery 34.5 (2020)

    Examples
    --------
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = RocketClassifier(num_kernels=500)
    >>> clf.fit(X_train, y_train)
    RocketClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "classifier_type": "kernel",
    }

    # valid rocket strings for input validity checking
    VALID_ROCKET_STRINGS = ["rocket", "minirocket", "multirocket"]

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        use_multivariate=True,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.use_multivariate = use_multivariate

        self.n_jobs = n_jobs
        self.random_state = random_state

        super(RocketClassifier, self).__init__()

        if rocket_transform == "rocket":
            rocket = Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self._threads_to_use,
            )
        elif rocket_transform == "minirocket":
            if use_multivariate:
                rocket = MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
            else:
                rocket = MiniRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
        elif self.rocket_transform == "multirocket":
            if use_multivariate > 1:
                rocket = MultiRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
            else:
                rocket = MultiRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
        else:
            raise ValueError(
                f"Invalid rocket_transform string, must be one of "
                f"{self.VALID_ROCKET_STRINGS}, but found {rocket_transform}"
            )

        self.estimator_ = make_pipeline(
            rocket,
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )

        if not use_multivariate:
            self.set_tags(**{"capability:multivariate": False})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"num_kernels": 100}
        else:
            return {"num_kernels": 20}

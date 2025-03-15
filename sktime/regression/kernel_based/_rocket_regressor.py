"""RandOm Convolutional KErnel Transform (Rocket) regressor.

Pipeline regressor using the ROCKET transformer and RidgeCV estimator.
"""

__author__ = ["fkiraly"]
__all__ = ["RocketRegressor"]

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from sktime.pipeline import make_pipeline
from sktime.regression._delegate import _DelegatedRegressor
from sktime.regression.base import BaseRegressor
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)


class RocketRegressor(_DelegatedRegressor, BaseRegressor):
    """Regressor wrapped for the Rocket transformer using RidgeCV regressor.

    This regressor simply transforms the input data using the Rocket [1]_
    transformer and builds a RidgeCV estimator using the transformed data.

    Shorthand for the pipeline
    ``rocket * StandardScaler(with_mean=False) * RidgeCV(alphas)``
    where ``alphas = np.logspace(-3, 3, 10)``, and
    where ``rocket`` depends on params ``rocket_transform``, ``use_multivariate`` as
    follows:

        | rocket_transform | ``use_multivariate`` | rocket (class)          |
        |------------------|--------------------|-------------------------|
        | "rocket"         | any                | Rocket                  |
        | "minirocket"     | "yes               | MiniRocketMultivariate  |
        | "minirocket"     | "no"               | MiniRocket              |
        | "multirocket"    | "yes"              | MultiRocketMultivariate |
        | "multirocket"    | "no"               | MultiRocket             |

    classes are sktime classes, other parameters are passed on to the rocket class.

    To build other regressors with rocket transformers, use ``make_pipeline`` or the
    pipeline dunder ``*``, and different transformers/regressors in combination.

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
    use_multivariate : str, ["auto", "yes", "no"], optional, default="auto"
        whether to use multivariate rocket transforms or univariate ones
        "auto" = multivariate iff data seen in fit is multivariate, otherwise univariate
        "yes" = always uses multivariate transformers, native multi/univariate
        "no" = always univariate transformers, multivariate by framework vectorization
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.
    estimator_ : RegressorPipeline
        RocketRegressor as a RegressorPipeline, fitted to data internally
    num_kernels_ : int
        The true number of kernels used in the rocket transform. When
        rocket_transform="rocket", this is num_kernels. When rocket_transform
        is either "minirocket" or "multirocket", this is num_kernels rounded
        down to the nearest multiple of 84. It is 84 if num_kernels is less
        than 84.

    See Also
    --------
    Rocket, RocketClassifier

    References
    ----------
    .. [1] Dempster, Angus, François Petitjean, and Geoffrey I. Webb. "Rocket:
       exceptionally fast and accurate time series classification using random
       convolutional kernels." Data Mining and Knowledge Discovery 34.5 (2020)

    Examples
    --------
    >>> from sktime.regression.kernel_based import RocketRegressor
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True) # doctest: +SKIP
    >>> reg = RocketRegressor(num_kernels=500) # doctest: +SKIP
    >>> reg.fit(X_train, y_train) # doctest: +SKIP
    RocketRegressor(...)
    >>> y_pred = reg.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "fkiraly",
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    # valid rocket strings for input validity checking
    VALID_ROCKET_STRINGS = ["rocket", "minirocket", "multirocket"]
    VALID_MULTIVAR_VALUES = ["auto", "yes", "no"]

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        use_multivariate="auto",
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.rocket_transform = rocket_transform

        if rocket_transform in ["multirocket", "minirocket"]:
            if self.num_kernels < 84:
                self.num_kernels_ = 84
            else:
                self.num_kernels_ = (self.num_kernels // 84) * 84

        else:
            self.num_kernels_ = num_kernels

        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.use_multivariate = use_multivariate

        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

        if use_multivariate not in self.VALID_MULTIVAR_VALUES:
            raise ValueError(
                f"Invalid use_multivariate value, must be one of "
                f"{self.VALID_MULTIVAR_VALUES}, but found {use_multivariate}"
            )

        common_params = {
            "num_kernels": self.num_kernels,
            "random_state": self.random_state,
            "max_dilations_per_kernel": self.max_dilations_per_kernel,
            "n_jobs": self._threads_to_use,
        }

        if rocket_transform == "rocket":
            del common_params["max_dilations_per_kernel"]
            univar_rocket = Rocket(**common_params)
            multivar_rocket = univar_rocket

        elif rocket_transform == "minirocket":
            multivar_rocket = MiniRocketMultivariate(**common_params)
            univar_rocket = MiniRocket(**common_params)

        elif self.rocket_transform == "multirocket":
            common_params["n_features_per_kernel"] = self.n_features_per_kernel
            multivar_rocket = MultiRocketMultivariate(**common_params)
            univar_rocket = MultiRocket(**common_params)

        else:
            raise ValueError(
                f"Invalid rocket_transform string, must be one of "
                f"{self.VALID_ROCKET_STRINGS}, but found {rocket_transform}"
            )

        self.multivar_rocket_ = make_pipeline(
            multivar_rocket,
            StandardScaler(with_mean=False),
            RidgeCV(alphas=np.logspace(-3, 3, 10)),
        )
        self.univar_rocket_ = make_pipeline(
            univar_rocket,
            StandardScaler(with_mean=False),
            RidgeCV(alphas=np.logspace(-3, 3, 10)),
        )

        if not use_multivariate:
            self.set_tags(**{"capability:multivariate": False})

    @property
    def estimator_(self):
        """Shorthand for the internal estimator that is fitted."""
        return self._get_delegate()

    def _get_delegate(self):
        use_multivariate = self.use_multivariate
        if use_multivariate == "auto":
            code_dict = {True: "yes", False: "no"}
            use_multivariate = code_dict[not self._X_metadata["is_univariate"]]

        if use_multivariate == "yes":
            delegate = self.multivar_rocket_
        else:
            delegate = self.univar_rocket_

        return delegate

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        params1 = {"num_kernels": 20}
        params2 = {
            "num_kernels": 30,
            "rocket_transform": "rocket",
            "max_dilations_per_kernel": 24,
            "n_features_per_kernel": 3,
        }
        params3 = {"num_kernels": 20, "rocket_transform": "minirocket"}
        params4 = {"num_kernels": 20, "rocket_transform": "multirocket"}
        return [params1, params2, params3, params4]

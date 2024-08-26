"""Rocket transformer, from pyts."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["RocketPyts"]

from sktime.base.adapters._pyts import _PytsAdapter
from sktime.transformations.base import BaseTransformer


class RocketPyts(_PytsAdapter, BaseTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET), from ``pyts``.

    Direct interface to ``pyts.transformation.rocket``.

    ROCKET [1]_ generates random convolutional kernels, including random length and
    dilation. It transforms the time series with two features per kernel. The first
    feature is global max pooling and the second is proportion of positive values.

    This transformer fits one set of paramereters per individual series,
    and applies the transform with fitted parameter i to the i-th series in transform.
    Vanilla use requires same number of series in fit and transform.

    To fit and transform series at the same time,
    without an identification of fit/transform instances,
    wrap this transformer in ``FitInTransform``,
    from ``sktime.transformations.compose``.

    Parameters
    ----------
    n_kernels : int (default = 10000)
        Number of kernels.

    kernel_sizes : array-like (default = (7, 9, 11))
        The possible sizes of the kernels.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    weights_ : array, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels. Zero padding values are added.

    length_ : array, shape = (n_kernels,)
        Length of each kernel.

    bias_ : array, shape = (n_kernels,)
        Bias of each kernel.

    dilation_ : array, shape = (n_kernels,)
        Dilation of each kernel.

    padding_ : array, shape = (n_kernels,)
        Padding of each kernel.

    See Also
    --------
    MultiRocketMultivariate, MiniRocket, MiniRocketMultivariate, Rocket

    References
    ----------
    .. [1] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph
        and Webb, Geoffrey I,
        "ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels",2020,
      https://link.springer.com/article/10.1007/s10618-020-00701-z,
      https://arxiv.org/abs/1910.13051

    Examples
    --------
    >>> from sktime.transformations.panel.rocket import RocketPyts
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test") # doctest: +SKIP
    >>> trf = RocketPyts(num_kernels=512) # doctest: +SKIP
    >>> trf.fit(X_train) # doctest: +SKIP
    Rocket(...)
    >>> X_train = trf.transform(X_train) # doctest: +SKIP
    >>> X_test = trf.transform(X_test) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["johannfaouzi", "fkiraly"],  # johannfaouzi is author of upstream
        "python_dependencies": "pyts",
        # estimator type
        # --------------
        "univariate-only": True,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
    }

    # defines the name of the attribute containing the pyts estimator
    _estimator_attr = "_pyts_rocket"

    def _get_pyts_class(self):
        """Get pyts class.

        should import and return pyts class
        """
        from pyts.transformation.rocket import ROCKET

        return ROCKET

    def __init__(
        self,
        n_kernels=10_000,
        kernel_sizes=(7, 9, 11),
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state

        super().__init__()

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"n_kernels": 234, "kernel_sizes": (5, 4)}
        params2 = {"n_kernels": 512, "kernel_sizes": (6, 7, 8)}
        return [params1, params2]

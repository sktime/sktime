import multiprocessing

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__author__ = ["ChangWeiTan", "fstinner", "angus924"]


class MultiRocketMultivariate(BaseTransformer):
    """Multi RandOm Convolutional KErnel Transform (MultiRocket).

    MultiRocket [1]_ is uses the same set of kernels as MiniRocket on both the raw
    series and the first order differenced series representation. It uses a different
    set of dilations and used for each representation. In addition to percentage of
    positive values (PPV) MultiRocket adds 3 pooling operators: Mean of Positive
    Values (MPV); Mean of Indices of Positive Values (MIPV); and Longest Stretch of
    Positive Values (LSPV). This version is the multivariate version.

    This transformer fits one set of paramereters per individual series,
    and applies the transform with fitted parameter i to the i-th series in transform.
    Vanilla use requires same number of series in fit and transform.

    To fit and transform series at the same time,
    without an identification of fit/transform instances,
    wrap this transformer in ``FitInTransform``,
    from ``sktime.transformations.compose``.

    Parameters
    ----------
    num_kernels : int, default=6,250
       number of random convolutional kernels. This should be a multiple of 84.
       If it is lower than 84, it will be set to 84. If it is higher than 84
       and not a multiple of 84, the number of kernels used to transform the
       data will rounded down to the next positive multiple of 84.
    max_dilations_per_kernel : int, default=32
        maximum number of dilations per kernel.
    n_features_per_kernel : int, default =4
        number of features per kernel.
    normalise : bool, default False
    n_jobs : int, default=1
        The number of jobs to run in parallel for `transform`. ``-1`` means using all
        processors.
    random_state : None or int, default = None

    Attributes
    ----------
    parameter : tuple
        parameter (dilations, num_features_per_dilation, biases) for
        transformation of input X
    parameter1 : tuple
        parameter (dilations, num_features_per_dilation, biases) for
        transformation of input X1 = np.diff(X, 1)
    num_kernels_ : int
        The true number of kernels used in the rocket transform. This is
        num_kernels rounded down to the nearest multiple of 84. It is 84 if
        num_kernels is less than 84. The calculated number of features is given
        as 2*n_features_per_kernel*num_kernels_.

    See Also
    --------
    MultiRocketMultivariate, MiniRocket, MiniRocketMultivariate, Rocket

    References
    ----------
    .. [1] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and
    Webb, Geoffrey I, "MultiRocket: Multiple pooling operators and transformations
    for fast and effective time series classification",2022,
    https://link.springer.com/article/10.1007/s10618-022-00844-1
    https://arxiv.org/abs/2102.00457

    Examples
    --------
     >>> from sktime.transformations.panel.rocket import Rocket
     >>> from sktime.datasets import load_basic_motions
     >>> X_train, y_train = load_basic_motions(split="train") # doctest: +SKIP
     >>> X_test, y_test = load_basic_motions(split="test") # doctest: +SKIP
     >>> trf = MultiRocketMultivariate(num_kernels=512) # doctest: +SKIP
     >>> trf.fit(X_train) # doctest: +SKIP
     MultiRocketMultivariate(...)
     >>> X_train = trf.transform(X_train) # doctest: +SKIP
     >>> X_test = trf.transform(X_test) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ChangWeiTan", "fstinner", "angus924"],
        "maintainers": ["ChangWeiTan", "fstinner", "angus924"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "univariate-only": False,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
    }

    def __init__(
        self,
        num_kernels=6_250,
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        normalise=False,
        n_jobs=1,
        random_state=None,
    ):
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.num_kernels = num_kernels
        self.num_kernels_ = None
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None

        self.parameter = None
        self.parameter1 = None

        super().__init__()

    def _fit(self, X, y=None):
        """Fit dilations and biases to input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )

        if X.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _X1 = np.zeros((X.shape[0], X.shape[1], 10), dtype=X.dtype)
            _X1[:, :, : X.shape[2]] = X
            X = _X1
            del _X1

        X = X.astype(np.float64)

        self.parameter = self._get_parameter(X)
        _X1 = np.diff(X, 1)

        self.parameter1 = self._get_parameter(_X1)

        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        return self

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame, transformed features
        """
        from numba import get_num_threads, set_num_threads

        from sktime.transformations.panel.rocket._multirocket_multi_numba import (
            _transform,
        )

        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )

        _X1 = np.diff(X, 1)

        # change n_jobs depended on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)

        X = _transform(
            X,
            _X1,
            self.parameter,
            self.parameter1,
            self.n_features_per_kernel,
        )
        X = np.nan_to_num(X)

        set_num_threads(prev_threads)

        return pd.DataFrame(X)

    def _get_parameter(self, X):
        from sktime.transformations.panel.rocket._multirocket_multi_numba import (
            _fit_biases,
            _fit_dilations,
            _quantiles,
        )

        if self.random_state is not None:
            np.random.seed(self.random_state)

        _, num_channels, input_length = X.shape

        num_kernels = 84

        dilations, num_features_per_dilation = _fit_dilations(
            input_length, self.num_kernels, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)

        quantiles = _quantiles(num_kernels * num_features_per_kernel)

        num_dilations = len(dilations)
        num_combinations = num_kernels * num_dilations

        max_num_channels = min(num_channels, 9)
        max_exponent = np.log2(max_num_channels + 1)

        num_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent, num_combinations)
        ).astype(np.int32)

        channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

        num_channels_start = 0
        for combination_index in range(num_combinations):
            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]
            num_channels_end = num_channels_start + num_channels_this_combination
            channel_indices[num_channels_start:num_channels_end] = np.random.choice(
                num_channels, num_channels_this_combination, replace=False
            )

            num_channels_start = num_channels_end

        biases = _fit_biases(
            X,
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            quantiles,
            self.random_state,
        )

        return (
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            biases,
        )

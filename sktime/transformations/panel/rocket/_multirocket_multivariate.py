# -*- coding: utf-8 -*-
import multiprocessing

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class MultiRocketMultivariate(BaseTransformer):
    """
    MultiRocket multivariate version.

    Multi RandOm Convolutional KErnel Transform

    **Multivariate**

    A provisional and naive extension of MultiRocket to multivariate input.  Use
    class MultiRocket for univariate input.

    @article{Tan2021MultiRocket,
    title={{MultiRocket}: Multiple pooling operators and transformations
    for fast and effective time series classification},
    author={Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and
    Webb, Geoffrey I},
    year={2021},
    journal={arxiv:2102.00457v3}
    }

    Parameters
    ----------
    num_kernels              : int, (default=6,250)
                                number of random convolutional kernels
                                calculated number of features is the nearest multiple of
                                n_features_per_kernel
                                (default is 4)*84=336 < 50,000
                                (2*n_features_per_kernel*num_kernels)
    max_dilations_per_kernel : int, (default=32)
                                maximum number of dilations per kernel
    n_features_per_kernel    : int, (default=4)
                                number of features per kernel
    normalise                : int, (default=False)
                                normalise the data
    n_jobs                   : int, (default=1)
                                The number of jobs to run in parallel
                                for `transform`. ``-1`` means using all processors.
    random_state             : int, (default=None)
                                random seed

    Attributes
    ----------
    parameter : tuple
        parameter (dilations, num_features_per_dilation, biases) for
        transformation of input X
    parameter1 : tuple
        parameter (dilations, num_features_per_dilation, biases) for
        transformation of input X1 = np.diff(X, 1)

    See Also
    --------
    MultiRocket, MiniRocket, MiniRocketMultivariateVariable, Rocket

    References
    ----------
    .. [1] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph
        and Webb, Geoffrey I,
        "MultiRocket: Multiple pooling operators and transformations
        for fast and effective time series classification",
        2021, https://arxiv.org/abs/2102.00457v3

    Examples
    --------
    >>> from sktime.transformations.panel.rocket._multirocket import MultiRocket
    >>> from sktime.datasets import load_basic_motions
    >>> X_train, y_train = load_basic_motions(split="train")
    >>> X_test, y_test = load_basic_motions(split="test")
    >>> trf = MultiRocket(num_kernels=512)
    >>> trf.fit(X_train)
    MultiRocket(...)
    >>> X_train = trf.transform(X_train)
    >>> X_test = trf.transform(X_test)
    """

    _tags = {
        "univariate-only": False,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "python_dependencies": "numba",
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
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None

        self.parameter = None
        self.parameter1 = None

        super(MultiRocketMultivariate, self).__init__()

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

        # change n_jobs dependend on value and existing cores
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

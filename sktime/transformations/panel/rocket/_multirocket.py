# -*- coding: utf-8 -*-
"""MultiRocket transform."""

import multiprocessing

import numpy as np
import pandas as pd

from sktime.datatypes import convert
from sktime.transformations.base import BaseTransformer


class MultiRocket(BaseTransformer):
    """Multi RandOm Convolutional KErnel Transform (MultiRocket).

    MultiRocket [1]_ is uses the same set of kernels as MiniRocket on both the raw
    series and the first order differenced series representation. It uses a different
    set of dilations and used for each representation. In addition to percentage of
    positive values (PPV) MultiRocket adds 3 pooling operators: Mean of Positive
    Values (MPV); Mean of Indices of Positive Values (MIPV); and Longest Stretch of
    Positive Values (LSPV). This version is for univariate time series only. Use class
    MultiRocketMultivariate for multivariate input.

    Parameters
    ----------
    num_kernels : int, default = 6,250
       number of random convolutional kernels. The calculated number of features is the
       nearest multiple of n_features_per_kernel(default 4)*84=336 < 50,000
       (2*n_features_per_kernel(default 4)*num_kernels(default 6,250)).
    max_dilations_per_kernel : int, default = 32
        maximum number of dilations per kernel.
    n_features_per_kernel : int, default = 4
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
     >>> from sktime.datasets import load_unit_test
     >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
     >>> X_test, y_test = load_unit_test(split="test") # doctest: +SKIP
     >>> trf = MultiRocket(num_kernels=512) # doctest: +SKIP
     >>> trf.fit(X_train) # doctest: +SKIP
     MultiRocket(...)
     >>> X_train = trf.transform(X_train) # doctest: +SKIP
     >>> X_test = trf.transform(X_test) # doctest: +SKIP
    """

    _tags = {
        "univariate-only": True,
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

        super(MultiRocket, self).__init__()

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
        X = X.astype(np.float64)
        X = convert(X, from_type="numpy3D", to_type="numpyflat", as_scitype="Panel")
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )

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

        from sktime.transformations.panel.rocket._multirocket_numba import _transform

        X = X.astype(np.float64)
        X = convert(X, from_type="numpy3D", to_type="numpyflat", as_scitype="Panel")
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )

        X1 = np.diff(X, 1)

        # change n_jobs dependend on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)

        X = _transform(
            X,
            X1,
            self.parameter,
            self.parameter1,
            self.n_features_per_kernel,
        )
        X = np.nan_to_num(X)

        set_num_threads(prev_threads)
        # # from_2d_array_to_3d_numpy
        # _X = np.reshape(_X, (_X.shape[0], 1, _X.shape[1])).astype(np.float64)
        return pd.DataFrame(X)

    def _get_parameter(self, X):
        from sktime.transformations.panel.rocket._multirocket_numba import (
            _fit_biases,
            _fit_dilations,
            _quantiles,
        )

        _, input_length = X.shape

        num_kernels = 84

        dilations, num_features_per_dilation = _fit_dilations(
            input_length, self.num_kernels, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)

        quantiles = _quantiles(num_kernels * num_features_per_kernel)

        biases = _fit_biases(
            X, dilations, num_features_per_dilation, quantiles, self.random_state
        )

        return dilations, num_features_per_dilation, biases

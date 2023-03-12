# -*- coding: utf-8 -*-
"""Multivariate MiniRocket transformer."""

__author__ = "angus924"
__all__ = ["MiniRocketMultivariate"]

import multiprocessing

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class MiniRocketMultivariate(BaseTransformer):
    """MINImally RandOm Convolutional KErnel Transform (MiniRocket) multivariate.

    MiniRocketMultivariate [1]_ is an almost deterministic version of Rocket. If creates
    convolutions of length of 9 with weights restricted to two values, and uses 84 fixed
    convolutions with six of one weight, three of the second weight to seed dilations.
    MiniRocketMultivariate works with univariate and multivariate time series.

    Parameters
    ----------
    num_kernels : int, default=10,000
       number of random convolutional kernels.
    max_dilations_per_kernel : int, default=32
        maximum number of dilations per kernel.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `transform`. ``-1`` means using all
        processors.
    random_state : None or int, default = None

    See Also
    --------
    MultiRocketMultivariate, MiniRocket, MiniRocketMultivariate, Rocket

    References
    ----------
    .. [1] Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I,
        "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series
        Classification",2020,
        https://dl.acm.org/doi/abs/10.1145/3447548.3467231,
        https://arxiv.org/abs/2012.08791

    Examples
    --------
     >>> from sktime.transformations.panel.rocket import Rocket
     >>> from sktime.datasets import load_basic_motions
     >>> X_train, y_train = load_basic_motions(split="train")
     >>> X_test, y_test = load_basic_motions(split="test") # doctest: +SKIP
     >>> trf = MiniRocketMultivariate(num_kernels=512) # doctest: +SKIP
     >>> trf.fit(X_train) # doctest: +SKIP
     MiniRocketMultivariate(...)
     >>> X_train = trf.transform(X_train) # doctest: +SKIP
     >>> X_test = trf.transform(X_test) # doctest: +SKIP
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
        num_kernels=10_000,
        max_dilations_per_kernel=32,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel

        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(
                f"random_state in MiniRocketMultivariate must be int or None, "
                f"but found {type(random_state)}"
            )
        if isinstance(random_state, int):
            self.random_state_ = np.int32(random_state)
        else:
            self.random_state_ = random_state

        super(MiniRocketMultivariate, self).__init__()

    def _fit(self, X, y=None):
        """Fits dilations and biases to input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        from sktime.transformations.panel.rocket._minirocket_multi_numba import (
            _fit_multi,
        )

        X = X.astype(np.float32)
        *_, n_timepoints = X.shape
        if n_timepoints < 9:
            raise ValueError(
                (
                    f"n_timepoints must be >= 9, but found {n_timepoints};"
                    " zero pad shorter series so that n_timepoints == 9"
                )
            )
        self.parameters = _fit_multi(
            X, self.num_kernels, self.max_dilations_per_kernel, self.random_state_
        )
        return self

    def _transform(self, X, y=None):
        """Transform input time series.

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

        from sktime.transformations.panel.rocket._minirocket_multi_numba import (
            _transform_multi,
        )

        X = X.astype(np.float32)
        # change n_jobs dependend on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        X_ = _transform_multi(X, self.parameters)
        set_num_threads(prev_threads)
        return pd.DataFrame(X_)

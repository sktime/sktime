"""Rocket transformer."""

__author__ = ["angus924"]
__all__ = ["Rocket"]

import multiprocessing

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class Rocket(BaseTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET).

    ROCKET [1]_ generates random convolutional kernels, including random length and
    dilation. It transforms the time series with two features per kernel. The first
    feature is global max pooling and the second is proportion of positive values.


    Parameters
    ----------
    num_kernels : int, default=10,000
       number of random convolutional kernels.
    normalise : boolean, default True
       whether or not to normalise the input time series per instance.
    n_jobs : int, default=1
       The number of jobs to run in parallel for `transform`. ``-1`` means use all
       processors.
    random_state : None or int, optional, default = None

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
    >>> from sktime.transformations.panel.rocket import Rocket
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test") # doctest: +SKIP
    >>> trf = Rocket(num_kernels=512) # doctest: +SKIP
    >>> trf.fit(X_train) # doctest: +SKIP
    Rocket(...)
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

    def __init__(self, num_kernels=10_000, normalise=True, n_jobs=1, random_state=None):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None
        super().__init__()

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels / dimensions (
        for multivariate time series) from input pandas DataFrame,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        from sktime.transformations.panel.rocket._rocket_numba import _generate_kernels

        _, self.n_columns, n_timepoints = X.shape
        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, self.n_columns, self.random_state
        )
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

        from sktime.transformations.panel.rocket._rocket_numba import _apply_kernels

        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        t = pd.DataFrame(_apply_kernels(X.astype(np.float32), self.kernels))
        set_num_threads(prev_threads)
        return t

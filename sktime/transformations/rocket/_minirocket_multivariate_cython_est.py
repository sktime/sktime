"""Multivariate MiniRocket transformer, Cython implementation (no numba)."""

__author__ = ["sssilvar"]
__all__ = ["MiniRocketMultivariateCython"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class MiniRocketMultivariateCython(BaseTransformer):
    """MiniRocket multivariate transform, Cython backend (no numba).

    Numerically equivalent to ``MiniRocketMultivariate`` but uses ahead-of-time
    compiled Cython kernels instead of numba, eliminating JIT "warmup" latency
    (which can exceed 30s on the first transform of larger series). The numba
    ``MiniRocketMultivariate`` is retained as the reference/groundtruth.

    MiniRocketMultivariate [1]_ is an almost deterministic version of Rocket. It
    creates convolutions of length 9 with weights restricted to two values, and
    uses 84 fixed convolutions with six of one weight, three of the second weight
    to seed dilations. Works with univariate and multivariate time series.

    This transformer fits one set of parameters per individual series, and
    applies the transform with fitted parameter i to the i-th series in transform.
    Vanilla use requires the same number of series in fit and transform.

    Migrating fitted parameters to/from ``MiniRocketMultivariate``
    --------------------------------------------------------------
    ``save``/``load`` are *not* interchangeable between this class and the numba
    ``MiniRocketMultivariate`` (each restores its own type). The fitted
    ``parameters`` tuple is, however, identical in format, so a fitted estimator
    can be migrated by copying that attribute (transforms then match to ~1e-5,
    a float32 rounding difference)::

        src = MiniRocketMultivariate(num_kernels=..., random_state=...).fit(X)
        dst = MiniRocketMultivariateCython(num_kernels=..., random_state=...)
        dst.fit(X)                 # any panel with the same n_columns
        dst.parameters = src.parameters

    The direction also works in reverse (Cython -> numba).

    Parameters
    ----------
    num_kernels : int, default=10,000
       number of random convolutional kernels. This should be a multiple of 84.
       If it is lower than 84, it will be set to 84. If it is higher than 84
       and not a multiple of 84, the number of kernels used to transform the
       data will be rounded down to the next positive multiple of 84.
    max_dilations_per_kernel : int, default=32
        maximum number of dilations per kernel.
    n_jobs : int, default=1
        Number of threads used in ``transform`` (the GIL-releasing Cython kernel
        is run over disjoint instance chunks). ``-1`` uses all processors.
    random_state : None or int, default = None

    Attributes
    ----------
    num_kernels_ : int
        The true number of kernels used in the rocket transform. This is
        num_kernels rounded down to the nearest multiple of 84. It is 84 if
        num_kernels is less than 84.

    See Also
    --------
    MiniRocketMultivariate, MultiRocketMultivariate, MiniRocket, Rocket

    References
    ----------
    .. [1] Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I,
        "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series
        Classification",2020,
        https://dl.acm.org/doi/abs/10.1145/3447548.3467231,
        https://arxiv.org/abs/2012.08791

    Examples
    --------
     >>> from sktime.transformations.rocket import MiniRocketMultivariateCython
     >>> from sktime.datasets import load_basic_motions
     >>> X_train, y_train = load_basic_motions(split="train") # doctest: +SKIP
     >>> trf = MiniRocketMultivariateCython(num_kernels=512) # doctest: +SKIP
     >>> trf.fit(X_train) # doctest: +SKIP
     MiniRocketMultivariateCython(...)
     >>> X_train = trf.transform(X_train) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["sssilvar"],
        "maintainers": ["sssilvar"],
        "python_dependencies": ["sktime-cython"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # test and CI flags
        # -----------------
        "tests:vm": True,
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
        self.num_kernels_ = None
        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(
                f"random_state in MiniRocketMultivariateCython must be int or None, "
                f"but found {type(random_state)}"
            )
        if isinstance(random_state, int):
            self.random_state_ = np.int32(random_state)
        else:
            self.random_state_ = random_state

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
        X = X.astype(np.float32)
        _, n_columns, n_timepoints = X.shape
        if n_timepoints < 9:
            raise ValueError(
                f"n_timepoints must be >= 9, but found {n_timepoints};"
                " zero pad shorter series so that n_timepoints == 9"
            )
        self.parameters = self._fit_params(
            X, self.num_kernels, self.max_dilations_per_kernel, self.random_state_
        )
        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        return self

    @staticmethod
    def _fit_params(X, num_features, max_dilations_per_kernel, seed):
        """Pure-numpy + Cython fit, equivalent to numba ``_fit_multi``."""
        # reuse the non-numba fit scaffolding from the numba module
        from sktime_cython.transformations.rocket import (
            _minirocket_multivariate_cython as _cy,
        )
        from sktime_cython.transformations.rocket._minirocket import (
            _fit_dilations,
            _quantiles,
        )

        if seed is not None:
            np.random.seed(seed)

        _, n_columns, n_timepoints = X.shape
        num_kernels = 84

        dilations, num_features_per_dilation = _fit_dilations(
            n_timepoints, num_features, max_dilations_per_kernel
        )
        num_features_per_kernel = np.sum(num_features_per_dilation)
        quantiles = _quantiles(num_kernels * num_features_per_kernel)

        num_dilations = len(dilations)
        num_combinations = num_kernels * num_dilations

        max_num_channels = min(n_columns, 9)
        max_exponent = np.log2(max_num_channels + 1)

        num_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent, num_combinations)
        ).astype(np.int32)

        channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)
        num_channels_start = 0
        for combination_index in range(num_combinations):
            n_this = num_channels_per_combination[combination_index]
            num_channels_end = num_channels_start + n_this
            channel_indices[num_channels_start:num_channels_end] = np.random.choice(
                n_columns, n_this, replace=False
            )
            num_channels_start = num_channels_end

        # biases: re-seed (matching numba _fit_biases_multi), draw one instance
        # index per combination, build C in Cython, then quantile per combination.
        if seed is not None:
            np.random.seed(seed)
        n_instances = X.shape[0]
        instance_indices = np.array(
            [np.random.randint(n_instances) for _ in range(num_combinations)],
            dtype=np.int32,
        )
        C = _cy.fit_biases(
            np.ascontiguousarray(X, dtype=np.float32),
            num_channels_per_combination,
            channel_indices,
            dilations.astype(np.int32),
            num_features_per_dilation.astype(np.int32),
            instance_indices,
        )

        biases = np.zeros(
            num_kernels * int(np.sum(num_features_per_dilation)), dtype=np.float32
        )
        feature_index_start = 0
        combination_index = 0
        for dilation_index in range(num_dilations):
            nfd = num_features_per_dilation[dilation_index]
            for _kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + nfd
                biases[feature_index_start:feature_index_end] = np.quantile(
                    C[combination_index],
                    quantiles[feature_index_start:feature_index_end],
                ).astype(np.float32)
                feature_index_start = feature_index_end
                combination_index += 1

        return (
            num_channels_per_combination,
            channel_indices,
            dilations.astype(np.int32),
            num_features_per_dilation.astype(np.int32),
            biases,
        )

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
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor

        from sktime_cython.transformations.rocket import (
            _minirocket_multivariate_cython as _cy,
        )

        X = np.ascontiguousarray(X, dtype=np.float32)

        n_instances = X.shape[0]
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        n_jobs = min(n_jobs, n_instances)

        if n_jobs <= 1:
            X_ = _cy.transform(X, *self.parameters)
        else:
            # the Cython kernel releases the GIL, so plain threads run truly
            # in parallel across disjoint instance chunks.
            bounds = np.linspace(0, n_instances, n_jobs + 1).astype(int)
            chunks = [
                np.ascontiguousarray(X[bounds[i] : bounds[i + 1]])
                for i in range(n_jobs)
                if bounds[i + 1] > bounds[i]
            ]
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                parts = list(
                    ex.map(lambda c: _cy.transform(c, *self.parameters), chunks)
                )
            X_ = np.vstack(parts)
        return pd.DataFrame(X_)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params = [
            {
                "num_kernels": 84,
                "random_state": 42,
                "max_dilations_per_kernel": 32,
            },
            {
                "num_kernels": 42,
                "random_state": 84,
                "max_dilations_per_kernel": 16,
            },
        ]
        return params

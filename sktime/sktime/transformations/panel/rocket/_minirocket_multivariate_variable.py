"""Multivariate MiniRocket transformer."""

__author__ = ["angus924", "michaelfeil"]
__all__ = ["MiniRocketMultivariateVariable"]

import multiprocessing
import warnings
from typing import Union

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class MiniRocketMultivariateVariable(BaseTransformer):
    """MINIROCKET (Multivariate, unequal length).

    MINImally RandOm Convolutional KErnel Transform. [1]_

    **Multivariate** and **unequal length**

    A provisional and naive extension of MINIROCKET to multivariate input
    with unequal length provided by the authors [2]_ .  For better
    performance, use the sktime class MiniRocket for univariate input,
    and MiniRocketMultivariate to equal length multivariate input.

    This transformer fits one set of paramereters per individual series,
    and applies the transform with fitted parameter i to the i-th series in transform.
    Vanilla use requires same number of series in fit and transform.

    To fit and transform series at the same time,
    without an identification of fit/transform instances,
    wrap this transformer in ``FitInTransform``,
    from ``sktime.transformations.compose``.

    Parameters
    ----------
    num_kernels : int, default=10_000
       number of random convolutional kernels. This should be a multiple of 84.
       If it is lower than 84, it will be set to 84. If it is higher than 84
       and not a multiple of 84, the number of kernels used to transform the
       data will rounded down to the next positive multiple of 84.
    max_dilations_per_kernel : int, default=32
        maximum number of dilations per kernel.
    reference_length : int or str, default = ``'max'``
        series-length of reference, str defines how to infer from X during 'fit'.
        options are ``'max'``, ``'mean'``, ``'median'``, ``'min'``.
    pad_value_short_series : float or None, default=None
        if padding series with len<9 to value. if None, not padding is performed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for ``transform``. ``-1`` means using all
        processors.
    random_state : None or int, default = None

    Attributes
    ----------
    num_kernels_ : int
        The true number of kernels used in the rocket transform. This is
        num_kernels rounded down to the nearest multiple of 84. It is 84 if
        num_kernels is less than 84.

    Examples
    --------
    >>> from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable
    >>> from sktime.datasets import load_japanese_vowels
    >>> # load multivariate and unequal length dataset
    >>> X_train, _ = load_japanese_vowels(split="train", return_X_y=True)
    >>> X_test, _ = load_japanese_vowels(split="test", return_X_y=True)
    >>> pre_clf = MiniRocketMultivariateVariable(
    ...     pad_value_short_series=0.0
    ... ) # doctest: +SKIP
    >>> pre_clf.fit(X_train, y=None) # doctest: +SKIP
    MiniRocketMultivariateVariable(...)
    >>> X_transformed = pre_clf.transform(X_test) # doctest: +SKIP
    >>> X_transformed.shape # doctest: +SKIP
    (370, 9996)

    Raises
    ------
    ValueError
        If any multivariate series_length in X is < 9 and
        pad_value_short_series is set to None

    See Also
    --------
    MultiRocket, MiniRocket, MiniRocketMultivariate, Rocket

    References
    ----------
    .. [1] Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
        MINIROCKET: A Very Fast (Almost) Deterministic Transform for
        Time Series Classification, 2020, arXiv:2012.08791

    .. [2] Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
           https://github.com/angus924/minirocket
    """

    _tags = {
        "authors": ["angus924", "michaelfeil"],
        "maintainers": ["angus924", "michaelfeil"],
        "univariate-only": False,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "capability:unequal_length": True,
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "df-list",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "requires_y": False,
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        num_kernels=10_000,
        max_dilations_per_kernel=32,
        reference_length="max",
        pad_value_short_series=None,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.reference_length = reference_length
        self._fitted_reference_length = None
        self.pad_value_short_series = pad_value_short_series
        self.num_kernels_ = None
        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is None:
            self.random_state_ = random_state
        elif isinstance(random_state, int):
            self.random_state_ = np.int32(random_state)
        else:
            raise ValueError(
                f"random_state in MiniRocketMultivariateVariable must be int or None, "
                f"but found <{type(random_state)} {random_state}>"
            )

        self._reference_modes = ["max", "mean", "median", "min"]
        if not (isinstance(reference_length, int) and reference_length >= 9) and not (
            isinstance(reference_length, str)
            and (reference_length in self._reference_modes)
        ):
            raise ValueError(
                "reference_length in MiniRocketMultivariateVariable must be int>=9 or "
                "'max', 'mean', 'median', but found reference_length="
                f"{reference_length}"
            )

        super().__init__()

    def _fit(self, X: list[pd.DataFrame], y=None):
        """Fits dilations and biases to input time series.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with n_instances-rows and n_dimensions-columns,
            each cell containing a series_length-long array.
            n_dimensions is equal across all instances in ``X``, and
            series_length is constant within each instance.
        y : ignored argument for interface compatibility

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If any multivariate series_length in X is < 9 and
            pad_value_short_series is set to None
        """
        from sktime.transformations.panel.rocket._minirocket_multi_var_numba import (
            _fit_multi_var,
        )

        X_2d_t, lengths_1darray = _nested_dataframe_to_transposed2D_array_and_len_list(
            X, pad=self.pad_value_short_series
        )

        if isinstance(self.reference_length, int):
            _reference_length = self.reference_length
        elif self.reference_length in self._reference_modes:
            # np.mean, np.max, np.median, np.min ..
            _reference_length = getattr(np, self.reference_length)(lengths_1darray)
        else:
            raise ValueError(
                "reference_length in MiniRocketMultivariateVariable must be int>=9 or "
                "'max', 'mean', 'median', but found reference_length="
                f"{self.reference_length}"
            )
        self._fitted_reference_length = int(max(9, _reference_length))

        if lengths_1darray.min() < 9:
            failed_index = np.where(lengths_1darray < 9)[0]
            raise ValueError(
                f"X must be >= 9 for all samples, but found minimum to be "
                f"{lengths_1darray.min()}; at index {failed_index}, pad shorter "
                "series so that n_timepoints >= 9 for all samples."
            )

        if lengths_1darray.min() == lengths_1darray.max():
            warnings.warn(
                "X is of equal length, consider using MiniRocketMultivariate for "
                "speedup and stability instead.",
                stacklevel=2,
            )
        if X_2d_t.shape[0] == 1:
            warnings.warn(
                "X is univariate, consider using MiniRocket as Univariante for "
                "speedup and stability instead.",
                stacklevel=2,
            )

        self.parameters = _fit_multi_var(
            X_2d_t,
            L=lengths_1darray,
            reference_length=self._fitted_reference_length,
            num_features=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            seed=self.random_state_,
        )
        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        return self

    def _transform(self, X, y=None):
        """Transform input time series.

        Parameters
        ----------
        X : pd.DataFrame with nested columns
            Dataframe with n_instances-rows and n_dimensions-columns,
            each cell containing a series_length-long array

        y : ignored argument for interface compatibility

        Returns
        -------
            pandas.DataFrame, size (n_instances, num_kernels)

        Raises
        ------
        ValueError
            If any multivariate series_length in X is < 9 and
            pad_value_short_series is set to None
        """
        from numba import get_num_threads, set_num_threads

        from sktime.transformations.panel.rocket._minirocket_multi_var_numba import (
            _transform_multi_var,
        )

        X_2d_t, L = _nested_dataframe_to_transposed2D_array_and_len_list(
            X, pad=self.pad_value_short_series
        )
        # change n_jobs depended on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        X_ = _transform_multi_var(X_2d_t, L, self.parameters)
        set_num_threads(prev_threads)
        return pd.DataFrame(X_)


def _nested_dataframe_to_transposed2D_array_and_len_list(
    X: list[pd.DataFrame], pad: Union[int, float, None] = 0
):
    """Convert a nested dataframe to a 2D array and a list of lengths.

    Parameters
    ----------
    X : List of dataframes
        List of length n_instances, with
        dataframes of series_length-rows and n_dimensions-columns
    pad : float or None. if float/int,pads multivariate series with 'pad',
        so that each series has at least length 9.
        if None, no padding is applied.

    Returns
    -------
    np.array: 2D array of shape =
        [n_dimensions, sum(length_series(i) for i in n_instances)],
        np.float32
    np.array: 1D array of shape = [n_instances]
        with length of each series, np.int32

    Raises
    ------
    ValueError
        If any multivariate series_length in X is < 9 and
        pad_value_short_series is set to None
    """
    if not len(X):
        raise ValueError("X is empty")

    if isinstance(X, (tuple, list)) and isinstance(X[0], (pd.DataFrame, np.array)):
        pass
    else:
        raise ValueError("X must be List of pd.DataFrame")

    if not all(X[0].shape[1] == _x.shape[1] for _x in X):
        raise ValueError(
            "X must be nested pd.DataFrame or List of pd.DataFrame with n_dimensions"
        )

    vec = []
    lengths = []

    for _x in X:
        _x_shape = _x.shape
        if _x_shape[0] < 9:
            if pad is not None:
                # emergency: pad with zeros up to 9.
                lengths.append(9)
                vec.append(
                    np.vstack(
                        [_x.values, np.full([9 - _x_shape[0], _x_shape[1]], float(pad))]
                    )
                )
            else:
                raise ValueError(
                    "X series_length must be >= 9 for all samples"
                    f"but sample with series_length {_x_shape[0]} found. Consider"
                    " padding, discard, or setting a pad_value_short_series value"
                )
        else:
            lengths.append(_x_shape[0])
            vec.append(_x.values)

    X_2d_t = np.vstack(vec).T.astype(dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)

    if not lengths.sum() == X_2d_t.shape[1]:
        raise ValueError("X_new and lengths do not match. check input dimension")

    return X_2d_t, lengths

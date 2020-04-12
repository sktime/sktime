# Script that includes preliminary re-implementation of some existing
# Transformers in sktime using an ExtensionArray data container

import numpy as np
import pandas as pd

from sktime.transformers.segment import IntervalSegmenter, check_is_fitted
from sktime.transformers.base import BaseTransformer
from sklearn.utils import check_random_state


class RandomIntervalSegmenter(IntervalSegmenter):
    def __init__(self, n_intervals='sqrt', min_length=2, random_state=None):
        if not isinstance(min_length, int):
            raise ValueError(f"Min_lenght must be an integer, but found: "
                             f"{type(min_length)}")
        if min_length < 1:
            raise ValueError(f"Min_lenght must be an positive integer (>= 1), "
                             f"but found: {min_length}")

        self.min_length = min_length
        self.n_intervals = n_intervals
        self.random_state = random_state

        super(RandomIntervalSegmenter, self).__init__()

    def fit(self, X, y=None):

        col = X.columns[0]
        X = X[col]
        self.input_shape_ = X.shape

        if not X.has_common_index:
            raise ValueError("All time series in transform column {} must share a common time index".format(col))
        self._time_index = X.time_index

        if self.n_intervals == 'random':
            self.intervals_ = self._rand_intervals_rand_n(self._time_index)
        else:
            self.intervals_ = self._rand_intervals_fixed_n(self._time_index, n_intervals=self.n_intervals)

        return self

    def transform(self, X, y=None):
        colname = X.columns[0]
        X = X[colname]

        # Check inputs.
        check_is_fitted(self, 'intervals_')

        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns of input is different from what was seen'
                             'in `fit`')

        slices = [X.slice_time(np.arange(start=a, stop=b)).to_frame() for (a, b) in self.intervals_]

        for s, i in zip(slices, self.intervals_):
            # TODO: make sure there are no duplicate names
            s.rename(columns={colname: f"{colname}_{i[0]}_{i[1]}"}, inplace=True)

        return pd.concat(slices, axis=1)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state
        self._rng = check_random_state(random_state)

    def _rand_intervals_rand_n(self, x):
        starts = []
        ends = []
        m = x.shape[0]  # series length
        W = self._rng.randint(1, m, size=int(np.sqrt(m)))
        for w in W:
            size = m - w + 1
            start = self._rng.randint(size, size=int(np.sqrt(size)))
            starts.extend(start)
            for s in start:
                end = s + w
                ends.append(end)
        return np.column_stack([starts, ends])

    def _rand_intervals_fixed_n(self, x, n_intervals):
        len_series = len(x)
        # compute number of random intervals relative to series length (m)
        # TODO use smarter dispatch at construction to avoid evaluating if-statements here each time function is called
        if np.issubdtype(type(n_intervals), np.integer) and (n_intervals >= 1):
            pass
        elif n_intervals == 'sqrt':
            n_intervals = int(np.sqrt(len_series))
        elif n_intervals == 'log':
            n_intervals = int(np.log(len_series))
        elif np.issubdtype(type(n_intervals), np.floating) and (n_intervals > 0) and (n_intervals <= 1):
            n_intervals = int(len_series * n_intervals)
        else:
            raise ValueError(f'Number of intervals must be either "random", "sqrt", a positive integer, or a float '
                             f'value between 0 and 1, but found {n_intervals}.')

        # make sure there is at least one interval
        n_intervals = np.maximum(1, n_intervals)

        starts = self._rng.randint(len_series - self.min_length + 1, size=n_intervals)
        if n_intervals == 1:
            starts = [starts]  # make it an iterable

        ends = [start + self._rng.randint(self.min_length, len_series - start + 1) for start in starts]
        return np.column_stack([starts, ends])


class UniversalFunctionTransformer(BaseTransformer):
    """A convenience wrapper that applies a Universal Function (ufunc) defined as an instance method to an Awkward array.
    """
    def __init__(self, u_func: str or np.ufunc):
        """
        Parameters
        ----------
        u_func : str or np.ufunc
            The name of the ufunc to apply if the function is an instance method, or the function defined in Numpy to apply.
        """
        if not isinstance(u_func, str) and not isinstance(u_func, np.ufunc):
            raise ValueError("u_func is not a str or a numpy.ufunc")
        else:
            self._numpy_method = True if isinstance(u_func, np.ufunc) else False
            self._u_func = u_func if isinstance(u_func, np.ufunc) else getattr(np.ndarray, u_func)
            self._is_fitted = False

    @property
    def u_func(self):
        """
        Returns
        -------
        str or np.ufunc
            The name of the ufunc to apply if the function is an instance method, or the function defined in Numpy to apply.
        """
        return self._u_func

    def fit(self, x, y=None):
        """Empty fit function that does nothing.
        Parameters
        ----------
        x : TimeFrame
            The training input samples.
        y : None
            None as it is transformer on x.
        Returns
        -------
        self : object
            Returns self.
        """
        self._is_fitted = True
        return self

    def transform(self, x, y=None):
        """Apply the Universal Function to x.
        Parameters
        ----------
        x : TimeFrame
            The training input samples.
        y : None
            None as it is transformer on x.
        Returns
        -------
        DataFrame
            The result of applying the Universal Function to x.
        """

        xt = {col:self._u_func(x[col].values.data, axis=1) for col in x.columns}
        return pd.DataFrame(data=xt)

    def inverse_transform(self, X, y=None):
        """Not implemented for this type of transformer.
        """
        raise NotImplementedError("Transformer does not have an inverse transform method")


class GenericFunctionTransformer(BaseTransformer):
    """A convenience wrapper that applies a provided function to a TimeFrame.
    """
    def __init__(self, func: callable, apply_to_container: bool):
        """
        Parameters
        ----------
        func : callable
            The function to apply.
        apply_to_container : bool
            True if the function should be applied to the array provided during training, False if it should be applied to the sub-arrays of this array.
        """
        if callable(func):
            self._func = func
            self._apply_to_container = apply_to_container
            self._is_fitted = False
        else:
            raise ValueError("func is not a callable")

    @property
    def func(self):
        """
        Returns
        -------
        callable
            The function to apply.
        """
        return self._func

    @property
    def apply_to_container(self):
        """
        Returns
        -------
        bool
            True if the function should be applied to the array provided during training, False if it should be applied to the sub-arrays of this array.
        """
        return self._apply_to_container

    def fit(self, x, y=None):
        """Empty fit function that does nothing.
        Parameters
        ----------
        x : TimeFrame
            The training input samples.
        y : None
            None as it is transformer on x.
        Returns
        -------
        self : object
            Returns self.
        """
        self._is_fitted = True
        return self

    def transform(self, x, y=None):
        """Apply the Universal Function to x.
        Parameters
        ----------
        x : TimeFrame
            The training input samples.
        y : None
            None as it is transformer on x.
        Returns
        -------
        pd.DataFrame
            The result of applying the function to x.
        """

        xt = {col: self._func(x[col]) for col in x.columns}
        return pd.DataFrame(data=xt)

    def inverse_transform(self, X, y=None):
        """Not implemented for this type of transformer.
        """
        raise NotImplementedError("Transformer does not have an inverse transform method")


def extarray_slope_func(ts):
    """Calculates the linear slope for a specified Time Series.
    Parameters
    ----------
    ts : TimeSeries
        The 2-D series whose sub-series that slope values should be returned for.
    Returns
    -------
    np.ndarray
        A 1-D array of slope values for ts.
    """
    x = ts.time_index
    y = ts.values.data

    x_demeaned = (x.values - np.mean(x))[None, :]
    y_demeaned = y - np.mean(y, axis=1)[:, None]
    cov = (x_demeaned * y_demeaned).sum(axis=1)
    return cov / (len(x) * np.var(x))



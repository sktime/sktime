__all__ = ["SlidingWindowSplitter", "ManualWindowSplitter"]
__author__ = "Markus Löning"

import numpy as np
import pandas as pd
from sktime.utils.validation import is_int
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_step_length
from sktime.utils.validation.forecasting import check_time_index
from sktime.utils.validation.forecasting import check_window_length

DEFAULT_STEP_LENGTH = 1
DEFAULT_WINDOW_LENGTH = 10
DEFAULT_FH = 1


class BaseTemporalCrossValidator:
    """Rolling window iterator that allows to split time series index into two windows,
    one containing observations used as feature data and one containing observations used as
    target data to be predicted. The target window has the length of the given forecasting horizon.

    Parameters
    ----------
    window_length : int
        Length of rolling window
    fh : array-like  or int, optional, (default=None)
        Single step ahead or array of steps ahead to forecast.
    """

    def __init__(self, fh=DEFAULT_FH, window_length=DEFAULT_WINDOW_LENGTH):
        # check input
        if not is_int(window_length) and (window_length < 1):
            raise ValueError(f"window_length must be a postive integer, but found: {type(window_length)}")

        # set during construction
        self._window_length = check_window_length(window_length)
        self._fh = check_fh(fh)
        self._n_splits = None

    def split(self, y):
        raise NotImplementedError()

    @property
    def fh(self):
        """Forecasting horizon"""
        return self._fh

    @property
    def window_length(self):
        """Window length"""
        return self._window_length

    def get_n_splits(self, y=None):
        raise NotImplementedError("abstract method")


class SlidingWindowSplitter(BaseTemporalCrossValidator):

    def __init__(self, fh=DEFAULT_FH, window_length=DEFAULT_WINDOW_LENGTH, step_length=DEFAULT_STEP_LENGTH):
        if not is_int(step_length) and (step_length < 1):
            raise ValueError(f"step_length must be an positive integer, but found: {type(step_length)}")

        self._step_length = check_step_length(step_length)
        super(SlidingWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    def split(self, y):
        """Split time series using sliding window cross-validation"""

        # check input
        if isinstance(y, pd.Series):
            y = y.index

        # check time index
        time_index = check_time_index(y)
        n_timepoints = len(time_index)

        # convert to zero-based integer index
        time_index = np.arange(n_timepoints)

        # compute parameters for splitting
        # end point is end of last window
        fh_max = self.fh.max()
        end = n_timepoints - fh_max + 1  #  non-inclusive end indexing

        # start point
        start = self.window_length

        # check if computed values are feasible with the provided index
        if self.window_length + fh_max > n_timepoints:
            raise ValueError(f"`window_length` + `max(fh)` must be before the end of the "
                             f"time index, but found: {self.window_length}+{fh_max} "
                             f"> {n_timepoints}.")

        # split into windows
        for split_point in range(start, end, self.step_length):
            training_window = time_index[split_point - self.window_length:split_point]
            test_window = time_index[split_point + self.fh - 1]
            yield training_window, test_window

    def get_n_splits(self, y=None):
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires `y` to compute the number of splits.")

        if isinstance(y, pd.Series):
            y = y.index

        # check time index
        time_index = check_time_index(y)
        n_timepoints = len(time_index)

        # compute parameters for splitting
        # end point is end of last window
        fh_max = self.fh.max()
        end = n_timepoints - fh_max + 1  #  non-inclusive end indexing
        # number of splits for given forecasting horizon, window length and step length
        return np.int(np.ceil((end - self.window_length) / self.step_length))

    @property
    def step_length(self):
        """Step length"""
        return self._step_length


class ManualWindowSplitter(BaseTemporalCrossValidator):

    def __init__(self, cutoffs, fh=DEFAULT_FH, window_length=DEFAULT_WINDOW_LENGTH):
        self.cutoffs = cutoffs
        super(ManualWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    def split(self, y):
        # check input
        if isinstance(y, pd.Series):
            y = y.index

        # check time index
        time_index = check_time_index(y)
        n_timepoints = len(time_index)

        # get parameters
        window_length = self._window_length
        fh = self._fh
        cutoffs = self.cutoffs

        self._n_splits = len(cutoffs)

        # check that all time points are in time index
        if not all(np.isin(cutoffs, time_index)):
            raise ValueError("`cutoff` points must be in time index.")

        if not all(np.isin(cutoffs - window_length + 1, time_index)):
            raise ValueError("Some windows would be outside of the time index; "
                             "please change `window length` or `time_points` ")

        # convert to zero-based integer index
        cutoffs = np.where(np.isin(time_index, cutoffs))[0]
        time_index = np.arange(n_timepoints)

        for time_point in cutoffs:
            training_window = time_index[time_point - window_length + 1:time_point + 1]
            test_window = time_index[time_point + fh]
            yield training_window, test_window

    def get_n_splits(self, y):
        return len(self.cutoffs)

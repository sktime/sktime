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
    def n_splits(self):
        """
        Return number of splits.
        """
        if self._n_splits is None:
            raise ValueError(f"`n_splits_` is only available after calling `split`. "
                             f"This is because it depends on the number of time points of the "
                             f"time series `y` which is passed to split.")
        return self._n_splits

    def get_n_splits(self):
        """
        Return number of splits.
        """
        return self.n_splits

    @property
    def fh(self):
        """Forecasting horizon"""
        return self._fh

    @property
    def window_length(self):
        """Window length"""
        return self._window_length


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
        first_split_point = self.window_length

        # number of splits for given forecasting horizon, window length and step length
        self._n_splits = np.int(np.ceil((end - self.window_length) / self.step_length))

        # check if computed values are feasible with the provided index
        if self.window_length + fh_max > n_timepoints:
            raise ValueError(f"`window_length` + `max(fh)` must be before the end of the "
                             f"time index, but found: {self.window_length}+{fh_max} "
                             f"> {n_timepoints}.")

        # split into windows
        for split_point in range(first_split_point, end, self.step_length):
            training_window = time_index[split_point - self.window_length:split_point]
            test_window = time_index[split_point + self.fh - 1]
            yield training_window, test_window

    @property
    def step_length(self):
        """Step length"""
        return self._step_length


class ManualWindowSplitter(BaseTemporalCrossValidator):

    def __init__(self, time_points, fh=DEFAULT_FH, window_length=DEFAULT_WINDOW_LENGTH):
        self.time_points = time_points
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
        time_points = self.time_points

        self._n_splits = len(time_points)

        # check that all time points are in time index
        if not all(np.isin(time_points, time_index)):
            raise ValueError("`time_points` not found in time index.")

        if not all(np.isin(time_points - window_length + 1, time_index)):
            raise ValueError("Some windows would be outside of the time index; "
                             "please change `window length` or `time_points` ")

        # convert to zero-based integer index
        time_points = np.where(np.isin(time_index, time_points))[0]
        time_index = np.arange(n_timepoints)

        for time_point in time_points:
            training_window = time_index[time_point - window_length + 1:time_point + 1]
            test_window = time_index[time_point + fh]
            yield training_window, test_window

__all__ = ["SlidingWindowSplitter"]
__author__ = "Markus Löning"

import numpy as np
import pandas as pd
from sktime.utils.validation import is_int
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_step_length
from sktime.utils.validation.forecasting import check_time_index
from sktime.utils.validation.forecasting import check_window_length
from sktime.utils.validation.forecasting import check_y


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
    step_length : int, optional (default=1)
        Step length
    """

    def __init__(self, fh=1, window_length=10, step_length=1):
        self._window_length = check_window_length(window_length)
        self._step_length = check_step_length(step_length)
        self._fh = check_fh(fh)
        self._n_splits = None

    def split(self, y):
        raise NotImplementedError("abstract method")

    @property
    def n_splits(self):
        """
        Return number of splits.
        """
        return self._n_splits

    def get_n_splits(self, y):
        """
        Return number of splits.
        """
        raise NotImplementedError("abstract method")

    @property
    def fh(self):
        """Forecasting horizon"""
        return self._fh

    @property
    def window_length(self):
        """Window length"""
        return self._window_length

    @property
    def step_length(self):
        """Step length"""
        return self._step_length


class SlidingWindowSplitter(BaseTemporalCrossValidator):

    def split(self, y):
        """Split time series using sliding window cross-validation"""

        # check input
        if isinstance(y, pd.Series):
            y = check_y(y)
            y = y.index

        # check time index
        check_time_index(y)

        # convert to numeric indices
        n_timepoints = len(y)
        y = np.arange(n_timepoints)

        # compute parameters for splitting
        # end point is end of last window
        fh_max = self.fh.max()
        end = n_timepoints - fh_max + 1

        # start point
        start = self.window_length

        # check if computed values are feasible with the provided index
        if self.window_length + fh_max > n_timepoints:
            raise ValueError(f"`window_length` + `max(fh)` must be smaller or equal to "
                             f"`len(y)`, but found: {self.window_length}+{fh_max} "
                             f"> {n_timepoints}.")

        # split into windows
        for window in range(start, end, self.step_length):
            training_window = y[window - self.window_length:window]
            test_window = y[window + self.fh - 1]
            yield training_window, test_window

    def get_n_splits(self, y):
        """Get number of splits"""
        y = check_y(y)
        n_timepoints = len(y)
        fh_max = self.fh.max()
        end = n_timepoints - fh_max + 1

        # number of splits for given forecasting horizon, window length and step length
        self._n_splits = np.int(np.ceil((end - self.window_length) / self.step_length))
        return self._n_splits


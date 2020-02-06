__all__ = ["SlidingWindowSplitter"]
__author__ = "Markus Löning"

import numpy as np

from sktime.utils.validation.forecasting import validate_fh
from sktime.utils.validation.forecasting import validate_time_index
from sktime.utils.validation import is_int


class _BaseTemporalCrossValidator:
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
        # check input
        if not is_int(window_length) and (window_length < 1):
            raise ValueError(f"window_length must be a postive integer, but found: {type(window_length)}")

        if not is_int(step_length) and (step_length < 1):
            raise ValueError(f"step_length must be an positive integer, but found: {type(step_length)}")

        # set during construction
        self.window_length = window_length
        self.step_length = step_length
        self.fh = fh
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

    @fh.setter
    def fh(self, fh):
        """Validate forecasting horizon before setting it"""
        self._fh = validate_fh(fh)


class SlidingWindowSplitter(_BaseTemporalCrossValidator):

    def split(self, y, X=None):
        """Split time series using sliding window cross-validation"""

        if X is not None:
            raise NotImplementedError()

        # check input
        time_index = validate_time_index(y)

        n_timepoints = len(time_index)
        fh_max = self.fh[-1]  # furthest step ahead, assume fh is sorted
        last_window_end = n_timepoints - fh_max + 1

        # compute number of splits for given forecasting horizon, window length and step length
        self._n_splits = np.int(np.ceil((last_window_end - self.window_length) / self.step_length))

        # check if computed values are feasible given n_timepoints
        if self.window_length + fh_max > n_timepoints:
            raise ValueError(f"`window_length` + `max(fh)` must be smaller than "
                             f"the number of time points in `y`, but found: "
                             f"{self.window_length} + {fh_max} > {n_timepoints}")

        # iterate over windows
        start = self.window_length
        for window in range(start, last_window_end, self.step_length):
            in_window = time_index[window - self.window_length:window]
            out_window = time_index[window + self.fh - 1]
            yield in_window, out_window

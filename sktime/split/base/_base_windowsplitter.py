#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class for window based time series splitters."""

__author__ = ["khrapovs", "mloning", "hazrulakmal"]

from typing import Optional

import numpy as np
import pandas as pd

from sktime.datatypes._utilities import get_index_for_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.split.base import BaseSplitter
from sktime.split.base._common import (
    ACCEPTED_Y_TYPES,
    FORECASTING_HORIZON_TYPES,
    SPLIT_ARRAY_TYPE,
    SPLIT_GENERATOR_TYPE,
    _check_fh,
    _check_inputs_for_compatibility,
    _get_end,
)
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    NON_FLOAT_WINDOW_LENGTH_TYPES,
    array_is_int,
    check_window_length,
    is_int,
    is_timedelta_or_date_offset,
)
from sktime.utils.validation.forecasting import check_step_length


def _check_window_lengths(
    y: pd.Index,
    fh: ForecastingHorizon,
    window_length: NON_FLOAT_WINDOW_LENGTH_TYPES,
    initial_window: NON_FLOAT_WINDOW_LENGTH_TYPES,
) -> None:
    """Check that combination of inputs is compatible.

    Parameters
    ----------
    y : pd.Index
        Index of time series
    fh : int, timedelta, list or np.ndarray of ints or timedeltas
    window_length : int or timedelta or pd.DateOffset
    initial_window : int or timedelta or pd.DateOffset
        Window length of first window

    Raises
    ------
    ValueError
        if window length plus max horizon is above the last observation in `y`,
        or if initial window plus max horizon is above the last observation in `y`
    TypeError
        if type of the input is not supported
    """
    n_timepoints = y.shape[0]
    fh_max = fh[-1]

    error_msg_for_incompatible_window_length = (
        f"The `window_length` and the forecasting horizon are incompatible "
        f"with the length of `y`. Found `window_length`={window_length}, "
        f"`max(fh)`={fh_max}, but len(y)={n_timepoints}. "
        f"It is required that the window length plus maximum forecast horizon "
        f"is smaller than the length of the time series `y` itself."
    )
    if is_timedelta_or_date_offset(x=window_length):
        if y[0] + window_length + fh_max > y[-1]:
            raise ValueError(error_msg_for_incompatible_window_length)
    else:
        if window_length + fh_max > n_timepoints:
            raise ValueError(error_msg_for_incompatible_window_length)

    error_msg_for_incompatible_initial_window = (
        f"The `initial_window` and the forecasting horizon are incompatible "
        f"with the length of `y`. Found `initial_window`={initial_window}, "
        f"`max(fh)`={fh_max}, but len(y)={n_timepoints}. "
        f"It is required that the initial window plus maximum forecast horizon "
        f"is smaller than the length of the time series `y` itself."
    )
    error_msg_for_incompatible_types = (
        "The `initial_window` and `window_length` types are incompatible. "
        "They should be either all timedelta or all int."
    )
    if initial_window is not None:
        if is_timedelta_or_date_offset(x=initial_window):
            if y[0] + initial_window + fh_max > y[-1]:
                raise ValueError(error_msg_for_incompatible_initial_window)
            if not is_timedelta_or_date_offset(x=window_length):
                raise TypeError(error_msg_for_incompatible_types)
        else:
            if initial_window + fh_max > n_timepoints:
                raise ValueError(error_msg_for_incompatible_initial_window)
            if is_timedelta_or_date_offset(x=window_length):
                raise TypeError(error_msg_for_incompatible_types)


class BaseWindowSplitter(BaseSplitter):
    """Base class for sliding and expanding window splitter."""

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES,
        start_with_window: bool,
    ) -> None:
        _check_inputs_for_compatibility(
            [fh, initial_window, window_length, step_length]
        )
        self.step_length = step_length
        self.start_with_window = start_with_window
        self.initial_window = initial_window
        super().__init__(fh=fh, window_length=window_length)

    @property
    def _initial_window(self):
        if hasattr(self, "initial_window"):
            return self.initial_window
        return None

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        window_length = check_window_length(
            window_length=self.window_length,
            n_timepoints=n_timepoints,
            name="window_length",
        )
        initial_window = check_window_length(
            window_length=self._initial_window,
            n_timepoints=n_timepoints,
            name="initial_window",
        )
        fh = _check_fh(self.fh)
        _check_window_lengths(
            y=y, fh=fh, window_length=window_length, initial_window=initial_window
        )

        if self._initial_window is not None:
            yield self._split_for_initial_window(y)

        yield from self._split_windows(window_length=window_length, y=y, fh=fh)

    def _split_for_initial_window(self, y: pd.Index) -> SPLIT_ARRAY_TYPE:
        """Get train/test splits for non-empty initial window.

        Parameters
        ----------
        y : pd.Index
            Index of the time series to split

        Returns
        -------
        (np.ndarray, np.ndarray)
            Integer indices of the train/test windows
        """
        fh = _check_fh(self.fh)
        if not self.start_with_window:
            raise ValueError(
                "`start_with_window` must be True if `initial_window` is given"
            )
        if self._initial_window <= self.window_length:
            raise ValueError("`initial_window` must greater than `window_length`")
        if is_int(x=self._initial_window):
            end = self._initial_window
        else:
            end = y.get_loc(y[0] + self._initial_window)
        train = self._get_train_window(y=y, train_start=0, split_point=end)
        if array_is_int(fh):
            test = end + fh.to_numpy() - 1
        else:
            test = np.argwhere(y.isin(y[end - 1] + fh)).flatten()
        return train, test

    def _split_windows(
        self,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        y: pd.Index,
        fh: ForecastingHorizon,
    ) -> SPLIT_GENERATOR_TYPE:
        """Abstract method for sliding/expanding windows."""
        raise NotImplementedError("abstract method")

    def _split_windows_generic(
        self,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        y: pd.Index,
        fh: ForecastingHorizon,
        expanding: bool,
    ) -> SPLIT_GENERATOR_TYPE:
        """Split `y` into training and test windows.

        This function encapsulates common functionality
        shared by concrete implementations of this abstract class.

        Parameters
        ----------
        window_length : int or timedelta or pd.DateOffset
            Length of training window
        y : pd.Index
            Index of time series to split
        fh : ForecastingHorizon
            Single step ahead or array of steps ahead to forecast.
        expanding : bool
            Expanding (True) or sliding window (False) splitter

        Yields
        ------
        train : 1D np.ndarray of int
            Training window iloc indices, in reference to y
        test : 1D np.ndarray of int
            Test window iloc indices, in reference to y
        """
        start = self._get_start(y=y, fh=fh)
        split_points = self.get_cutoffs(pd.Series(index=y, dtype=float)) + 1
        split_points = (
            split_points if self._initial_window is None else split_points[1:]
        )
        for split_point in split_points:
            train_start = self._get_train_start(
                start=start if expanding else split_point,
                window_length=window_length,
                y=y,
            )
            train = self._get_train_window(
                y=y, train_start=train_start, split_point=split_point
            )
            if array_is_int(fh):
                test = split_point + fh.to_numpy() - 1
            else:
                test = np.argwhere(
                    y.isin(y[max(0, split_point - 1)] + fh.to_pandas())
                ).flatten()
                if split_point == 0:
                    test -= 1
            yield train, test

    @staticmethod
    def _get_train_start(
        start: int, window_length: ACCEPTED_WINDOW_LENGTH_TYPES, y: pd.Index
    ) -> int:
        if is_timedelta_or_date_offset(x=window_length):
            train_start = y.get_loc(
                max(y[min(start, len(y) - 1)] - window_length, min(y))
            )
            if start >= len(y):
                train_start += 1
        else:
            train_start = start - window_length
        return train_start

    def _get_start(self, y: pd.Index, fh: ForecastingHorizon) -> int:
        """Get the first split point."""
        # By default, the first split point is the index zero, the first
        # observation in
        # the data.
        start = 0

        # If we start with a full window, the first split point depends on the window
        # length.
        if hasattr(self, "start_with_window") and self.start_with_window:
            if self._initial_window not in [None, 0]:
                if is_timedelta_or_date_offset(x=self._initial_window):
                    start = y.get_loc(
                        y[start] + self._initial_window + self.step_length
                    )
                else:
                    start += self._initial_window + self.step_length
            else:
                if is_timedelta_or_date_offset(x=self.window_length):
                    start = y.get_loc(y[start] + self.window_length)
                else:
                    start += self.window_length

        # For in-sample forecasting horizons, the first split must ensure that
        # in-sample test set is still within the data.
        if not fh.is_all_out_of_sample():
            fh_min = abs(fh[0])
            if is_int(fh_min):
                start = fh_min + 1 if fh_min >= start else start
            else:
                shifted_y0 = y[0] + fh_min
                start = np.argmin(y <= shifted_y0) if shifted_y0 >= y[start] else start
        return start

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        from sktime.datatypes import check_is_scitype, convert

        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the "
                f"number of splits."
            )

        multi_scitypes = ["Hierarchical", "Panel"]
        is_non_single, _, metadata = check_is_scitype(y, multi_scitypes, [])

        # n_splits based on the first instance of the lowest level series cutoffs
        if is_non_single:
            from_mtype = metadata.get("mtype")
            scitype = metadata.get("scitype")
            if scitype == "Panel":
                to_mtype = "pd-multiindex"
            else:
                to_mtype = "pd_multiindex_hier"

            y = convert(y, from_type=from_mtype, to_type=to_mtype, as_scitype=scitype)

            index = self._coerce_to_index(y)
            for _, values in y.groupby(index.droplevel(-1)):
                # convert to a single ts
                instance_series = values.reset_index().iloc[:, -2:]
                instance_series.set_index(instance_series.columns[0], inplace=True)
                n_splits = len(self.get_cutoffs(instance_series))
                break
        else:
            n_splits = len(self.get_cutoffs(y))
        return n_splits

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        y = get_index_for_series(y)
        fh = _check_fh(self.fh)
        step_length = check_step_length(self.step_length)

        if self._initial_window is None:
            start = self._get_start(y=y, fh=fh)
        elif is_int(x=self._initial_window):
            start = self._initial_window
        else:
            start = y.get_loc(y[0] + self._initial_window)

        end = _get_end(y_index=y, fh=fh) + 2
        if is_int(x=step_length):
            return np.arange(start, end, step_length) - 1
        else:
            offset = step_length if start == 0 else pd.Timedelta(0)
            start_date = y[y < y[start] + offset][-1]
            end_date = y[end - 1] - step_length if end <= len(y) else y[-1]
            date_cutoffs = pd.date_range(
                start=start_date, end=end_date, freq=step_length
            )
            cutoffs = np.argwhere(y.isin(date_cutoffs)).flatten()
            if start <= 0:
                cutoffs = np.hstack((-1, cutoffs))
            return cutoffs

# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ForecastingHorizon"]

import numpy as np
import pandas as pd

from sktime.utils.validation.series import VALID_INDEX_TYPES
from functools import lru_cache

from sktime.utils.validation.series import check_time_index

RELATIVE_TYPES = (pd.Int64Index, pd.RangeIndex)
ABSOLUTE_TYPES = (pd.Int64Index, pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex)
assert set(RELATIVE_TYPES).issubset(VALID_INDEX_TYPES)
assert set(ABSOLUTE_TYPES).issubset(VALID_INDEX_TYPES)

DELEGATED_METHODS = (
    "__sub__",
    "__add__",
    "__mul__",
    "__div__",
    "__divmod__",
    "__pow__",
    "__gt__",
    "__ge__",
    "__ne__",
    "__lt__",
    "__eq__",
    "__le__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rdiv__",
    "__rmod__",
    "__rdivmod__",
    "__rpow__",
    "__getitem__",
    "__len__",
)


def _delegator(method):
    """Helper function to automatically decorate ForecastingHorizon class with
    methods from pandas.Index and delegate method calls to wrapped pandas.Index
    object."""

    def delegated(obj, *args, **kwargs):
        return getattr(obj.to_pandas(), method)(*args, **kwargs)

    return delegated


def _check_values(values):
    """Validate forecasting horizon values and converts them to supported
    pandas.Index types if possible.

    Parameters
    ----------
    values : int, list, array, certain pd.Index types
        Forecasting horizon with steps ahead to predict.

    Raises
    ------
    TypeError : if values type is not supported

    Returns
    -------
    values : pd.Index
        Sorted and validated forecasting horizon values.
    """
    # if values are one of the supported pandas index types, we don't have
    # to do
    # anything as the forecasting horizon directly wraps the index, note that
    # isinstance() does not work here, because index types inherit from each
    # other,
    # hence we check for type equality here
    if type(values) in VALID_INDEX_TYPES:
        pass

    # convert single integer to pandas index, no further checks needed
    elif isinstance(values, (int, np.integer)):
        return pd.Int64Index([values], dtype=np.int)

    # convert np.array or list to pandas index
    elif isinstance(values, (list, np.ndarray)):
        values = pd.Int64Index(values, dtype=np.int)

    # otherwise, raise type error
    else:
        valid_types = (
            "int",
            "np.array",
            "list",
            *[f"pd.{index_type.__name__}" for index_type in VALID_INDEX_TYPES],
        )
        raise TypeError(
            f"`values` type not supported. `values` must be one of"
            f" {valid_types}, but found: {type(values)}"
        )

    # check values does not contain duplicates
    if len(values) != values.nunique():
        raise ValueError("`values` must not contain duplicates.")

    # return sorted values
    return values.sort_values()


class ForecastingHorizon:
    """Forecasting horizon

    Parameters
    ----------
    values : pd.Index, np.array, list or int
        Values of forecasting horizon
    is_relative : bool, optional (default=True)
        - If True, values are relative to end of training series.
        - If False, values are absolute.
    """

    def __new__(cls, values=None, is_relative=True):
        # We want the ForecastingHorizon class to be an extension of the
        # pandas index, but since subclassing pandas indices is not
        # straightforward, we wrap the index object instead. In order to
        # still support the basic methods of a pandas index, we dynamically
        # add some basic methods and delegate the method calls to the wrapped
        # index object.
        for method in DELEGATED_METHODS:
            setattr(cls, method, _delegator(method))
        return object.__new__(cls)

    def __init__(self, values=None, is_relative=True):
        if not isinstance(is_relative, bool):
            raise TypeError("`is_relative` must be a boolean")
        values = _check_values(values)

        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we check for type equality
        error_msg = (
            f"`values` type is not compatible with `is_relative=" f"{is_relative}`."
        )
        if is_relative:
            if not type(values) in RELATIVE_TYPES:
                raise TypeError(error_msg)
        else:
            if not type(values) in ABSOLUTE_TYPES:
                raise TypeError(error_msg)

        self._values = values
        self._is_relative = is_relative

    def _new(self, values=None, is_relative=None):
        """Construct new ForecastingHorizon based on current object"""
        if values is None:
            values = self._values
        if is_relative is None:
            is_relative = self.is_relative
        return type(self)(values, is_relative)

    @property
    def is_relative(self):
        """Whether forecasting horizon is relative to the end of the
        training series.

        Returns
        -------
        is_relative : bool
        """
        return self._is_relative

    def to_pandas(self):
        """Returns underlying values as pd.Index

        Returns
        -------
        fh : pd.Index
        """
        return self._values

    def to_numpy(self, **kwargs):
        """Returns underlying values as np.array

        Parameters
        ----------
        **kwargs : dict of kwargs
            kwargs passed to `to_numpy()` of wrapped pandas index.

        Returns
        -------
        fh : np.ndarray
        """
        return self.to_pandas().to_numpy(**kwargs)

    # we cache the results from `to_relative()` and `to_absolute()` calls to speed up
    # computations, as these are the basic methods and often required internally when
    # calling different methods
    @lru_cache(typed=True)
    def to_relative(self, cutoff=None):
        """Return relative values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            Relative representation of forecasting horizon
        """
        if self.is_relative:
            return self._new()

        else:
            self._check_cutoff(cutoff)
            values = self.to_pandas() - cutoff

            if isinstance(self.to_pandas(), (pd.PeriodIndex, pd.DatetimeIndex)):
                values = _coerce_duration_to_int(values, unit=_get_unit(cutoff))

            return self._new(values, is_relative=True)

    @lru_cache(typed=True)
    def to_absolute(self, cutoff=None):
        """Return absolute values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            Absolute representation of forecasting horizon
        """
        if not self.is_relative:
            return self._new()

        else:
            self._check_cutoff(cutoff)
            index = self.to_pandas()
            values = _shift(cutoff, by=index)
            return self._new(values, is_relative=False)

    def to_absolute_int(self, start, cutoff=None):
        """Return absolute values as zero-based integer index
        Parameters
        ----------
        start : pd.Period, pd.Timestamp, int
            Start value
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.
        Returns
        -------
        fh : ForecastingHorizon
            Absolute representation of forecasting horizon as zero-based
            integer index
        """
        self._check_cutoff(start)
        absolute = self.to_absolute(cutoff).to_pandas()
        values = absolute - start
        if isinstance(absolute, (pd.PeriodIndex, pd.DatetimeIndex)):
            values = _coerce_duration_to_int(values, unit=_get_unit(cutoff))
        return self._new(values, is_relative=False)

    def to_in_sample(self, cutoff=None):
        """Return in-sample values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            In-sample values of forecasting horizon
        """
        is_in_sample = self._is_in_sample(cutoff)
        in_sample = self.to_pandas()[is_in_sample]
        return self._new(in_sample)

    def to_out_of_sample(self, cutoff=None):
        """Return out-of-sample values
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        fh : ForecastingHorizon
            Out-of-sample values of forecasting horizon
        """
        is_out_of_sample = self._is_out_of_sample(cutoff)
        out_of_sample = self.to_pandas()[is_out_of_sample]
        return self._new(out_of_sample)

    def _is_in_sample(self, cutoff=None):
        """Get index location of in-sample values"""
        return self.to_relative(cutoff).to_pandas() <= 0

    def is_all_in_sample(self, cutoff=None):
        """Whether or not the forecasting horizon is purely in-sample for given
        cutoff.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        ret : bool
            True if the forecasting horizon is purely in-sample for given cutoff.
        """
        return sum(self._is_in_sample(cutoff)) == len(self)

    def _is_out_of_sample(self, cutoff=None):
        """Get index location of out-of-sample values"""
        # return ~self._in_sample_idx(cutoff)
        return self.to_relative(cutoff).to_pandas() > 0

    def is_all_out_of_sample(self, cutoff=None):
        """Whether or not the forecasting horizon is purely out-of-sample for
        given cutoff.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.

        Returns
        -------
        ret : bool
            True if the forecasting horizon is purely out-of-sample for given
            cutoff.
        """
        return sum(self._is_out_of_sample(cutoff)) == len(self)

    def to_indexer(self, cutoff=None, from_cutoff=True):
        """Return zero-based indexer values for easy indexing into arrays.
        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one and vice versa.
        from_cutoff : bool, optional (default=True)
            - If True, zero-based relative to cutoff.
            - If False, zero-based relative to first value in forecasting
            horizon.

        Returns
        -------
        fh : pd.Index
            Indexer
        """
        if from_cutoff:
            return self.to_relative(cutoff).to_pandas() - 1
        else:
            relative = self.to_relative(cutoff)
            return relative - relative.to_pandas()[0]

    def __repr__(self):
        # generate repr based on wrapped index repr
        class_name = self.__class__.__name__
        pandas_repr = repr(self.to_pandas()).split("(")[-1].strip(")")
        return f"{class_name}({pandas_repr}, is_relative={self.is_relative})"

    def _check_cutoff(self, cutoff):
        """Helper function to check fh type compatibility against cutoff"""
        if cutoff is None:
            raise ValueError("`cutoff` must be provided.")

        index = self.to_pandas()

        if isinstance(index, pd.PeriodIndex):
            assert isinstance(cutoff, pd.Period)

        if isinstance(index, pd.DatetimeIndex):
            assert isinstance(cutoff, pd.Timestamp)


def _coerce_duration_to_int(duration, unit=None):
    """Coerce durations into integer representations for a given unit of
    duration

    Parameters
    ----------
    duration : pd.DateOffset, pd.Timedelta, pd.TimedeltaIndex, pd.Index, int
        Duration type or collection of duration types
    unit : str
        Time unit

    Returns
    -------
    ret : int
        Duration in integer values for given unit
    """
    if isinstance(duration, pd.tseries.offsets.DateOffset):
        return duration.n
    elif isinstance(duration, pd.Index) and isinstance(
        duration[0], pd.tseries.offsets.BaseOffset
    ):
        return pd.Int64Index([d.n for d in duration])
    elif isinstance(duration, (pd.Timedelta, pd.TimedeltaIndex)):
        if unit is None:
            raise ValueError("`unit` missing")
        # integer conversion only works reliably with non-ambiguous units (
        # e.g. days, seconds but not months, years)
        try:
            if isinstance(duration, pd.Timedelta):
                return int(duration / pd.Timedelta(1, unit))
            if isinstance(duration, pd.TimedeltaIndex):
                return (duration / pd.Timedelta(1, unit)).astype(np.int)
        except ValueError:
            raise ValueError(
                "Index type not supported. Please consider using " "pd.PeriodIndex."
            )
    else:
        raise TypeError("`duration` type not understood.")


def _get_unit(x):
    """Get unit for conversion of time deltas to integers"""
    if hasattr(x, "freqstr"):
        return x.freqstr
    else:
        return None


def _shift(x, by=1):
    """Shift time point `x` by a step (`by`) given frequency of `x`

    Parameters
    ----------
    x : pd.Period, pd.Timestamp, int
        Time point
    by : int

    Returns
    -------
    ret : pd.Period, pd.Timestamp, int
        Shifted time point
    """
    assert isinstance(x, (pd.Period, pd.Timestamp, int, np.integer)), type(x)
    assert isinstance(by, (int, np.integer, pd.Int64Index)), type(by)
    if isinstance(x, pd.Timestamp):
        if not hasattr(x, "freq"):
            raise ValueError("No `freq` information available")
        by *= x.freq
    return x + by


def _get_duration(x, y=None, coerce_to_int=False, unit=None):
    """Compute duration of time index `x` or durations between time
    points `x` and `y` if `y` is given

    Parameters
    ----------
    x : pd.Index, pd.Timestamp, pd.Period, int
    y : pd.Timestamp, pd.Period, int, optional (default=None)
    coerce_to_int : bool
        If True, duration is returned as integer value for given unit
    unit : str
        Time unit

    Returns
    -------
    ret : duration type
        Duration
    """
    if y is None:
        x = check_time_index(x)
        duration = x[-1] - x[0]
    else:
        assert isinstance(x, (int, np.integer, pd.Period, pd.Timestamp))
        # check types allowing (np.integer, int) combinations to pass
        assert type(x) is type(y) or (
            isinstance(x, (np.integer, int)) and isinstance(x, (np.integer, int))
        )
        duration = x - y

    # coerce to integer result for given time unit
    if coerce_to_int and isinstance(
        x, (pd.PeriodIndex, pd.DatetimeIndex, pd.Period, pd.Timestamp)
    ):
        if unit is None:
            # try to get the unit from the data if not given
            unit = _get_unit(x)
        duration = _coerce_duration_to_int(duration, unit=unit)
    return duration

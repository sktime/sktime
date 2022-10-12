# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functionality for specifying forecast horizons in sktime."""

__author__ = ["mloning", "fkiraly", "eenticott-shell", "khrapovs"]
__all__ = ["ForecastingHorizon"]

from functools import lru_cache
from typing import Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.validation import (
    array_is_int,
    array_is_timedelta_or_date_offset,
    is_array,
    is_int,
    is_timedelta_or_date_offset,
)
from sktime.utils.validation.series import (
    VALID_INDEX_TYPES,
    is_in_valid_absolute_index_types,
    is_in_valid_index_types,
    is_in_valid_relative_index_types,
    is_integer_index,
)

VALID_FORECASTING_HORIZON_TYPES = (int, list, np.ndarray, pd.Index)

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
    "max",
    "min",
)


def _delegator(method):
    """Automatically decorate ForecastingHorizon class with pandas.Index methods.

    Also delegates method calls to wrapped pandas.Index object.
    methods from pandas.Index and delegate method calls to wrapped pandas.Index
    """

    def delegated(obj, *args, **kwargs):
        return getattr(obj.to_pandas(), method)(*args, **kwargs)

    return delegated


def _check_values(values: Union[VALID_FORECASTING_HORIZON_TYPES]) -> pd.Index:
    """Validate forecasting horizon values.

    Validation checks validity and also converts forecasting horizon values
    to supported pandas.Index types if possible.

    Parameters
    ----------
    values : int, list, array, certain pd.Index types
        Forecasting horizon with steps ahead to predict.

    Raises
    ------
    TypeError :
        Raised if `values` type is not supported

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
    if is_in_valid_index_types(values):
        pass

    # convert single integer or timedelta or dateoffset
    # to pandas index, no further checks needed
    elif is_int(values):
        values = pd.Index([values], dtype=int)

    elif is_timedelta_or_date_offset(values):
        values = pd.Index([values])

    # convert np.array or list to pandas index
    elif is_array(values) and array_is_int(values):
        values = pd.Index(values, dtype=int)

    elif is_array(values) and array_is_timedelta_or_date_offset(values):
        values = pd.Index(values)

    # otherwise, raise type error
    else:
        valid_types = (
            "int",
            "1D np.ndarray of type int",
            "1D np.ndarray of type timedelta or dateoffset",
            "list",
            *[f"pd.{index_type.__name__}" for index_type in VALID_INDEX_TYPES],
        )
        raise TypeError(
            f"Invalid `fh`. The type of the passed `fh` values is not supported. "
            f"Please use one of {valid_types}, but found type {type(values)}, "
            f"values = {values}"
        )

    # check values does not contain duplicates
    if len(values) != values.nunique():
        raise ValueError(
            "Invalid `fh`. The `fh` values must not contain any duplicates."
        )

    # return sorted values
    return values.sort_values()


def _check_freq(obj):
    """Coerce obj to a pandas frequency offset for the ForecastingHorizon.

    Parameters
    ----------
    obj : pd.Index, pd.Period, pandas offset, or None

    Returns
    -------
    pd offset

    Raises
    ------
    TypeError if the type assumption on obj is not met
    """
    if isinstance(obj, pd.offsets.BaseOffset):
        return obj
    elif hasattr(obj, "_cutoff"):
        return _check_freq(obj._cutoff)
    elif isinstance(obj, (pd.Period, pd.Index)):
        return _extract_freq_from_cutoff(obj)
    elif isinstance(obj, str) or obj is None:
        return to_offset(obj)
    else:
        return None


def _extract_freq_from_cutoff(x) -> Optional[str]:
    """Extract frequency string from cutoff.

    Parameters
    ----------
    x : pd.Period, pd.PeriodIndex, pd.DatetimeIndex

    Returns
    -------
    str : Frequency string or None
    """
    if isinstance(x, (pd.Period, pd.PeriodIndex, pd.DatetimeIndex)):
        return x.freq
    else:
        return None


class ForecastingHorizon:
    """Forecasting horizon.

    Parameters
    ----------
    values : pd.Index, pd.TimedeltaIndex, np.array, list, pd.Timedelta, or int
        Values of forecasting horizon
    is_relative : bool, optional (default=None)
        - If True, a relative ForecastingHorizon is created:
                values are relative to end of training series.
        - If False, an absolute ForecastingHorizon is created:
                values are absolute.
        - if None, the flag is determined automatically:
            relative, if values are of supported relative index type
            absolute, if not relative and values of supported absolute index type
    freq : str, pd.Index, pandas offset, or sktime forecaster, optional (default=None)
        object carrying frequency information on values
        ignored unless values is without inferrable freq

    Examples
    --------
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> import numpy as np
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=6)

        List as ForecastingHorizon
    >>> ForecastingHorizon([1, 2, 3])
    ForecastingHorizon([1, 2, 3], dtype='int64', is_relative=True)

        Numpy as ForecastingHorizon
    >>> ForecastingHorizon(np.arange(1, 7))
    ForecastingHorizon([1, 2, 3, 4, 5, 6], dtype='int64', is_relative=True)

        Absolute ForecastingHorizon with a pandas Index
    >>> ForecastingHorizon(y_test.index, is_relative=False) # doctest: +SKIP
    ForecastingHorizon(['1960-07', '1960-08', '1960-09', '1960-10',
        '1960-11', '1960-12'], dtype='period[M]', name='Period', is_relative=False)

        Converting
    >>> # set cutoff (last time point of training data)
    >>> cutoff = y_train.index[-1]
    >>> cutoff
    Period('1960-06', 'M')
    >>> # to_relative
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> fh.to_relative(cutoff=cutoff)
    ForecastingHorizon([1, 2, 3, 4, 5, 6], dtype='int64', is_relative=True)

    >>> # to_absolute
    >>> fh = ForecastingHorizon([1, 2, 3, 4, 5, 6], is_relative=True)
    >>> fh.to_absolute(cutoff=cutoff) # doctest: +SKIP
    ForecastingHorizon(['1960-07', '1960-08', '1960-09', '1960-10',
        '1960-11', '1960-12'], dtype='period[M]', is_relative=False)

        Automatically casted ForecastingHorizon from list when calling predict()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> forecaster.fit(y_train)
    NaiveForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    >>> forecaster.fh
    ForecastingHorizon([1, 2, 3], dtype='int64', is_relative=True)

        This is identical to give an object of ForecastingHorizon
    >>> y_pred = forecaster.predict(fh=ForecastingHorizon([1,2,3]))
    >>> forecaster.fh
    ForecastingHorizon([1, 2, 3], dtype='int64', is_relative=True)
    """

    def __new__(
        cls,
        values: Union[VALID_FORECASTING_HORIZON_TYPES] = None,
        is_relative: bool = None,
        freq=None,
    ):
        """Create a new ForecastingHorizon object."""
        # We want the ForecastingHorizon class to be an extension of the
        # pandas index, but since subclassing pandas indices is not
        # straightforward, we wrap the index object instead. In order to
        # still support the basic methods of a pandas index, we dynamically
        # add some basic methods and delegate the method calls to the wrapped
        # index object.
        for method in DELEGATED_METHODS:
            setattr(cls, method, _delegator(method))
        return object.__new__(cls)

    def __init__(
        self,
        values: Union[VALID_FORECASTING_HORIZON_TYPES] = None,
        is_relative: Optional[bool] = True,
        freq=None,
    ):
        # coercing inputs

        # values to pd.Index self._values
        values = _check_values(values)
        self._values = values

        # infer freq from values, if available
        # if not, infer from freq argument, if available
        if hasattr(values, "index") and hasattr(values.index, "freq"):
            self.freq = values.index.freq
        elif hasattr(values, "freq"):
            self.freq = values.freq
        self.freq = freq

        # infer self._is_relative from is_relative, and type of values
        # depending on type of values, is_relative is inferred
        # integers and timedeltas are interpreted as relative, by default, etc
        if is_relative is not None and not isinstance(is_relative, bool):
            raise TypeError("`is_relative` must be a boolean or None")
        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we check for type equality
        error_msg = f"`values` type is not compatible with `is_relative={is_relative}`."
        if is_relative is None:
            if is_in_valid_relative_index_types(values):
                is_relative = True
            elif is_in_valid_absolute_index_types(values):
                is_relative = False
            else:
                raise TypeError(f"{type(values)} is not a supported fh index type")
        if is_relative:
            if not is_in_valid_relative_index_types(values):
                raise TypeError(error_msg)
        else:
            if not is_in_valid_absolute_index_types(values):
                raise TypeError(error_msg)
        self._is_relative = is_relative

    def _new(
        self,
        values: Union[VALID_FORECASTING_HORIZON_TYPES] = None,
        is_relative: bool = None,
        freq: str = None,
    ):
        """Construct new ForecastingHorizon based on current object.

        Parameters
        ----------
        values : pd.Index, pd.TimedeltaIndex, np.array, list, pd.Timedelta, or int
            Values of forecasting horizon.
        is_relative : bool, default=same as self.is_relative
            - If None, determined automatically: same as self.is_relative
            - If True, values are relative to end of training series.
            - If False, values are absolute.
        freq : str, optional (default=None)
            Frequency string

        Returns
        -------
        ForecastingHorizon :
            New ForecastingHorizon based on current object
        """
        if values is None:
            values = self._values
        if is_relative is None:
            is_relative = self._is_relative
        if freq is None:
            freq = self._freq
        return type(self)(values=values, is_relative=is_relative, freq=freq)

    @property
    def is_relative(self) -> bool:
        """Whether forecasting horizon is relative to the end of the training series.

        Returns
        -------
        is_relative : bool
        """
        return self._is_relative

    @property
    def freq(self) -> str:
        """Frequency attribute.

        Returns
        -------
        freq : pandas frequency string
        """
        if hasattr(self, "_freq") and hasattr(self._freq, "freqstr"):
            # _freq is a pandas offset, frequency string is obtained via freqstr
            return self._freq.freqstr
        else:
            return None

    @freq.setter
    def freq(self, obj) -> None:
        """Frequency setter.

        Attempts to set/update frequency from obj.
        Sets self._freq to a pandas offset object (frequency representation).
        Frequency is extracted from obj, via _check_freq.
        Raises error if _freq is already set, and discrepant from frequency of obj.

        Parameters
        ----------
        obj : str, pd.Index, BaseForecaster, pandas offset
            object carrying frequency information on self.values

        Raises
        ------
        ValueError : if freq is already set and discrepant from frequency of obj
        """
        freq_from_obj = _check_freq(obj)
        if hasattr(self, "_freq"):
            freq_from_self = self._freq
        else:
            freq_from_self = None

        if freq_from_self is not None and freq_from_obj is not None:
            if freq_from_self != freq_from_obj:
                raise ValueError(
                    "Frequencies from two sources do not coincide: "
                    f"Current: {freq_from_self}, from update: {freq_from_obj}."
                )
        elif freq_from_obj is not None:  # only freq_from_obj is not None
            self._freq = freq_from_obj
        else:
            # leave self._freq as freq_from_self, or set to None if does not exist yet
            self._freq = freq_from_self

    def to_pandas(self) -> pd.Index:
        """Return forecasting horizon's underlying values as pd.Index.

        Returns
        -------
        fh : pd.Index
            pandas Index containing forecasting horizon's underlying values.
        """
        return self._values

    def to_numpy(self, **kwargs) -> np.ndarray:
        """Return forecasting horizon's underlying values as np.array.

        Parameters
        ----------
        **kwargs : dict of kwargs
            kwargs passed to `to_numpy()` of wrapped pandas index.

        Returns
        -------
        fh : np.ndarray
            NumPy array containg forecasting horizon's underlying values.
        """
        return self.to_pandas().to_numpy(**kwargs)

    def _coerce_cutoff_to_index_element(self, cutoff):
        """Coerces cutoff to index element, and updates self.freq with cutoff."""
        self.freq = cutoff
        if isinstance(cutoff, pd.Index):
            assert len(cutoff) > 0
            cutoff = cutoff[-1]
        return cutoff

    def to_relative(self, cutoff=None):
        """Return forecasting horizon values relative to a cutoff.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional (default=None)
            Cutoff value required to convert a relative forecasting
            horizon to an absolute one (and vice versa).
            If pd.Index, last/latest value is considered the cutoff

        Returns
        -------
        fh : ForecastingHorizon
            Relative representation of forecasting horizon.
        """
        cutoff = self._coerce_cutoff_to_index_element(cutoff)
        return _to_relative(fh=self, cutoff=cutoff)

    def to_absolute(self, cutoff):
        """Return absolute version of forecasting horizon values.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one (and vice versa).
            If pd.Index, last/latest value is considered the cutoff

        Returns
        -------
        fh : ForecastingHorizon
            Absolute representation of forecasting horizon.
        """
        cutoff = self._coerce_cutoff_to_index_element(cutoff)
        return _to_absolute(fh=self, cutoff=cutoff)

    def to_absolute_int(self, start, cutoff=None):
        """Return absolute values as zero-based integer index starting from `start`.

        Parameters
        ----------
        start : pd.Period, pd.Timestamp, int
            Start value returned as zero.
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional (default=None)
            Cutoff value required to convert a relative forecasting
            horizon to an absolute one (and vice versa).
            If pd.Index, last/latest value is considered the cutoff

        Returns
        -------
        fh : ForecastingHorizon
            Absolute representation of forecasting horizon as zero-based
            integer index.
        """
        cutoff = self._coerce_cutoff_to_index_element(cutoff)
        freq = self.freq

        if isinstance(cutoff, pd.Timestamp):
            # coerce to pd.Period for reliable arithmetic operations and
            # computations of time deltas
            cutoff = _coerce_to_period(cutoff, freq=freq)

        absolute = self.to_absolute(cutoff).to_pandas()
        if isinstance(absolute, pd.DatetimeIndex):
            # coerce to pd.Period for reliable arithmetics and computations of
            # time deltas
            absolute = _coerce_to_period(absolute, freq=freq)

        # We here check the start value, the cutoff value is checked when we use it
        # to convert the horizon to the absolute representation below
        if isinstance(start, pd.Timestamp):
            start = _coerce_to_period(start, freq=freq)
        _check_cutoff(start, absolute)

        # Note: We should here also coerce to periods for more reliable arithmetic
        # operations as in `to_relative` but currently doesn't work with
        # `update_predict` and incomplete time indices where the `freq` information
        # is lost, see comment on issue #534
        # The following line circumvents the bug in pandas
        # periods = pd.period_range(start="2021-01-01", periods=3, freq="2H")
        # periods - periods[0]
        # Out: Index([<0 * Hours>, <4 * Hours>, <8 * Hours>], dtype = 'object')
        # [v - periods[0] for v in periods]
        # Out: Index([<0 * Hours>, <2 * Hours>, <4 * Hours>], dtype='object')
        integers = pd.Index([date - start for date in absolute])

        if isinstance(absolute, (pd.PeriodIndex, pd.DatetimeIndex)):
            integers = _coerce_duration_to_int(integers, freq=freq)

        return self._new(integers, is_relative=False)

    def to_in_sample(self, cutoff=None):
        """Return in-sample index values of fh.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value required to convert a relative forecasting
            horizon to an absolute one (and vice versa).

        Returns
        -------
        fh : ForecastingHorizon
            In-sample values of forecasting horizon.
        """
        is_in_sample = self._is_in_sample(cutoff)
        in_sample = self.to_pandas()[is_in_sample]
        return self._new(in_sample)

    def to_out_of_sample(self, cutoff=None):
        """Return out-of-sample values of fh.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one (and vice versa).

        Returns
        -------
        fh : ForecastingHorizon
            Out-of-sample values of forecasting horizon.
        """
        is_out_of_sample = self._is_out_of_sample(cutoff)
        out_of_sample = self.to_pandas()[is_out_of_sample]
        return self._new(out_of_sample)

    def _is_in_sample(self, cutoff=None) -> np.ndarray:
        """Get index location of in-sample values."""
        relative = self.to_relative(cutoff).to_pandas()
        null = 0 if is_integer_index(relative) else pd.Timedelta(0)
        return relative <= null

    def is_all_in_sample(self, cutoff=None) -> bool:
        """Whether the forecasting horizon is purely in-sample for given cutoff.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, default=None
            Cutoff value used to check if forecasting horizon is purely in-sample.

        Returns
        -------
        ret : bool
            True if the forecasting horizon is purely in-sample for given cutoff.
        """
        return sum(self._is_in_sample(cutoff)) == len(self)

    def _is_out_of_sample(self, cutoff=None) -> np.ndarray:
        """Get index location of out-of-sample values."""
        return np.logical_not(self._is_in_sample(cutoff))

    def is_all_out_of_sample(self, cutoff=None) -> bool:
        """Whether the forecasting horizon is purely out-of-sample for given cutoff.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
            Cutoff value used to check if forecasting horizon is purely
            out-of-sample.

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
            Cutoff value required to convert a relative forecasting
            horizon to an absolute one and vice versa.
        from_cutoff : bool, optional (default=True)
            - If True, zero-based relative to cutoff.
            - If False, zero-based relative to first value in forecasting
            horizon.

        Returns
        -------
        fh : pd.Index
            Indexer.
        """
        if from_cutoff:
            relative_index = self.to_relative(cutoff).to_pandas()
            if is_integer_index(relative_index):
                return relative_index - 1
            else:
                # What does indexer mean if fh is timedelta?
                msg = (
                    "The indexer for timedelta-like forecasting horizon "
                    "is not yet implemented"
                )
                raise NotImplementedError(msg)
        else:
            relative = self.to_relative(cutoff)
            return relative - relative.to_pandas()[0]

    def __repr__(self):
        """Generate repr based on wrapped index repr."""
        class_name = self.__class__.__name__
        pandas_repr = repr(self.to_pandas()).split("(")[-1].strip(")")
        return f"{class_name}({pandas_repr}, is_relative={self.is_relative})"


# This function needs to be outside ForecastingHorizon
# since the lru_cache decorator has known, problematic interactions
# with object methods, see B019 error of flake8-bugbear for a detail explanation.
# See more here: https://github.com/sktime/sktime/issues/2338
# We cache the results from `to_relative()` and `to_absolute()` calls to speed up
# computations, as these are the basic methods and often required internally when
# calling different methods.
@lru_cache(typed=True)
def _to_relative(fh: ForecastingHorizon, cutoff=None) -> ForecastingHorizon:
    """Return forecasting horizon values relative to a cutoff.

    Parameters
    ----------
    fh : ForecastingHorizon
    cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
        Cutoff value required to convert a relative forecasting
        horizon to an absolute one (and vice versa).

    Returns
    -------
    fh : ForecastingHorizon
        Relative representation of forecasting horizon.
    """
    if fh.is_relative:
        return fh._new()

    else:
        absolute = fh.to_pandas()
        _check_cutoff(cutoff, absolute)

        if isinstance(absolute, pd.DatetimeIndex):
            # coerce to pd.Period for reliable arithmetics and computations of
            # time deltas
            absolute = _coerce_to_period(absolute, freq=fh.freq)
            cutoff = _coerce_to_period(cutoff, freq=fh.freq)

        # TODO: Replace when we upgrade our lower pandas bound
        #  to a version where this is fixed
        # Compute relative values
        # The following line circumvents the bug in pandas
        # periods = pd.period_range(start="2021-01-01", periods=3, freq="2H")
        # periods - periods[0]
        # Out: Index([<0 * Hours>, <4 * Hours>, <8 * Hours>], dtype = 'object')
        # [v - periods[0] for v in periods]
        # Out: Index([<0 * Hours>, <2 * Hours>, <4 * Hours>], dtype='object')
        # TODO: 0.14.0: Check if this comment below can be removed,
        # so check if pandas has released the fix to PyPI:
        # This bug was reported: https://github.com/pandas-dev/pandas/issues/45999
        # and fixed: https://github.com/pandas-dev/pandas/pull/46006
        # Most likely it will be released with pandas 1.5
        # Once the bug is fixed the line should simply be:
        # relative = absolute - cutoff
        relative = pd.Index([date - cutoff for date in absolute])

        # Coerce durations (time deltas) into integer values for given frequency
        if isinstance(absolute, (pd.PeriodIndex, pd.DatetimeIndex)):
            relative = _coerce_duration_to_int(relative, freq=fh.freq)

        return fh._new(relative, is_relative=True, freq=fh.freq)


# This function needs to be outside ForecastingHorizon
# since the lru_cache decorator has known, problematic interactions
# with object methods, see B019 error of flake8-bugbear for a detail explanation.
# See more here: https://github.com/sktime/sktime/issues/2338
@lru_cache(typed=True)
def _to_absolute(fh: ForecastingHorizon, cutoff) -> ForecastingHorizon:
    """Return absolute version of forecasting horizon values.

    Parameters
    ----------
    fh : ForecastingHorizon
    cutoff : pd.Period, pd.Timestamp, int
        Cutoff value is required to convert a relative forecasting
        horizon to an absolute one (and vice versa).

    Returns
    -------
    fh : ForecastingHorizon
        Absolute representation of forecasting horizon.
    """
    if not fh.is_relative:
        return fh._new()

    else:
        relative = fh.to_pandas()
        _check_cutoff(cutoff, relative)
        is_timestamp = isinstance(cutoff, pd.Timestamp)

        if is_timestamp:
            # coerce to pd.Period for reliable arithmetic operations and
            # computations of time deltas
            cutoff = _coerce_to_period(cutoff, freq=fh.freq)

        absolute = cutoff + relative

        if is_timestamp:
            # coerce back to DatetimeIndex after operation
            absolute = absolute.to_timestamp(fh.freq)

        return fh._new(absolute, is_relative=False, freq=fh.freq)


def _check_cutoff(cutoff, index):
    """Check if the cutoff is valid based on time index of forecasting horizon.

    Validates that the cutoff is
    compatible with the time index of the forecasting horizon.

    Parameters
    ----------
    cutoff : pd.Period, pd.Timestamp, int, optional (default=None)
        Cutoff value is required to convert a relative forecasting
        horizon to an absolute one and vice versa.
    index : pd.PeriodIndex or pd.DataTimeIndex
        Forecasting horizon time index that the cutoff value will be checked
        against.
    """
    if cutoff is None:
        raise ValueError("`cutoff` must be given, but found none.")

    if isinstance(index, pd.PeriodIndex):
        assert isinstance(cutoff, pd.Period)
        assert index.freqstr == cutoff.freqstr

    if isinstance(index, pd.DatetimeIndex):
        assert isinstance(cutoff, pd.Timestamp)


def _coerce_to_period(x, freq=None):
    """Coerce pandas time index to a alternative pandas time index.

    This coerces pd.Timestamp to pd.Period or pd.DatetimeIndex to
    pd.PeriodIndex, because pd.Period and pd.PeriodIndex allow more reliable
    arithmetic operations with time indices.

    Parameters
    ----------
    x : pandas Index or index element
        pandas Index to convert.
    freq : pandas frequency string

    Returns
    -------
    index : pd.Period or pd.PeriodIndex
        Index or index element coerced to period based format.
    """
    # timestamp/freq combinations are deprecated from 0.13.0
    # warning should be replaced by exception in 0.14.0
    if isinstance(x, pd.Timestamp) and freq is None:
        freq = x.freq
        warn(
            "use of ForecastingHorizon methods with pd.Timestamp carrying freq "
            "is deprecated since 0.13.0 and will raise exception from 0.14.0"
        )
    #   raise ValueError("_coerce_to_period requires freq if x is pd.Timestamp")
    try:
        return x.to_period(freq)
    except (ValueError, AttributeError) as e:
        msg = str(e)
        if "Invalid frequency" in msg or "_period_dtype_code" in msg:
            raise ValueError(
                "Invalid frequency. Please select a frequency that can "
                "be converted to a regular `pd.PeriodIndex`. For other "
                "frequencies, basic arithmetic operation to compute "
                "durations currently do not work reliably."
            )
        else:
            raise


def _index_range(relative, cutoff):
    """Return Index Range relative to cutoff."""
    _check_cutoff(cutoff, relative)
    is_timestamp = isinstance(cutoff, pd.Timestamp)

    if is_timestamp:
        # coerce to pd.Period for reliable arithmetic operations and
        # computations of time deltas
        cutoff = _coerce_to_period(cutoff, freq=cutoff.freqstr)

    absolute = cutoff + relative

    if is_timestamp:
        # coerce back to DatetimeIndex after operation
        absolute = absolute.to_timestamp(cutoff.freqstr)
    return absolute

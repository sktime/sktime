# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
ForecastingHorizon: pandas-agnostic forecasting horizon implementation.

Architecture: ForecastingHorizon stores only {_values, _is_relative, _freq,
_values_are_nanos}. All temporal inputs are normalized to integer steps
(period ordinals) at construction.
This means that all internal arithmetic is pure integer math,
and the only place where pandas logic is needed is in the conversion of inputs to
this internal representation (PandasFHConverter).
This design allows ForecastingHorizon to be pandas-free,
while still supporting all the same input types and frequencies as before.
All pandas-specific logic is delegated to the _fh_utils module.

Internal state of ForecastingHorizon consists of the following attributes:

``_values``: int64 numpy array — integer steps (period ordinals for
  absolute, step counts for relative), or raw nanoseconds when
  ``_values_are_nanos`` is True. Read-only after construction.

``_is_relative``: bool — whether values are relative to training cutoff.

``_freq``: str or None — frequency mnemonic (e.g. ``"M"``, ``"D"``).
  None for plain integer horizons or when freq has not yet been assigned.
  The ``freq`` setter is the only deliberate mutation point on the object.

``_values_are_nanos``: bool — True when values are raw nanoseconds
  pending conversion to integer steps (e.g. freq-less TimedeltaIndex input).
  Set to False once freq is assigned via the ``freq`` setter.
"""

__all__ = ["ForecastingHorizon"]

import numpy as np

from sktime.forecasting.base._fh_utils import PandasFHConverter
from sktime.forecasting.base._fh_values import (
    FHValues,
    FHValueType,
)  # remove this later
from sktime.forecasting.base._freq_mnemonic import validate_freq

# <check></check>
# this is the marker left to mark all delayed checks
# all occurences must be removed/addressed before merging this code

# types whose is_relative is compatible with both True and False
_RELATIVE_NEUTRAL_TYPES = (int, np.integer, list, range, np.ndarray)


class ForecastingHorizon:
    """Represents the time points to forecast, relative or absolute.

    A forecasting horizon specifies which future (or past) time points a
    forecaster should predict. It accepts a wide range of input types:
    plain integers, pandas PeriodIndex, DatetimeIndex, TimedeltaIndex, etc.
    and normalizes them internally to a sorted, deduplicated int64 numpy
    array of integer steps (period ordinals for absolute, step counts for
    relative). Temporal inputs that cannot be immediately converted to
    integer steps (e.g. freq-less TimedeltaIndex) are stored as raw
    nanoseconds and converted when frequency information becomes available.

    Parameters
    ----------
    values : int, list, np.ndarray, range, pd.Index, pd.Timedelta,
        pd.offsets.BaseOffset
        Values of forecasting horizon.
        Supported types without pandas dependency:
        - ``int`` or ``np.integer`` : single integer step
        - ``list[int]`` : list of integer steps
        - ``np.ndarray`` : integer or timedelta64 array
        - ``range`` : Python range object
        Supported pandas types (delegated to PandasFHConverter):
        - ``pd.PeriodIndex``, ``pd.DatetimeIndex``, ``pd.TimedeltaIndex``
        - ``pd.RangeIndex``, ``pd.Index`` (integer or timedelta dtype)
        - ``pd.Timedelta``, ``pd.offsets.BaseOffset``
        - ``list[pd.Period]``, ``list[pd.Timestamp]``, ``list[pd.Timedelta]``
        - ``list[pd.offsets.BaseOffset]``, ``list[np.timedelta64]``
    is_relative : bool, optional (default=None)
        Whether the forecasting horizon is relative to the training cutoff.
        If True, values are relative to end of training series.
        If False, values are absolute.
        If None, inferred from value type:
        - int values default to relative (is_relative=True)
        - timedelta values are always relative
        - Period and Timestamp values are always absolute
        Note: integer values are compatible with both relative and absolute
        interpretations. For integers, pass ``is_relative=False`` explicitly
        if absolute is intended, as the default inference interprets it as
        relative.
    freq : str, pd.Index, pd.Period, pandas offset, or sktime forecaster,
        optional (default=None)
        Frequency information for the horizon values.
        When values already carry frequency (e.g. ``pd.PeriodIndex``,
        ``pd.DatetimeIndex`` with freq, or ``pd.TimedeltaIndex`` with freq),
        provided ``freq`` must match the values' frequency, otherwise a
        ValueError is raised.
        When values do not carry frequency (e.g. int, list, np.ndarray,
        or freq-less ``pd.TimedeltaIndex``), ``freq`` is used directly if
        provided. For freq-less ``pd.TimedeltaIndex``, values are stored as
        raw nanoseconds until freq is assigned (via this parameter or later
        through the ``freq`` setter).

    Examples
    --------
    >>> from sktime.forecasting.base._fh_v2 import ForecastingHorizon
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> fh.is_relative
    True
    >>> fh.to_numpy()
    array([1, 2, 3])
    """

    def __init__(
        self,
        values=None,
        is_relative: bool | None = None,
        freq=None,
    ):
        # convert values to internal representation
        # both paths return (values, is_relative, freq, values_are_nanos)
        if isinstance(values, (int, np.integer, range, np.ndarray)) or (
            isinstance(values, list)
            and len(values) > 0
            and isinstance(values[0], (int, np.integer))
        ):
            # canonical path: plain Python/numpy types — no pandas needed
            vals, inferred_is_relative, freq_val, nanos_flag = self._coerce_canonical(
                values
            )
        else:
            # coerced path: pandas types and non-int lists — delegate to converter
            vals, inferred_is_relative, freq_val, nanos_flag = (
                PandasFHConverter.to_internal(values)
            )

        # sort, deduplicate, and store
        self._values = np.unique(vals)
        self._freq = freq_val
        self._values_are_nanos = nanos_flag

        # handle empty arrays
        if len(self._values) == 0:
            self._freq = None
            self._values_are_nanos = False

        # set freq via setter (single gate for validation)
        if freq is not None:
            self.freq = freq

        # determine is_relative
        self._is_relative = ForecastingHorizon._resolve_is_relative(
            is_relative, inferred_is_relative, values
        )

        # lock values array against accidental mutation
        self._values.flags.writeable = False

    @staticmethod
    def _coerce_canonical(values):
        """Coerce canonical (non-pandas) values to internal representation.

        Handles int, np.integer, range, np.ndarray, and list[int].

        Parameters
        ----------
        values : int, np.integer, range, np.ndarray, or list[int]
            Input values.

        Returns
        -------
        tuple of (np.ndarray, bool, str or None, bool)
            (values_array, inferred_is_relative, freq, values_are_nanos)
            Same order as PandasFHConverter.to_internal.
        """
        inferred_is_relative = True
        freq = None
        values_are_nanos = False

        if isinstance(values, (int, np.integer)):
            arr = np.array([int(values)], dtype=np.int64)
            return arr, inferred_is_relative, freq, values_are_nanos

        if isinstance(values, range):
            arr = np.array(list(values), dtype=np.int64)
            return arr, inferred_is_relative, freq, values_are_nanos

        if isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError(f"Expected 1-D array, got {values.ndim}-D array")
            if len(values) == 0:
                raise ValueError("Forecasting horizon values must not be empty.")
            if np.issubdtype(values.dtype, np.timedelta64):
                arr = values.astype("timedelta64[ns]").view(np.int64).copy()
                values_are_nanos = True
                return arr, inferred_is_relative, freq, values_are_nanos
            if np.issubdtype(values.dtype, np.integer):
                arr = values.astype(np.int64).copy()
                return arr, inferred_is_relative, freq, values_are_nanos
            raise TypeError(
                f"np.ndarray with dtype {values.dtype} is not supported. "
                f"Expected integer or timedelta64 dtype."
            )

        # list[int]
        for i, v in enumerate(values[1:], start=1):
            if not isinstance(v, (int, np.integer)):
                raise TypeError(
                    f"Element at index 0 is of type "
                    f"{type(values[0]).__name__}, but element at "
                    f"index {i} is {type(v).__name__}. "
                    "All list elements must be of the same type."
                )
        arr = np.array(values, dtype=np.int64)
        return arr, inferred_is_relative, freq, values_are_nanos

    @staticmethod
    def _resolve_is_relative(is_relative, inferred_is_relative, values):
        """Resolve is_relative from user-provided and inferred values.

        Parameters
        ----------
        is_relative : bool or None
            User-provided is_relative flag.
        inferred_is_relative : bool
            is_relative inferred from the type of values.
        values : object
            Original values passed to ForecastingHorizon.__init__.

        Returns
        -------
        bool
            Resolved is_relative value.

        Raises
        ------
        TypeError
            If is_relative is not a boolean or None.
        ValueError
            If is_relative conflicts with the inferred value and the input
            type strictly implies one interpretation (e.g. PeriodIndex is always
            absolute, TimedeltaIndex is always relative).
        """
        if is_relative is None:
            return inferred_is_relative

        if not isinstance(is_relative, bool):
            raise TypeError("`is_relative` must be a boolean or None")

        if inferred_is_relative is not None and is_relative != inferred_is_relative:
            # integers are compatible with both relative and absolute,
            # so only raise when the type strictly implies one interpretation
            # (e.g. PeriodIndex is always absolute, TimedeltaIndex always relative)
            if not isinstance(values, _RELATIVE_NEUTRAL_TYPES):
                raise ValueError(
                    f"Conflict between inferred is_relative={inferred_is_relative} "
                    f"and provided is_relative={is_relative}. Please resolve the "
                    "conflict by providing a consistent `is_relative` value or "
                    "adjusting the input `values`."
                )

        return is_relative

    @classmethod
    def _create(cls, values, is_relative, freq=None, values_are_nanos=False):
        """Construct a ForecastingHorizon without coercion or validation.

        Fast-path constructor for internal use. Creates a new instance
        directly from pre-computed attributes, bypassing ``__init__``.
        Values must already be sorted and deduplicated.

        Parameters
        ----------
        values : np.ndarray
            Sorted, deduplicated int64 array of values.
        is_relative : bool
            Whether the horizon is relative.
        freq : str or None, optional
            Frequency string.
        values_are_nanos : bool, optional (default=False)
            Whether values are raw nanoseconds.

        Returns
        -------
        ForecastingHorizon
            New instance.
        """
        obj = object.__new__(cls)
        assert len(values) == 0 or np.all(np.diff(values) > 0), (
            "_create expects sorted, unique values"
        )
        obj._values = values
        obj._values.flags.writeable = False
        obj._is_relative = is_relative
        obj._freq = freq
        obj._values_are_nanos = values_are_nanos
        return obj

    @property
    def is_relative(self) -> bool:
        """Whether forecasting horizon is relative to the end of the training series.

        Returns
        -------
        is_relative : bool
        """
        return self._is_relative

    @is_relative.setter
    def is_relative(self, value: bool) -> None:
        """Set is_relative flag."""
        self._is_relative = value

    @property
    def freq(self) -> str | None:
        """Frequency string, or None."""
        return self._fhvalues.freq

    @freq.setter
    def freq(self, obj) -> None:
        """Set frequency from string, pd.Index, pd.offset, or forecaster.

        For string inputs, the frequency is validated against accepted
        standard time series frequencies (e.g. ``"M"``, ``"D"``, ``"2h"``).
        If validation fails, a fallback check via pandas ``to_offset`` is
        attempted before raising an error.

        For non-string inputs (pd.Index, pd.offsets.BaseOffset, forecaster),
        frequency is extracted and normalized via PandasFHConverter.

        If the FHValues already carry a frequency (inferred from values),
        the new frequency must match, otherwise a ValueError is raised.

        Parameters
        ----------
        obj : str, pd.Index, pd.offsets.BaseOffset, or forecaster
            Object carrying frequency information.
            When str, must be a valid frequency mnemonic, optionally
            with an integer multiplier prefix.
            Accepted base frequencies: Y, Q, M, W, D, h, min, s, ms, us, ns.
            Examples: ``"M"``, ``"2D"``, ``"4h"``, ``"15min"``.

        Raises
        ------
        ValueError
            If freq is already set and conflicts with new value, or
            if a string freq is not a recognized frequency mnemonic.
        """
        if isinstance(obj, str):
            try:
                new_freq = validate_freq(obj)
            except ValueError:
                # fallback: try pandas to_offset for exotic but valid freqs
                new_freq = PandasFHConverter.extract_freq(obj)
                if new_freq is None:
                    raise
        elif obj is None:
            return
        else:
            new_freq = PandasFHConverter.extract_freq(obj)

        old_freq = self._fhvalues.freq
        if old_freq is not None and new_freq is not None and old_freq != new_freq:
            raise ValueError(
                f"Frequencies do not match: current={old_freq!r}, new={new_freq!r}"
            )
        if new_freq is not None:
            self._fhvalues = self._fhvalues._new(freq=new_freq)

    # core conversion methods

    # <check>
    # for a drop-in replacement of the old FH,
    # we want to allow users to call to_relative without cutoff
    # but the old FH also raised an error at `cutoff is None`
    # why not make it explicit and require cutoff to be passed for
    # to_relative and to_absolute methods?
    # </check>
    def to_relative(self, cutoff=None):
        """Return relative version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value required for conversion.

        Returns
        -------
        ForecastingHorizon
            Relative representation of forecasting horizon.
        """
        if self._is_relative:
            return self._new()

        if cutoff is None:
            raise ValueError(
                "`cutoff` must be provided to convert absolute FH to relative."
            )

        cutoff_val, cutoff_type, cutoff_freq, cutoff_tz = (
            PandasFHConverter.cutoff_to_internal(cutoff, freq=self.freq)
        )

        # mismatch between the FH frequency and cutoff frequency
        # can happen and should be flagged
        if (
            self.freq is not None
            and cutoff_freq is not None
            and self.freq != cutoff_freq
        ):
            raise ValueError(
                f"Frequency mismatch between FH and cutoff: "
                f"FH freq={self.freq}, cutoff freq={cutoff_freq}"
            )
        freq = self.freq or cutoff_freq

        # vtype can only be absolute types (PERIOD, DATETIME, or INT) at this point,
        # because if it were a relative type,
        # to_relative would return at the start of the method
        vtype = self._fhvalues.value_type
        vals = self._fhvalues.values

        # <check>
        # PandasFHConverter methods not yet implemented
        # </check>
        if vtype == FHValueType.PERIOD:
            # ordinal difference -> integer steps
            # divide by freq multiplier to get step count
            # e.g., "2D" has multiplier 2, so ordinal diff of 4 = 2 steps
            relative_vals = vals - cutoff_val
            if freq is not None:
                mult = PandasFHConverter.freq_multiplier(freq)
                if mult != 1:
                    relative_vals = relative_vals // mult
            fhv = FHValues(relative_vals.astype(np.int64), FHValueType.INT, freq=freq)
            return self._new(fhvalues=fhv, is_relative=True)
            # another place where _new is needed to create a new ForecastingHorizon
            # instance with modified values but same metadata

        if vtype == FHValueType.DATETIME:
            # nanosecond difference
            relative_nanos = (vals - cutoff_val).astype(np.int64)
            if freq is not None:
                # convert nanosecond diffs to integer steps using freq
                relative_vals = PandasFHConverter.nanos_to_steps(
                    relative_nanos, freq, ref_nanos=cutoff_val
                )
                fhv = FHValues(relative_vals, FHValueType.INT, freq=freq)
            else:
                # no freq: return as TIMEDELTA nanoseconds
                fhv = FHValues(relative_nanos, FHValueType.TIMEDELTA, freq=freq)
            return self._new(fhvalues=fhv, is_relative=True)

        if vtype == FHValueType.INT:
            # absolute int - cutoff int -> relative int
            relative_vals = vals - cutoff_val
            fhv = FHValues(relative_vals.astype(np.int64), FHValueType.INT, freq=freq)
            return self._new(fhvalues=fhv, is_relative=True)

        # if we reach this point,
        # it means the value type is not compatible with relative representation
        raise TypeError(f"Cannot convert {vtype.name} to relative.")

    def to_absolute(self, cutoff):
        """Return absolute version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one (and vice versa).
            If pd.Index, last/latest value is considered the cutoff

        Returns
        -------
        ForecastingHorizon
            Absolute representation of forecasting horizon.
        """
        if not self._is_relative:
            # <check> _new is not yet implemented </check>"
            return self._new()

        cutoff_val, cutoff_type, cutoff_freq, cutoff_tz = (
            PandasFHConverter.cutoff_to_internal(cutoff, freq=self.freq)
        )

        # mismatch between the FH frequency and cutoff frequency
        # can happen and should be flagged
        if (
            self.freq is not None
            and cutoff_freq is not None
            and self.freq != cutoff_freq
        ):
            raise ValueError(
                f"Frequency mismatch between FH and cutoff: "
                f"FH freq={self.freq}, cutoff freq={cutoff_freq}"
            )
        freq = self.freq or cutoff_freq

        # vtype can only be relative types (INT or TIMEDELTA) at this point,
        # because if it were an absolute type,
        # to_absolute would return at the start of the method
        vtype = self._fhvalues.value_type
        vals = self._fhvalues.values

        if vtype == FHValueType.INT:
            if cutoff_type == FHValueType.PERIOD:
                # int steps + period ordinal -> period ordinals
                # multiply by freq multiplier for multi-step freqs
                # e.g., "2D" has multiplier 2, so step 1 = 2 ordinals
                step_vals = vals
                if freq is not None:
                    mult = PandasFHConverter.freq_multiplier(freq)
                    if mult != 1:
                        step_vals = vals * mult
                absolute_vals = cutoff_val + step_vals
                fhv = FHValues(
                    absolute_vals.astype(np.int64),
                    FHValueType.PERIOD,
                    freq=freq,
                )
                return self._new(fhvalues=fhv, is_relative=False)
            if cutoff_type == FHValueType.DATETIME:
                if freq is None:
                    raise ValueError(
                        "freq is required to convert integer relative FH "
                        "to absolute datetime. Set freq on the FH or provide "
                        "a cutoff with frequency information."
                    )
                nanos = PandasFHConverter.steps_to_nanos(
                    vals, freq, ref_nanos=cutoff_val
                )
                absolute_vals = cutoff_val + nanos
                fhv = FHValues(
                    absolute_vals.astype(np.int64),
                    FHValueType.DATETIME,
                    freq=freq,
                    timezone=cutoff_tz,
                )
                return self._new(fhvalues=fhv, is_relative=False)

            if cutoff_type == FHValueType.INT:
                # int + int -> int (absolute)
                absolute_vals = cutoff_val + vals
                fhv = FHValues(
                    absolute_vals.astype(np.int64),
                    FHValueType.INT,
                    freq=freq,
                )
                return self._new(fhvalues=fhv, is_relative=False)
        if vtype == FHValueType.TIMEDELTA:
            if cutoff_type == FHValueType.DATETIME:
                # nanos + nanos -> absolute datetime nanos
                absolute_vals = cutoff_val + vals
                fhv = FHValues(
                    absolute_vals.astype(np.int64),
                    FHValueType.DATETIME,
                    freq=freq,
                    timezone=cutoff_tz,
                )
                return self._new(fhvalues=fhv, is_relative=False)
        # if we reach this point,
        # it means the value type is not compatible with absolute representation
        raise TypeError(
            f"Cannot convert {vtype.name} (relative) to absolute "
            f"with cutoff type {cutoff_type.name}."
        )

    def to_pandas(self):
        """Return forecasting horizon values as pd.Index.

        Returns
        -------
        pd.Index
            Pandas Index containing the forecasting horizon values.
        """
        return PandasFHConverter.to_pandas_index(self._fhvalues)

    def to_numpy(self, **kwargs) -> np.ndarray:
        """Return forecasting horizon values as numpy array.

        Returns
        -------
        np.ndarray
            Numpy array of int64 values.
        """
        return self._fhvalues.values.copy()

    def to_absolute_index(self, cutoff=None):
        """Return absolute values as pandas Index.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value for conversion.

        Returns
        -------
        pd.Index
            Absolute forecasting horizon as pandas Index.
        """
        return self.to_absolute(cutoff).to_pandas()

    def to_absolute_int(self, start, cutoff=None):
        """Return absolute values as zero-based integer index from ``start``.

        Parameters
        ----------
        start : pd.Period, pd.Timestamp, int
            Start value returned as zero.
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value for conversion.

        Returns
        -------
        ForecastingHorizon
            Absolute representation as zero-based integer index.
        """
        # get absolute representation
        absolute = self.to_absolute(cutoff)
        abs_vals = absolute._fhvalues.values
        abs_type = absolute._fhvalues.value_type
        abs_freq = absolute._fhvalues.freq

        # convert start to internal
        start_val, start_type, start_freq, _ = PandasFHConverter.cutoff_to_internal(
            start, freq=self.freq
        )

        # compute zero-based integers
        if abs_type == FHValueType.PERIOD:
            integers = abs_vals - start_val
            # check for frequency mismatch between FH freq, cutoff freq, and start freq
            freq = None
            for candidate in (abs_freq, self.freq, start_freq):
                if candidate is not None:
                    if freq is None:
                        freq = candidate
                    elif candidate != freq:
                        # below error message may need better wording
                        # the idea is to flag any mismatch between the three freqs
                        raise ValueError(
                            f"Frequency mismatch in to_absolute_int: "
                            f"abs_freq={abs_freq}, self.freq={self.freq}, "
                            f"start_freq={start_freq}. All must agree."
                        )
            # <check> Can the freq be ever None here? If so, how to handle? </check>
            # divide by freq multiplier for multi-step freqs
            if freq is not None:
                mult = PandasFHConverter.freq_multiplier(freq)
                if mult != 1:
                    integers = integers // mult
            else:
                # <check>
                # no freq available
                # raw ordinal differences may be incorrect for multi-step frequencies
                # but we have no way to normalize without freq information
                # should this be flagged as a warning? or an error? or just left as is?
                # </check>
                pass
        elif abs_type == FHValueType.DATETIME:
            nanos_diff = abs_vals - start_val
            # check for frequency mismatch between FH freq, cutoff freq, and start freq
            freq = None
            for candidate in (abs_freq, self.freq, start_freq):
                if candidate is not None:
                    if freq is None:
                        freq = candidate
                    elif candidate != freq:
                        # below error message may need better wording
                        # the idea is to flag any mismatch between the three freqs
                        raise ValueError(
                            f"Frequency mismatch in to_absolute_int: "
                            f"abs_freq={abs_freq}, self.freq={self.freq}, "
                            f"start_freq={start_freq}. All must agree."
                        )
            if freq is not None:
                integers = PandasFHConverter.nanos_to_steps(
                    nanos_diff, freq, ref_nanos=start_val
                )
            else:
                # fall back to raw nanos difference
                integers = nanos_diff
        else:
            integers = abs_vals - start_val

        fhv = FHValues(integers.astype(np.int64), FHValueType.INT, freq=self.freq)
        return self._new(fhvalues=fhv, is_relative=False)

    # In-sample and out-of-sample methods

    def _is_in_sample(self, cutoff=None) -> np.ndarray:
        """Return boolean array indicating in-sample values.

        In-sample values have relative representation <= 0.
        """
        relative = self.to_relative(cutoff)
        return relative._fhvalues.values <= 0

    def _is_out_of_sample(self, cutoff=None) -> np.ndarray:
        """Return boolean array indicating out-of-sample values."""
        return np.logical_not(self._is_in_sample(cutoff))

    def is_all_in_sample(self, cutoff=None) -> bool:
        """Whether the forecasting horizon is purely in-sample.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value.

        Returns
        -------
        bool
        """
        return bool(self._is_in_sample(cutoff).all())

    def is_all_out_of_sample(self, cutoff=None) -> bool:
        """Whether the forecasting horizon is purely out-of-sample.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value.

        Returns
        -------
        bool
        """
        return bool(self._is_out_of_sample(cutoff).all())

    def to_in_sample(self, cutoff=None):
        """Return in-sample values of fh.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value for conversion.

        Returns
        -------
        ForecastingHorizon
            In-sample values of forecasting horizon.
        """
        mask = self._is_in_sample(cutoff)
        filtered_vals = self._fhvalues.values[mask]
        fhv = self._fhvalues._new(values=filtered_vals)
        return self._new(fhvalues=fhv)

    def to_out_of_sample(self, cutoff=None):
        """Return out-of-sample values of fh.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value for conversion.

        Returns
        -------
        ForecastingHorizon
            Out-of-sample values of forecasting horizon.
        """
        mask = self._is_out_of_sample(cutoff)
        filtered_vals = self._fhvalues.values[mask]
        fhv = self._fhvalues._new(values=filtered_vals)
        return self._new(fhvalues=fhv)

    # indexer method
    # <check> partial implementation, supports relative integer FH
    # </check>
    def to_indexer(self, cutoff=None, from_cutoff=True):
        """Return zero-based indexer for array access.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, optional
            Cutoff value for conversion.
        from_cutoff : bool, optional (default=True)
            If True, zero-based relative to cutoff.
            If False, zero-based relative to first value in fh.

        Returns
        -------
        pd.Index
            Zero-based integer indexer.
        """
        if from_cutoff:
            relative = self.to_relative(cutoff)
            vtype = relative._fhvalues.value_type
            if vtype == FHValueType.INT:
                indexer_vals = relative._fhvalues.values - 1
            elif vtype == FHValueType.TIMEDELTA:
                freq = self.freq
                if freq is None and cutoff is not None:
                    _, _, cutoff_freq, _ = PandasFHConverter.cutoff_to_internal(
                        cutoff, freq=self.freq
                    )
                    freq = cutoff_freq
                if freq is None:
                    raise ValueError(
                        "freq is required to compute an integer indexer "
                        "from timedelta-based forecasting horizon. "
                        "Set freq on the FH or provide a cutoff with "
                        "frequency information."
                    )
                # get cutoff nanos for calendar-aware conversion
                if cutoff is not None:
                    cutoff_val, _, _, _ = PandasFHConverter.cutoff_to_internal(
                        cutoff, freq=self.freq
                    )
                    ref_nanos = cutoff_val
                else:
                    ref_nanos = np.int64(0)
                # convert timedelta nanos to integer steps, then zero-base
                indexer_vals = (
                    PandasFHConverter.nanos_to_steps(
                        relative._fhvalues.values, freq, ref_nanos=ref_nanos
                    )
                    - 1
                )
            else:
                raise TypeError(
                    f"Cannot compute indexer for relative FH with "
                    f"value type {vtype.name}."
                )
        else:
            relative = self.to_relative(cutoff)
            vals = relative._fhvalues.values
            indexer_vals = vals - vals[0]

        fhv = FHValues(indexer_vals.astype(np.int64), FHValueType.INT)
        return PandasFHConverter.to_pandas_index(fhv)

    def _is_contiguous(self) -> bool:
        """Check if forecasting horizon values form a contiguous sequence.

        Returns
        -------
        bool
        """
        return self._fhvalues.is_contiguous()

    def get_expected_pred_idx(self, y=None, cutoff=None, sort_by_time=False):
        """Construct expected prediction output index.

        Parameters
        ----------
        y : pd.DataFrame, pd.Series, pd.Index, or None (default=None)
            Data to compute fh relative to.
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional (default=None)
            Cutoff value. If None, inferred from ``y``.
        sort_by_time : bool, optional (default=False)
            For MultiIndex returns, whether to sort by time index.

        Returns
        -------
        pd.Index
            Expected index of y_pred returned by predict.
        """
        return PandasFHConverter.build_pred_index(
            fh=self,
            y=y,
            cutoff=cutoff,
            sort_by_time=sort_by_time,
        )

    # Dunders -> Arithmatic operators

    def __add__(self, other):
        if isinstance(other, ForecastingHorizon):
            result = self._fhvalues.values + other._fhvalues.values
        else:
            result = self._fhvalues.values + np.int64(other)
        fhv = self._fhvalues._new(values=result)
        return self._new(fhvalues=fhv)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ForecastingHorizon):
            result = self._fhvalues.values - other._fhvalues.values
        else:
            result = self._fhvalues.values - np.int64(other)
        fhv = self._fhvalues._new(values=result)
        return self._new(fhvalues=fhv)

    def __rsub__(self, other):
        # not checking if other is FH here
        # because __rsub__ is mostly called
        # when other does not support the operation with FH,
        # in which case we want to treat other as a scalar.
        # If other is FH, then other minus self
        # would have been handled by other.__sub__
        # and this method would not be called
        result = np.int64(other) - self._fhvalues.values
        fhv = self._fhvalues._new(values=result)
        return self._new(fhvalues=fhv)

    def __mul__(self, other):
        if isinstance(other, ForecastingHorizon):
            result = self._fhvalues.values * other._fhvalues.values
        else:
            result = self._fhvalues.values * np.int64(other)
        fhv = self._fhvalues._new(values=result)
        return self._new(fhvalues=fhv)

    def __rmul__(self, other):
        return self.__mul__(other)

    # Dunders -> comparison operators
    # Note:
    # for euqality operator we can either do:
    # 1. Element-wise comparison (numpy-style):
    #   compare only raw int64 arrays elementwise
    #   and return a boolean array,
    #   fh == 3 → array([False, False, True])
    # 2.Object identity/equality (Python-style):
    #   "are these two FH objects the same?"
    #   compare the entire FHValues instances,
    #   which would take into account the value type,
    #   freq, and timezone as well and return a single boolean
    #   indicating whether the two FHValues instances are equal in all aspects.
    #
    # Number 2 seems more consistent with how equality is usually
    # implemented in Python classes,
    # but 1 might be usefull for comparing two forecasting horizons elementwise,
    # for example when aligning two forecasting horizons with different cutoffs.
    #
    # Current implementation is for number 1

    def __eq__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._fhvalues.values == other._fhvalues.values
        return self._fhvalues.values == np.int64(other)

    def __ne__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._fhvalues.values != other._fhvalues.values
        return self._fhvalues.values != np.int64(other)

    def __lt__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._fhvalues.values < other._fhvalues.values
        return self._fhvalues.values < np.int64(other)

    def __le__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._fhvalues.values <= other._fhvalues.values
        return self._fhvalues.values <= np.int64(other)

    def __gt__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._fhvalues.values > other._fhvalues.values
        return self._fhvalues.values > np.int64(other)

    def __ge__(self, other):
        if isinstance(other, ForecastingHorizon):
            return self._fhvalues.values >= other._fhvalues.values
        return self._fhvalues.values >= np.int64(other)

    # Dunders -> container methods len, getitem, max, min
    def __len__(self):
        return len(self._fhvalues)

    def __getitem__(self, key):
        result = self._fhvalues[key]
        if isinstance(result, FHValues):
            return self._new(fhvalues=result)
        # scalar — return as-is
        return result

    def max(self):
        """Return the maximum value."""
        return self._fhvalues.max()

    def min(self):
        """Return the minimum value."""
        return self._fhvalues.min()

    # Below method computes a hash for the ForecastingHorizon instance,
    # The hash is computed based on the tuple containing:
    # 1. the internal FHValues instance which itself has a custom __hash__ based
    #    on its int64 array bytes + metadata
    # 2. the is_relative boolean flag, natively hashable
    # <check>
    # this implementation assumes that FHValues
    # has a proper __hash__ method implemented.
    # Note: currently there's an inconsistency between __eq__ and __hash__
    # Python requires:
    #   If a == b, then hash(a) == hash(b)
    # current __eq__ only compares raw int64 arrays element-wise
    # and returns a numpy array, not a bool.
    # while __hash__ considers numpy array + all metadata + is_relative.
    # This violates the contract.
    # Two objects could be "=="" (same raw values)
    # but have different hashes (different freq or is_relative).
    # To fix, either:
    # Make __eq__ return a single bool comparing all attributes when other is
    # ForecastingHorizon, or
    # Move element-wise comparison to a separate method
    # and keep __eq__ consistent with __hash__.
    # Need to consider this in th context of forecasting horizon usage.
    # </check>
    def __hash__(self):
        return hash((self._fhvalues, self._is_relative))

    def __repr__(self):
        class_name = type(self).__name__
        vals = self._fhvalues
        vtype = vals.value_type.name
        n = len(vals)
        parts = [f"n={n}", f"type={vtype}", f"is_relative={self._is_relative}"]
        if vals.freq is not None:
            parts.append(f"freq={vals.freq!r}")
        # if less than 6 values, show all values in repr,
        # otherwise show 1st and last 3 only
        if n <= 6:
            parts.append(f"values={vals.values.tolist()}")
        else:
            head = vals.values[:3].tolist()
            tail = vals.values[-3:].tolist()
            parts.append(
                f"values=[{head[0]}, {head[1]}, {head[2]}, ..., "
                f"{tail[0]}, {tail[1]}, {tail[2]}]"
            )
        return f"{class_name}({', '.join(parts)})"

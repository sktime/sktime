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

__all__ = ["ForecastingHorizon", "VALID_FORECASTING_HORIZON_TYPES"]

import numpy as np

from sktime.forecasting.base._fh_utils import (
    _PANDAS_FH_INPUT_TYPES,
    PandasFHConverter,
)
from sktime.forecasting.base._freq_mnemonic import validate_freq

VALID_FORECASTING_HORIZON_TYPES = int | list | np.ndarray | _PANDAS_FH_INPUT_TYPES

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

        - ``int`` or ``np.integer``: single integer, coerced to ``range(1, values + 1)``
        - ``list[int]``: list of integer steps
        - ``np.ndarray``: integer or timedelta64 array
        - ``range``: Python range object

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
        # if values is already an FH, use _create directly
        if isinstance(values, ForecastingHorizon):
            src = values
            copy = self._create(
                src._values.copy(),
                src._is_relative,
                src._freq,
                src._values_are_nanos,
            )
            self._values = copy._values
            self._is_relative = copy._is_relative
            self._freq = copy._freq
            self._values_are_nanos = copy._values_are_nanos
            # apply overrides if provided
            if freq is not None:
                self.freq = freq  # setter validates mismatch
            if is_relative is not None:
                self._is_relative = self._resolve_is_relative(
                    is_relative, src._is_relative, values
                )
            return

        # canonical path: plain Python/numpy types: no pandas needed
        # `np.timedelta64` is a subclass of `np.integer`,
        # without exclusion it enters the int path instead of the converter path.
        # Empty lists `[]` also need special handling to avoid downstream errors
        if (
            isinstance(values, (int, np.integer, range, np.ndarray))
            and not isinstance(values, np.timedelta64)
            or (
                isinstance(values, list)
                and (
                    len(values) == 0
                    or (
                        isinstance(values[0], (int, np.integer))
                        and not isinstance(values[0], np.timedelta64)
                    )
                )
            )
        ):
            vals, inferred_is_relative, freq_val, nanos_flag = self._coerce_canonical(
                values
            )
        else:
            # coerced path: pandas types and non-int lists — delegate to converter.
            # Passed freq is only used when DatatimeIndex type is passed without freq
            # attribute, as a fallback to extract freq from the provided freq parameter
            # in all other cases, freq is extracted from the values themselves
            # if present.
            # Mismatch between extracted freq and passed freq is done later in this
            # constructor after coercion,
            # to allow the converter to handle pandas alias normalization first.
            vals, inferred_is_relative, freq_val, nanos_flag = (
                PandasFHConverter.to_internal(values, freq)
            )

        # sort, deduplicate, and store
        # reject if duplicates found
        sorted_vals = np.unique(vals)
        if len(sorted_vals) != len(vals):
            raise ValueError(
                "Forecasting horizon values must be unique. "
                f"Found duplicates in: {vals!r}"
            )
        self._values = sorted_vals
        self._values_are_nanos = nanos_flag

        # handle empty arrays
        if len(self._values) == 0:
            self._freq = None
            self._values_are_nanos = False

        # set freq via setter (single gate for validation and nanos conversion)
        self._freq = None
        if freq_val is not None:
            self.freq = freq_val
            # setter normalizes string and sets _freq, raises ValueError on mismatch
        if freq is not None:
            self.freq = freq
            # setter normalizes, checks mismatch with existing _freq

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
            # processing depends on the sign of the integer
            n = int(values)
            # if positive, create range(1, n+1)
            if n > 0:
                arr = np.arange(1, n + 1, dtype=np.int64)
            # if negative, create array with a single value n
            else:
                arr = np.array([n], dtype=np.int64)
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

        # list[int] or empty list
        if len(values) == 0:
            arr = np.array([], dtype=np.int64)
            return arr, inferred_is_relative, freq, values_are_nanos

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
        if len(values) > 0 and not np.all(np.diff(values) > 0):
            raise ValueError("_create expects sorted, unique values")
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
        return self._freq

    @freq.setter
    def freq(self, obj) -> None:
        """Set frequency from string, pd.Index, pd.offset, or forecaster.

        If the FH has _values_are_nanos=True (freq-less TimedeltaIndex),
        setting freq triggers conversion of nanosecond values to integer
        steps.

        If the FH already carries a frequency, the new frequency must match,
        otherwise a ValueError is raised.

        Parameters
        ----------
        obj : str, pd.Index, pd.Period, pd.Timestamp,
            pd.offsets.BaseOffset, or forecaster
            Object carrying frequency information.
            Frequency is extracted via ``PandasFHConverter.extract_freq``.
            Types that always carry freq (``pd.Period``, ``pd.PeriodIndex``,
            ``pd.offsets.BaseOffset``) will set the frequency.
            Types that may or may not carry freq (``pd.DatetimeIndex``,
            ``pd.TimedeltaIndex``) will set freq only if present.
            Types that never carry freq (``pd.Timestamp``, integer
            ``pd.Index``) are silently ignored (no-op).

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

        if new_freq is None:
            return

        old_freq = self._freq

        if old_freq is not None and old_freq != new_freq:
            raise ValueError(
                f"Frequencies do not match: current={old_freq!r}, new={new_freq!r}"
            )

        # if values are nanos, convert to steps using the new freq
        if self._values_are_nanos:
            new_values = PandasFHConverter.nanos_to_steps(self._values, new_freq)
            new_values.flags.writeable = False
            self._values = new_values
            self._values_are_nanos = False
            self._freq = new_freq
            return

        self._freq = new_freq

    # core conversion methods

    # <check>
    # for a drop-in replacement of the old FH,
    # we can allow users to call to_relative without cutoff
    # but the old FH raised an error at `cutoff is None`
    # Making it explicit in this new version is preferred.
    # Not providing cutoff will raise an error and is a breaking change
    # </check>
    def to_relative(self, cutoff=None):
        """Return relative version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value is required to convert an absolute forecasting
            horizon to a relative one.
            If pd.Index, last/latest value is considered the cutoff

        Returns
        -------
        ForecastingHorizon
            Relative representation of forecasting horizon.
        """
        if self._is_relative:
            return self._create(
                self._values.copy(),
                self._is_relative,
                self._freq,
                self._values_are_nanos,
            )

        if cutoff is None:
            raise ValueError(
                "`cutoff` must be provided to convert absolute FH to relative."
            )

        if self._values_are_nanos:
            raise ValueError(
                "Cannot convert to relative: values are raw nanoseconds "
                "pending freq assignment. Set freq first."
            )

        cutoff_step = PandasFHConverter.cutoff_to_steps(cutoff, freq=self._freq)
        ordinal_diffs = self._values - cutoff_step

        mult = PandasFHConverter.freq_multiplier(self._freq)
        if mult != 1:
            remainder = ordinal_diffs % mult
            if np.any(remainder != 0):
                raise ValueError(
                    f"FH values and cutoff are not on the same period grid "
                    f"for freq={self._freq!r}. Ordinal differences are not "
                    f"evenly divisible by the frequency multiplier {mult}."
                )
            relative_vals = ordinal_diffs // mult
        else:
            relative_vals = ordinal_diffs

        return self._create(
            values=relative_vals.astype(np.int64),
            is_relative=True,
            freq=self._freq,
        )

    def to_absolute(self, cutoff):
        """Return absolute version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index
            Cutoff value is required to convert a relative forecasting
            horizon to an absolute one.

        Returns
        -------
        ForecastingHorizon
            Absolute representation of forecasting horizon.
        """
        if not self._is_relative:
            return self._create(
                self._values.copy(),
                self._is_relative,
                self._freq,
                self._values_are_nanos,
            )

        if cutoff is None:
            raise ValueError(
                "`cutoff` must be provided to convert relative FH to absolute."
            )

        if self._values_are_nanos:
            # attempt to extract freq from cutoff for deferred conversion
            cutoff_freq = PandasFHConverter.extract_freq(cutoff)
            if cutoff_freq is not None:
                values = PandasFHConverter.nanos_to_steps(self._values, cutoff_freq)
                freq = cutoff_freq
            else:
                raise ValueError(
                    "Cannot convert to absolute: values are raw nanoseconds "
                    "and no freq is available. Set freq on the FH or provide "
                    "a cutoff with frequency information."
                )
        else:
            values = self._values
            freq = self._freq

        cutoff_step = PandasFHConverter.cutoff_to_steps(cutoff, freq=freq)
        mult = PandasFHConverter.freq_multiplier(freq)
        absolute_vals = cutoff_step + values * mult

        return self._create(
            values=absolute_vals.astype(np.int64),
            is_relative=False,
            freq=freq,
        )

    def to_pandas(self):
        """Return forecasting horizon values as pd.Index.

        Output type depends on the internal state of the FH:

        - ``values_are_nanos=True``: ``pd.TimedeltaIndex`` (raw
        nanoseconds from a freq-less TimedeltaIndex input).
        - ``is_relative=False`` and ``freq is not None``:
        ``pd.PeriodIndex`` (absolute period ordinals reconstructed
        with freq).
        - All other cases: plain ``pd.Index`` with integer dtype.
        This covers relative FH (with or without freq) and absolute
        integer FH without freq.

        Note: relative FH that originated from a TimedeltaIndex (with
        freq) is returned as integer Index, not TimedeltaIndex,
        because the original input type is not preserved after
        normalization to integer steps. For DatetimeIndex reconstruction
        and cutoff-aware absolute output, use ``to_absolute_index``
        with a cutoff.

        Returns
        -------
        pd.TimedeltaIndex, pd.PeriodIndex, or pd.Index
            Pandas Index matching the semantic state of the FH.
        """
        return PandasFHConverter.to_pandas_index(
            self._values, self._is_relative, self._freq, self._values_are_nanos
        )

    def to_numpy(self) -> np.ndarray:
        """Return forecasting horizon values as numpy array.

        Returns
        -------
        np.ndarray
            Numpy array of int64 values.
        """
        return self._values.copy()

    def to_absolute_index(self, cutoff=None):
        """Return absolute values as pandas Index.

        Output type is determined by the cutoff type:

        - ``pd.DatetimeIndex`` or ``pd.Timestamp`` cutoff: returns
        ``pd.DatetimeIndex``, with timezone from cutoff if present.
        - ``pd.PeriodIndex``, ``pd.Period``, or int cutoff: returns
        ``pd.PeriodIndex`` (if freq is set) or integer ``pd.Index``.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index
            Cutoff value for conversion. Required for relative FH.
            If the FH is already absolute, cutoff is only used to
            determine the output Index type.

        Returns
        -------
        pd.DatetimeIndex, pd.PeriodIndex, or pd.Index
            Absolute forecasting horizon as pandas Index.
        """
        abs_fh = self.to_absolute(cutoff)

        # if cutoff is DatetimeIndex or Timestamp, produce DatetimeIndex output
        if cutoff is not None and PandasFHConverter.cutoff_is_dti_ts(cutoff):
            tz = PandasFHConverter.cutoff_tz(cutoff)
            return PandasFHConverter.steps_to_datetime(
                abs_fh._values, abs_fh._freq, tz=tz
            )

        return abs_fh.to_pandas()

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
        absolute = self.to_absolute(cutoff)
        freq = absolute._freq

        # safeguard: if self already has freq, it must match absolute's freq.
        # They can only differ if to_absolute introduced a new freq (bug).
        # self._freq is None is fine — means freq came from cutoff via nanos path.
        if self._freq is not None and self._freq != freq:
            raise ValueError(
                f"Frequency mismatch after to_absolute: "
                f"self._freq={self._freq!r}, absolute._freq={freq!r}. "
                f"This should not happen — please report as a bug."
            )

        start_step = PandasFHConverter.cutoff_to_steps(start, freq=freq)
        ordinal_diffs = absolute._values - start_step

        mult = PandasFHConverter.freq_multiplier(freq)
        if mult != 1:
            remainder = ordinal_diffs % mult
            if np.any(remainder != 0):
                raise ValueError(
                    f"Start value and FH values are not on the same period "
                    f"grid for freq={freq!r}. Ordinal differences are not "
                    f"evenly divisible by the frequency multiplier {mult}."
                )
            integers = ordinal_diffs // mult
        else:
            integers = ordinal_diffs

        return self._create(
            values=integers.astype(np.int64),
            is_relative=False,
            freq=None,  # integer indices, not period ordinals
        )

    # In-sample and out-of-sample methods

    def _is_in_sample(self, cutoff=None) -> np.ndarray:
        """Return boolean array indicating in-sample values.

        In-sample values have relative representation <= 0.
        """
        relative = self.to_relative(cutoff)
        return relative._values <= 0

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
        return self._create(
            self._values[mask],
            self._is_relative,
            self._freq,
            self._values_are_nanos,
        )

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
        return self._create(
            self._values[mask],
            self._is_relative,
            self._freq,
            self._values_are_nanos,
        )

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
            indexer_vals = relative._values - 1
        else:
            relative = self.to_relative(cutoff)
            vals = relative._values
            indexer_vals = vals - vals[0]

        return PandasFHConverter.to_pandas_index(
            indexer_vals.astype(np.int64), is_relative=True
        )

    def _is_contiguous(self) -> bool:
        """Check if forecasting horizon values form a contiguous sequence.

        Returns
        -------
        bool
        """
        if len(self._values) <= 1:
            return True
        if self._values_are_nanos:
            # for nanos, check uniform spacing
            diffs = np.diff(self._values)
            return bool(np.all(diffs == diffs[0]))
        # for absolute FH with multi-step freq, ordinals have diffs of mult
        if not self._is_relative and self._freq is not None:
            mult = PandasFHConverter.freq_multiplier(self._freq)
        else:
            mult = 1
        expected_len = int(self._values[-1] - self._values[0]) // mult + 1
        return len(self._values) == expected_len

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

    @staticmethod
    def _check_scalar(other):
        if isinstance(other, ForecastingHorizon):
            raise TypeError(
                "Arithmetic between two ForecastingHorizon objects is not "
                "supported. Use scalar operands (int, np.integer)."
            )
        return np.int64(other)

    def __add__(self, other):
        scalar = self._check_scalar(other)
        if not self._is_relative and self._freq is not None:
            mult = PandasFHConverter.freq_multiplier(self._freq)
            result = self._values + scalar * mult
        else:
            result = self._values + scalar
        return self._create(
            result, self._is_relative, self._freq, self._values_are_nanos
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        scalar = self._check_scalar(other)
        if not self._is_relative and self._freq is not None:
            mult = PandasFHConverter.freq_multiplier(self._freq)
            result = self._values - scalar * mult
        else:
            result = self._values - scalar
        return self._create(
            result, self._is_relative, self._freq, self._values_are_nanos
        )

    def __mul__(self, other):
        if not self._is_relative and self._freq is not None:
            raise TypeError(
                "Multiplication is not supported for absolute "
                "ForecastingHorizon with frequency."
            )
        scalar = self._check_scalar(other)
        result = np.unique(self._values * scalar)
        return self._create(
            result, self._is_relative, self._freq, self._values_are_nanos
        )

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
            return (
                np.array_equal(self._values, other._values)
                and self._is_relative == other._is_relative
                and self._freq == other._freq
                and self._values_are_nanos == other._values_are_nanos
            )
        return self._values == np.int64(other)

    def __ne__(self, other):
        if isinstance(other, ForecastingHorizon):
            return not self.__eq__(other)
        return self._values != np.int64(other)

    def __lt__(self, other):
        self._check_scalar(other)
        return self._values < np.int64(other)

    def __le__(self, other):
        self._check_scalar(other)
        return self._values <= np.int64(other)

    def __gt__(self, other):
        self._check_scalar(other)
        return self._values > np.int64(other)

    def __ge__(self, other):
        self._check_scalar(other)
        return self._values >= np.int64(other)

    # Dunders -> container methods len, getitem, max, min
    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        result = self._values[key]
        if isinstance(result, np.ndarray):
            return self._create(
                result, self._is_relative, self._freq, self._values_are_nanos
            )
        return result

    def max(self):
        """Return the maximum value."""
        return self._values.max() if len(self._values) > 0 else None

    def min(self):
        """Return the minimum value."""
        return self._values.min() if len(self._values) > 0 else None

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
        return hash(
            (
                self._values.tobytes(),
                self._is_relative,
                self._freq,
                self._values_are_nanos,
            )
        )

    def __repr__(self):
        class_name = type(self).__name__
        n = len(self._values)
        parts = [f"n={n}", f"is_relative={self._is_relative}"]
        if self._freq is not None:
            parts.append(f"freq={self._freq!r}")
        if self._values_are_nanos:
            parts.append("values_are_nanos=True")
        if n <= 6:
            parts.append(f"values={self._values.tolist()}")
        else:
            head = self._values[:3].tolist()
            tail = self._values[-3:].tolist()
            parts.append(
                f"values=[{head[0]}, {head[1]}, {head[2]}, ..., "
                f"{tail[0]}, {tail[1]}, {tail[2]}]"
            )
        return f"{class_name}({', '.join(parts)})"

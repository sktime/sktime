# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Isolated pandas conversion layer for ForecastingHorizonV2.

ALL pandas-specific imports and logic live in this module.
The core _FHValues and ForecastingHorizonV2 classes should never import pandas directly,
they go through this converter instead.

This module handles:
Converting user-facing input types (int, list, pd.Index, etc.)
to the internal _FHValues representation.
Converting _FHValues back to pd.Index for interoperability with sktime.
Extracting and normalizing frequency strings from pandas objects.
Converting cutoff values from pandas types to _FHValues.
"""

___all__ = ["PandasFHConverter"]

import numpy as np
import pandas as pd

from sktime.forecasting.base._fh_values import FHValues, FHValueType


class PandasFHConverter:
    """Static conversion layer between pandas types and FHValues.

    This class collects all pandas-coupled logic in one place so that
    the rest of the ForecastingHorizonV2 code can remain pandas-free.
    """

    # input -> FHValues (internal representation) conversion
    @staticmethod
    def to_internal(values, freq=None) -> FHValues:
        """Convert user-facing input values to internal FHValues representation.

        Parameters
        ----------
        values : int, list, range, np.ndarray, pd.Index, pd.Timedelta,
                 pd.offsets.BaseOffset, or FHValues
            Forecasting horizon values in any supported format.
        freq : str or None, optional
            Frequency hint, used when it cannot be inferred from values.

        Returns
        -------
        FHValues
            Internal representation with int64 numpy array.

        Raises
        ------
        TypeError
            If ``values`` type is not supported.
        """
        # if already internal — return a defensive copy to avoid shared mutable state.
        # Without it, in-place changes to the internal numpy array
        # through one reference would silently affect the other.
        if isinstance(values, FHValues):
            return values.copy()

        # <check>
        # check for freq mismatch/incompatibility with values where possible,
        # and raise informative errors
        # Example: if freq is provided as "D" but values are datetime64 with
        # hourly frequency, that should raise an error about freq mismatch.
        # This is a non-trivial amount of logic to implement.
        # what needs to be handled explicitly here and what can be deferred to pandas
        # to raise errors - is under consideration, will revisit after initial
        # implementation of the main conversion logic.
        # Current implementation relies on pandas to raise errors
        # </check>

        # integer scalars and ranges: type is known,
        # convert directly to int64 numpy array without needing to inspect contents
        if isinstance(values, (int, np.integer)):
            arr = np.array([int(values)], dtype=np.int64)
            return FHValues(arr, FHValueType.INT, freq=freq)
        if isinstance(values, range):
            arr = np.array(list(values), dtype=np.int64)
            return FHValues(arr, FHValueType.INT, freq=freq)

        # for np.ndarray and list, we need to inspect the contents
        # to determine the value type and how to convert to int64 numpy array
        if isinstance(values, np.ndarray):
            return PandasFHConverter._ndarray_to_internal(values, freq)
        if isinstance(values, list):
            return PandasFHConverter._list_to_internal(values, freq)

        # pandas Timedelta and offset objects
        if isinstance(values, pd.Timedelta):
            arr = np.array([values.value], dtype=np.int64)
            # the above mentioned check would be performed here
            # and all such subsequent places in the below if-blocks
            freq_str = freq or PandasFHConverter._extract_freq_str(values)
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq_str)
        if isinstance(values, pd.offsets.BaseOffset):
            td = pd.Timedelta(values)
            arr = np.array([td.value], dtype=np.int64)
            freq_str = freq or PandasFHConverter._offset_to_freq_str(values)
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq_str)

        # pandas Index types (specific types checked before generic)
        if isinstance(values, pd.PeriodIndex):
            arr = values.asi8.copy()
            freq_str = freq or PandasFHConverter._freqstr(values)
            return FHValues(arr, FHValueType.PERIOD, freq=freq_str)
        if isinstance(values, pd.DatetimeIndex):
            arr = values.asi8.copy()
            freq_str = freq or PandasFHConverter._freqstr(values)
            tz = str(values.tz) if values.tz is not None else None
            return FHValues(arr, FHValueType.DATETIME, freq=freq_str, timezone=tz)
        if isinstance(values, pd.TimedeltaIndex):
            arr = values.asi8.copy()
            freq_str = freq or PandasFHConverter._freqstr(values)
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq_str)
        if isinstance(values, pd.RangeIndex):
            arr = values.to_numpy().astype(np.int64)
            return FHValues(arr, FHValueType.INT, freq=freq)

        # generic pd.Index - convert based on dtype
        if isinstance(values, pd.Index):
            if pd.api.types.is_integer_dtype(values.dtype):
                # for integer dtype, convert directly to int64 numpy array
                # using pandas' to_numpy
                arr = values.to_numpy().astype(np.int64)
                return FHValues(arr, FHValueType.INT, freq=freq)
            # timedelta-like elements in a generic Index
            if pd.api.types.is_timedelta64_dtype(values.dtype):
                arr = values.to_numpy().view(np.int64).copy()
                return FHValues(arr, FHValueType.TIMEDELTA, freq=freq)
            raise TypeError(
                f"pd.Index with dtype {values.dtype} is not supported. "
                f"Expected integer or timedelta dtype."
            )

        # if no match till this point, the type is not supported
        raise TypeError(
            f"Unsupported type for forecasting horizon values: "
            f"{type(values).__name__}. Expected int, list, range, np.ndarray, "
            f"pd.Index, pd.Timedelta, or pd.offsets.BaseOffset."
        )

    @staticmethod
    def _ndarray_to_internal(values: np.ndarray, freq=None) -> FHValues:
        """Convert 1-D numpy array (integer, timedelta64, or datetime64) to FHValues."""
        if values.ndim != 1:
            raise ValueError(f"Expected 1-D array, got {values.ndim}-D array")
        # empty array check
        if len(values) == 0:
            raise ValueError("Forecasting horizon values must not be empty.")

        # <check>
        # check for freq mismatch
        # </check>

        # timedelta64 and datetime64 checked before integer as a defensive measure
        # as some numpy versions might consider datetime64 as a subtype of integer,
        # which might cause incorrect classification.
        # This ordering keeps the type checking explicit and
        # avoids any future dtype hierarchy surprises
        if np.issubdtype(values.dtype, np.timedelta64):
            # Convert to ns resolution first, then view as int64
            arr = values.astype("timedelta64[ns]").view(np.int64).copy()
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq)

        if np.issubdtype(values.dtype, np.datetime64):
            arr = values.astype("datetime64[ns]").view(np.int64).copy()
            return FHValues(arr, FHValueType.DATETIME, freq=freq)

        if np.issubdtype(values.dtype, np.integer):
            arr = values.astype(np.int64).copy()
            return FHValues(arr, FHValueType.INT, freq=freq)

        raise TypeError(
            f"np.ndarray with dtype {values.dtype} is not supported. "
            f"Expected integer, timedelta64, or datetime64 dtype."
        )

    @staticmethod
    def _list_to_internal(values: list, freq=None) -> FHValues:
        """Convert list of supported scalar types to FHValues."""
        from datetime import timedelta as _timedelta

        if len(values) == 0:
            raise ValueError("Forecasting horizon values must not be empty.")

        # <check>
        # check for freq mismatch
        # </check>

        # timedelta-like checked before int
        # see comment in _ndarray_to_internal for rationale

        # pd.Timedelta, np.timedelta64, and stdlib datetime.timedelta are combined
        # into a single if-case because they all represent the same concept and
        # pd.TimedeltaIndex accepts all three, so mixed lists like
        # [pd.Timedelta("1D"), timedelta(days=2)] work naturally.
        _timedelta_types = (pd.Timedelta, np.timedelta64, _timedelta)
        if isinstance(values[0], _timedelta_types):
            PandasFHConverter._check_list_homogeneity(values, _timedelta_types)
            idx = pd.TimedeltaIndex(values)
            arr = idx.asi8.copy()
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq)

        # integer values — convert directly to int64 array
        if isinstance(values[0], (int, np.integer)):
            PandasFHConverter._check_list_homogeneity(values, (int, np.integer))
            arr = np.array(values, dtype=np.int64)
            return FHValues(arr, FHValueType.INT, freq=freq)

        # period values — extract ordinals via PeriodIndex
        if isinstance(values[0], pd.Period):
            PandasFHConverter._check_list_homogeneity(values, pd.Period)
            idx = pd.PeriodIndex(values)
            arr = idx.asi8.copy()
            freq_str = freq or PandasFHConverter._freqstr(idx)
            return FHValues(arr, FHValueType.PERIOD, freq=freq_str)

        # timestamp values — extract nanoseconds via DatetimeIndex
        if isinstance(values[0], pd.Timestamp):
            PandasFHConverter._check_list_homogeneity(values, pd.Timestamp)
            idx = pd.DatetimeIndex(values)
            arr = idx.asi8.copy()
            freq_str = freq or PandasFHConverter._freqstr(idx)
            tz = str(idx.tz) if idx.tz is not None else None
            return FHValues(arr, FHValueType.DATETIME, freq=freq_str, timezone=tz)

        # offset objects — convert to Timedelta first, then to nanoseconds
        if isinstance(values[0], pd.offsets.BaseOffset):
            PandasFHConverter._check_list_homogeneity(values, pd.offsets.BaseOffset)
            tds = [pd.Timedelta(v) for v in values]
            idx = pd.TimedeltaIndex(tds)
            arr = idx.asi8.copy()
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq)

        raise TypeError(
            f"List with element type {type(values[0]).__name__} is not supported."
        )

    # FHValues (internal representation) -> pandas conversion
    @staticmethod
    def to_pandas_index(fhv: "FHValues") -> pd.Index:
        """Convert internal FHValues to pandas Index.

        Parameters
        ----------
        fhv : FHValues
            Internal representation.

        Returns
        -------
        pd.Index
            Pandas Index matching the semantic type.
        """
        vtype = fhv.value_type
        vals = fhv.values  # read-only view, int64

        if vtype == FHValueType.INT:
            return pd.Index(vals.copy(), dtype=int)

        if vtype == FHValueType.PERIOD:
            # PeriodIndex from ordinals requires a writable copy
            return pd.PeriodIndex.from_ordinals(vals.copy(), freq=fhv.freq)

        if vtype == FHValueType.DATETIME:
            dt_arr = vals.copy().view("datetime64[ns]")
            idx = pd.DatetimeIndex(dt_arr)
            if fhv.timezone is not None:
                idx = idx.tz_localize("UTC").tz_convert(fhv.timezone)
            return idx

        if vtype == FHValueType.TIMEDELTA:
            td_arr = vals.copy().view("timedelta64[ns]")
            return pd.TimedeltaIndex(td_arr)

        # control should never reach here due to FHValueType validation in FHValues
        # if it does, it indicates a bug in FHValues or a missing case in this function
        raise ValueError(f"Unknown FHValueType: {vtype}")

    # cutoff conversion
    @staticmethod
    def cutoff_to_internal(cutoff, freq=None):
        """Convert cutoff to internal representation.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, pd.Index, or np.integer
            Cutoff value. If pd.Index, the last element is used.
        freq : str or None
            Frequency hint.

        Returns
        -------
        tuple of (np.int64, FHValueType, str or None, str or None)
            (value, value_type, freq, timezone)
        """
        pass

    @staticmethod
    def cutoff_to_pandas(cutoff_internal):
        pass

    @staticmethod
    def steps_to_nanos(steps: np.ndarray, freq: str, ref_nanos=None) -> np.ndarray:
        """Convert integer steps to int64 nanosecond offsets.

        Parameters
        ----------
        steps : np.ndarray of int64
            Integer step counts.
        freq : str
            Frequency string (e.g. "D", "h", "M").
        ref_nanos : int or np.int64 or None, default=None
            Reference point as nanoseconds since Unix epoch.
            Used to correctly compute offsets for variable-length periods
            (months, years). If None, uses 2000-01-01 as reference.

        Returns
        -------
        np.ndarray of int64
            Nanosecond offsets corresponding to each step.
        """
        pass

    # frequency helper functions

    # 1. frequency extraction function,
    #    to get freq from pandas objects when needed
    # 2. frequency normalization function,
    #    to convert pandas freq strings to a canonical form
    # 3. offset handler
    # coerce function, a pandas-aware wrapper around FHValues
    # coercion that can handle pandas types as input

    @staticmethod
    def normalize_freq(freq_str: str | None) -> str | None:
        """Normalize frequency string.

        Handles pandas frequency alias changes (e.g. "ME" -> "M").

        Parameters
        ----------
        freq_str : str or None
            Raw frequency string.

        Returns
        -------
        str or None
            Normalized frequency string.
        """
        if freq_str is None:
            return None
        # <check>
        # 1. check for unsupported frequencies and raise informative errors
        # 2. check for completeness of the alias map and add any missing aliases
        # 3. is there way to leverage pandas frequency parsing/normalization logic
        #    instead of hardcoding an alias map here?
        # 4. if we keep the alias map, make it more comprehensive and robust,
        #    and add tests for it.
        #    For example, handling both "M" and "ME" for month-end frequencies,
        #    and ensuring that all common aliases are covered.
        # 5. whether to handle frequency strings in a case-insensitive manner,
        #    e.g. treating "m" and "M" as the same frequency,
        #    and whether to add checks for that.
        # 6. whether to use pandas.tseries.frequencies.to_offset to validate and
        #    normalize freq strings, which would leverage pandas'
        #    internal logic and ensure consistency with pandas behavior.
        #    one way of achieving 1. and 2. without hardcoding an alias map
        #    if it succeeds,
        #    use the resulting offset's name as the normalized freq string.
        # 7. Make the alias map a frozenset to prevent accidental modifications
        # </check>
        alias_map = {
            "ME": "M",
            "QE": "Q",
            "YE": "Y",
            "BQE": "BQ",
            "BYE": "BY",
            "SME": "SM",
        }
        return alias_map.get(freq_str, freq_str)

    # below function is directly moved from ForecastingHorizonV2.get_expected_pred_idx()
    # to avoid pandas imports in ForecastingHorizonV2
    # it may contain some parts/checks which might require adjustments after the move
    @staticmethod
    def build_pred_index(fh, y=None, cutoff=None, sort_by_time=False):
        """Construct expected prediction output index.

        This contains all pandas-dependent logic for building prediction
        indices, including MultiIndex/DataFrame handling. Called by
        ForecastingHorizonV2.get_expected_pred_idx().

        Parameters
        ----------
        fh : ForecastingHorizonV2
            Forecasting horizon instance.
        y : pd.DataFrame, pd.Series, pd.Index, or None (default=None)
            Data to compute fh relative to.
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value. If None, inferred from ``y``.
        sort_by_time : bool, optional (default=False)
            For MultiIndex returns, whether to sort by time index.

        Returns
        -------
        pd.Index
            Expected index of y_pred returned by predict.
        """
        from sktime.datatypes import get_cutoff

        if hasattr(y, "index"):
            y_index = y.index
        elif isinstance(y, pd.Index):
            y_index = y
            y = pd.DataFrame(index=y_index)
        elif cutoff is None and y is not None:
            y_index = pd.Index(y)
            y = pd.DataFrame(index=y_index)
        else:
            y_index = None

        # MultiIndex case: compute per-instance cutoffs and predictions
        if y_index is not None and isinstance(y_index, pd.MultiIndex):
            y_inst_idx = y_index.droplevel(-1).unique()

            def _per_instance_pred(inst_key):
                """Get absolute FH for a single instance."""
                inst_cutoff = get_cutoff(y.loc[inst_key])
                return fh.to_absolute_index(inst_cutoff)

            if cutoff is not None:
                # Global cutoff provided: use global absolute FH for all
                y_inst_idx = y_inst_idx.sort_values()
                fh_abs_idx = fh.to_absolute_index(cutoff)
                if isinstance(y_inst_idx, pd.MultiIndex) and sort_by_time:
                    fh_list = [x + (t,) for t in fh_abs_idx for x in y_inst_idx]
                elif isinstance(y_inst_idx, pd.MultiIndex):
                    fh_list = [x + (t,) for x in y_inst_idx for t in fh_abs_idx]
                elif sort_by_time:
                    fh_list = [(x, t) for t in fh_abs_idx for x in y_inst_idx]
                else:
                    fh_list = [(x, t) for x in y_inst_idx for t in fh_abs_idx]
            else:
                # Per-instance cutoffs
                fh_list = []
                for inst_key in y_inst_idx:
                    inst_abs = _per_instance_pred(inst_key)
                    if isinstance(y_inst_idx, pd.MultiIndex):
                        fh_list.extend([inst_key + (t,) for t in inst_abs])
                    else:
                        fh_list.extend([(inst_key, t) for t in inst_abs])

            fh_idx = pd.Index(fh_list)

            if sort_by_time and cutoff is None:
                fh_df = pd.DataFrame(index=fh_idx)
                fh_idx = fh_df.sort_index(level=-1).index

            # replicate index names
            if y_index.names is not None:
                fh_idx.names = y_index.names

            return fh_idx

        # non-MultiIndex case
        if cutoff is None and y_index is not None:
            cutoff = get_cutoff(y)

        fh_abs_idx = fh.to_absolute_index(cutoff)

        # replicate index names
        if y_index is not None and y_index.names is not None:
            fh_abs_idx.names = y_index.names

        return fh_abs_idx

    # private helper functions
    @staticmethod
    def _check_list_homogeneity(values, expected_types):
        """Check all list elements match expected types.

        Parameters
        ----------
        values : list
            List of values to check. Must be non-empty.
        expected_types : type or tuple of types
            Accepted types for isinstance check.

        Raises
        ------
        TypeError
            If any element does not match expected_types.
        """
        # starting from index 1 since the
        # check for first element's type against expected_types
        # is done in the caller before this function is called
        for i, v in enumerate(values[1:], start=1):
            if not isinstance(v, expected_types):
                raise TypeError(
                    f"Element at index 0 is of type {type(values[0]).__name__}, "
                    f"but element at index {i} is {type(v).__name__}. "
                    "All list elements must be of the same type."
                )

    @staticmethod
    def _extract_freq_str(obj) -> str | None:
        """Extract freq string from scalar pandas time objects."""
        if hasattr(obj, "freq") and obj.freq is not None:
            return PandasFHConverter.normalize_freq(str(obj.freq))
        return None

    @staticmethod
    def _offset_to_freq_str(offset) -> str | None:
        """Extract freq string from a pandas offset object."""
        if hasattr(offset, "freqstr"):
            return PandasFHConverter.normalize_freq(offset.freqstr)
        return None

    @staticmethod
    def _freqstr(idx) -> str | None:
        """Extract normalized frequency string from a pandas Index."""
        if hasattr(idx, "freq") and idx.freq is not None:
            return PandasFHConverter.normalize_freq(idx.freqstr)
        return None

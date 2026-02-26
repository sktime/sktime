# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Isolated pandas conversion layer for ForecastingHorizon.

ALL pandas-specific imports and logic live in this module.
The core _FHValues and ForecastingHorizon classes should never import pandas directly,
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
    the rest of the ForecastingHorizon code can remain pandas-free.
    """

    # input -> FHValues (internal representation) conversion
    @staticmethod
    def to_internal(values) -> FHValues:
        """Convert pandas input values to internal FHValues representation.

        Frequency is inferred from values only. The ``freq`` parameter on
        ``ForecastingHorizon`` is handled separately by the ``freq`` setter,
        which is the single gate for all frequency setting and validation.

        Parameters
        ----------
        values : pandas type or list of pandas scalars
            Forecasting horizon values in a pandas-specific format.
            Supported types:
            - ``pd.PeriodIndex`` : converted to PERIOD ordinals
            - ``pd.DatetimeIndex`` : converted to DATETIME nanoseconds
            - ``pd.TimedeltaIndex`` : converted to TIMEDELTA nanoseconds
            - ``pd.RangeIndex`` : converted to INT
            - ``pd.Index`` with integer or timedelta64 dtype
            - ``pd.Timedelta`` : single timedelta scalar
            - ``pd.offsets.BaseOffset`` : single offset scalar
            - ``list`` of ``pd.Period``, ``pd.Timestamp``, ``pd.Timedelta``,
              ``pd.offsets.BaseOffset``, ``np.timedelta64``, or
              ``datetime.timedelta`` scalars

        Returns
        -------
        FHValues
            Internal representation with int64 numpy array.

        Raises
        ------
        TypeError
            If ``values`` type is not supported.
        """
        # pandas Timedelta and offset objects
        if isinstance(values, pd.Timedelta):
            arr = np.array([values.value], dtype=np.int64)
            freq_str = PandasFHConverter._extract_freq_str(values)
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq_str)
        if isinstance(values, pd.offsets.BaseOffset):
            td = pd.Timedelta(values)
            arr = np.array([td.value], dtype=np.int64)
            freq_str = PandasFHConverter._offset_to_freq_str(values)
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq_str)

        # pandas Index types (specific types checked before generic)
        if isinstance(values, pd.PeriodIndex):
            arr = values.asi8.copy()
            freq_str = PandasFHConverter._freqstr(values)
            return FHValues(arr, FHValueType.PERIOD, freq=freq_str)
        if isinstance(values, pd.DatetimeIndex):
            arr = values.asi8.copy()
            freq_str = PandasFHConverter._freqstr(values)
            tz = str(values.tz) if values.tz is not None else None
            return FHValues(arr, FHValueType.DATETIME, freq=freq_str, timezone=tz)
        if isinstance(values, pd.TimedeltaIndex):
            arr = values.asi8.copy()
            freq_str = PandasFHConverter._freqstr(values)
            return FHValues(arr, FHValueType.TIMEDELTA, freq=freq_str)
        if isinstance(values, pd.RangeIndex):
            arr = values.to_numpy().astype(np.int64)
            return FHValues(arr, FHValueType.INT)

        # generic pd.Index - convert based on dtype
        if isinstance(values, pd.Index):
            if pd.api.types.is_integer_dtype(values.dtype):
                arr = values.to_numpy().astype(np.int64)
                return FHValues(arr, FHValueType.INT)
            if pd.api.types.is_timedelta64_dtype(values.dtype):
                arr = values.to_numpy().view(np.int64).copy()
                return FHValues(arr, FHValueType.TIMEDELTA)
            raise TypeError(
                f"pd.Index with dtype {values.dtype} is not supported. "
                f"Expected integer or timedelta dtype."
            )

        # lists of pandas/timedelta scalars
        if isinstance(values, list):
            return PandasFHConverter._list_to_internal(values)

        # if no match, the type is not supported
        raise TypeError(
            f"Unsupported type for forecasting horizon values: "
            f"{type(values).__name__}. When passing pandas objects, "
            f"following are expected types: pd.PeriodIndex, "
            f"pd.DatetimeIndex, pd.TimedeltaIndex, pd.RangeIndex, "
            f"pd.Index (integer or timedelta dtype), pd.Timedelta, "
            f"pd.offsets.BaseOffset, or list of pandas scalars."
        )

    @staticmethod
    def _list_to_internal(values: list) -> FHValues:
        """Convert list of supported scalar types to FHValues."""
        from datetime import timedelta as _timedelta

        if len(values) == 0:
            raise ValueError("Forecasting horizon values must not be empty.")

        # pd.Timedelta, np.timedelta64, and stdlib datetime.timedelta are
        # combined into a single case because they all represent the same
        # concept and pd.TimedeltaIndex accepts all three.
        _timedelta_types = (pd.Timedelta, np.timedelta64, _timedelta)
        if isinstance(values[0], _timedelta_types):
            PandasFHConverter._check_list_homogeneity(values, _timedelta_types)
            idx = pd.TimedeltaIndex(values)
            arr = idx.asi8.copy()
            return FHValues(arr, FHValueType.TIMEDELTA)

        # period values — extract ordinals via PeriodIndex
        if isinstance(values[0], pd.Period):
            PandasFHConverter._check_list_homogeneity(values, pd.Period)
            idx = pd.PeriodIndex(values)
            arr = idx.asi8.copy()
            freq_str = PandasFHConverter._freqstr(idx)
            return FHValues(arr, FHValueType.PERIOD, freq=freq_str)

        # timestamp values — extract nanoseconds via DatetimeIndex
        if isinstance(values[0], pd.Timestamp):
            PandasFHConverter._check_list_homogeneity(values, pd.Timestamp)
            idx = pd.DatetimeIndex(values)
            arr = idx.asi8.copy()
            freq_str = PandasFHConverter._freqstr(idx)
            tz = str(idx.tz) if idx.tz is not None else None
            return FHValues(arr, FHValueType.DATETIME, freq=freq_str, timezone=tz)

        # offset objects — convert to Timedelta first
        if isinstance(values[0], pd.offsets.BaseOffset):
            PandasFHConverter._check_list_homogeneity(values, pd.offsets.BaseOffset)
            tds = [pd.Timedelta(v) for v in values]
            idx = pd.TimedeltaIndex(tds)
            arr = idx.asi8.copy()
            return FHValues(arr, FHValueType.TIMEDELTA)

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

    # final check pending for this function
    @staticmethod
    def extract_freq(obj) -> str | None:
        """Extract and normalize a frequency string from a pandas object.

        Handles pd.Index, pd.Period, pd.offsets.BaseOffset, strings,
        and objects with a ``cutoff`` attribute (e.g. sktime forecasters).

        Parameters
        ----------
        obj : pd.Index, pd.Period, pd.offsets.BaseOffset, str, or forecaster
            Object carrying frequency information.

        Returns
        -------
        str or None
            Normalized frequency string, or None if no frequency
            could be extracted.
        """
        if isinstance(obj, str):
            try:
                from pandas.tseries.frequencies import to_offset

                offset = to_offset(obj)
                if offset is not None:
                    return PandasFHConverter.normalize_freq(str(offset.freqstr))
            except ValueError:
                return None
            return None

        if isinstance(obj, pd.offsets.BaseOffset):
            return PandasFHConverter.normalize_freq(obj.freqstr)

        if hasattr(obj, "cutoff"):
            # sktime forecasters: extract freq from cutoff attribute
            return PandasFHConverter.extract_freq(obj.cutoff)

        if isinstance(obj, pd.Period):
            return PandasFHConverter.normalize_freq(obj.freqstr)

        if isinstance(obj, (pd.PeriodIndex, pd.DatetimeIndex)):
            return PandasFHConverter._freqstr(obj)

        if isinstance(obj, pd.Index):
            # generic pd.Index — no freq attribute
            return None

        return None

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

    # below function is directly moved from ForecastingHorizon.get_expected_pred_idx()
    # to avoid pandas imports in ForecastingHorizon
    # it may contain some parts/checks which might require adjustments after the move
    @staticmethod
    def build_pred_index(fh, y=None, cutoff=None, sort_by_time=False):
        """Construct expected prediction output index.

        This contains all pandas-dependent logic for building prediction
        indices, including MultiIndex/DataFrame handling. Called by
        ForecastingHorizon.get_expected_pred_idx().

        Parameters
        ----------
        fh : ForecastingHorizon
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

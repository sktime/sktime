# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Isolated pandas conversion layer for ForecastingHorizon.

ALL pandas-specific imports and logic live in this module.
The core ForecastingHorizon class should never import pandas directly,
it goes through this converter instead.

All temporal inputs are normalized to integer steps (period ordinals) at construction.
The converter handles:
- Converting user-facing pandas types to integer steps + metadata
- Converting integer steps back to pandas Index for output
- Extracting and normalizing frequency strings from pandas objects
- Converting cutoff values to integer steps
"""

___all__ = ["PandasFHConverter", "_PANDAS_FH_INPUT_TYPES"]

import warnings

import numpy as np
import pandas as pd

_PANDAS_FH_INPUT_TYPES = pd.Index

from sktime.forecasting.base._freq_mnemonic import (
    _ALIAS_TO_CANONICAL_STATIC,
    _FREQ_GROUPS,
    VALID_FREQ_BASES,
    _parse_freq,
)

# Frequency resolution system
#
# Note: pandas has TWO freq string APIs with DIFFERENT preferences:
#
# - Period-context APIs (.to_period, PeriodIndex.from_ordinals, .asfreq):
#   Want the OLD form: "M", "Q", "Y", etc.
#   Reject the new form: PeriodIndex.from_ordinals(v, freq="ME") raises.
#
# - Offset-context APIs (to_offset):
#  Want the NEW form: "ME", "QE", "YE", etc.
#  Emit FutureWarning on old form: to_offset("M") warns.
#
# Our canonical freq strings (VALID_FREQ_BASES: "M", "D", "h", etc.)
# align with Period-context APIs, so those calls can use canonical directly.
# Only Offset-context calls (to_offset) need _resolve_pandas_freq wrapping.
#
# Three auto-populated caches that bridge our canonical freq names
# and pandas's changing aliases. The offset TYPE (MonthEnd, Day, etc.)
# is the stable anchor — pandas changes string aliases but not the
# offset class hierarchy.
#
# When pandas renames "M" -> "ME" -> "MO" (hypothetical), the offset
# type stays MonthEnd, and our canonical stays "M". The caches
# auto-discover what pandas currently prefers via to_offset().

# Any freq base string -> our canonical base
# seeded statically from _FREQ_GROUPS, extended dynamically.
_ALIAS_TO_CANONICAL = dict(_ALIAS_TO_CANONICAL_STATIC)
for _base in VALID_FREQ_BASES:
    _ALIAS_TO_CANONICAL.setdefault(_base, _base)

# pandas offset type -> our canonical base. Populated by _register_freq.
_OFFSET_TYPE_TO_CANONICAL = {}

# Our canonical base -> pandas's currently preferred base string.
# Populated by _bootstrap_pandas_prefs.
_CANONICAL_TO_PANDAS = {}

# Full freq string cache: any freq string -> pandas-accepted freq string.
_RESOLVE_CACHE = {}


def _register_freq(base_str):
    """Discover offset type and pandas preferred form for a base string.

    Calls ``to_offset()`` with FutureWarning suppressed. Populates
    ``_OFFSET_TYPE_TO_CANONICAL`` and ``_CANONICAL_TO_PANDAS``.

    Parameters
    ----------
    base_str : str
        A frequency base string (without multiplier or suffix).
    """
    from pandas.tseries.frequencies import to_offset

    canonical = _ALIAS_TO_CANONICAL.get(base_str, base_str)
    if canonical in _CANONICAL_TO_PANDAS:
        return  # already have preferred form for this canonical

    # Collect all aliases to try (canonical first, then known aliases)
    aliases_to_try = {canonical}
    for group in _FREQ_GROUPS:
        if canonical in group or base_str in group:
            aliases_to_try.update(group)
            break
    aliases_to_try = [canonical] + sorted(aliases_to_try - {canonical})

    for alias in aliases_to_try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                offset = to_offset(alias)
        except ValueError:
            continue

        offset_type = type(offset)
        pref_base = _parse_freq(offset.freqstr)[1]

        _OFFSET_TYPE_TO_CANONICAL[offset_type] = canonical
        _CANONICAL_TO_PANDAS[canonical] = pref_base
        _ALIAS_TO_CANONICAL.setdefault(pref_base, canonical)
        return


def _bootstrap_pandas_prefs():
    """Pre-populate caches by discovering pandas preferences for all bases."""
    all_canonicals = set(_ALIAS_TO_CANONICAL.values())
    for canonical in all_canonicals:
        if canonical not in _CANONICAL_TO_PANDAS:
            _register_freq(canonical)


# Run bootstrap at module load
_bootstrap_pandas_prefs()


def _normalize_freq(freq_str):
    """Normalize freq string to our canonical form.

    Parses the freq string, maps the base to canonical via
    ``_ALIAS_TO_CANONICAL``, reconstructs with original multiplier
    and suffix. Handles multiplied (``"2ME"`` -> ``"2M"``) and
    anchored (``"QE-DEC"`` -> ``"Q-DEC"``) forms.

    For unknown bases, attempts dynamic discovery via
    ``_register_freq``.

    Parameters
    ----------
    freq_str : str or None
        Frequency string to normalize.

    Returns
    -------
    str or None
        Normalized frequency string, or None if input is None.
    """
    if freq_str is None:
        return None
    mult, base, suffix = _parse_freq(freq_str)
    if base not in _ALIAS_TO_CANONICAL:
        _register_freq(base)
    canonical_base = _ALIAS_TO_CANONICAL.get(base, base)
    return f"{mult}{canonical_base}{suffix}"


def _resolve_pandas_freq(freq_str):
    """Resolve freq string to a form pandas currently accepts.

    Normalizes to canonical, then maps to pandas preferred form.
    Handles multiplied and anchored forms.

    Parameters
    ----------
    freq_str : str or None
        Frequency string (canonical or alias).

    Returns
    -------
    str or None
        Pandas-accepted frequency string, or None if input is None.
    """
    if freq_str is None:
        return None
    if freq_str in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[freq_str]

    mult, base, suffix = _parse_freq(freq_str)
    canonical_base = _ALIAS_TO_CANONICAL.get(base, base)

    if canonical_base not in _CANONICAL_TO_PANDAS:
        _register_freq(canonical_base)

    pandas_base = _CANONICAL_TO_PANDAS.get(canonical_base, canonical_base)
    resolved = f"{mult}{pandas_base}{suffix}"
    _RESOLVE_CACHE[freq_str] = resolved
    return resolved


class PandasFHConverter:
    """Static conversion layer between pandas types and ForecastingHorizon.

    This class collects all pandas-coupled logic in one place so that
    the rest of the ForecastingHorizon code can remain pandas-free.
    All methods are stateless.
    """

    # input -> internal representation conversion

    @staticmethod
    def to_internal(values, freq=None, is_relative=None) -> tuple:
        """Convert pandas input values to internal representation.

        All temporal inputs are normalized to integer steps:

        - PeriodIndex: .asi8 gives period ordinals (already integer steps)
        - DatetimeIndex with freq: .to_period(freq).asi8 gives ordinals
        - DatetimeIndex without freq: uses user-provided ``freq`` fallback if provided,
        otherwise raises ValueError
        - TimedeltaIndex with freq: timedelta / freq_timedelta gives steps
        - TimedeltaIndex without freq: stores nanoseconds with
        values_are_nanos=True (deferred conversion)

        The converter infers is_relative from the input type as per the below rules
        and sends back the infered_is_relative:

        - PeriodIndex, DatetimeIndex -> absolute (is_relative=False)
        - TimedeltaIndex -> relative (is_relative=True)
        - RangeIndex, integer Index -> relative (is_relative=True)

        The ``freq`` parameter on ``ForecastingHorizon`` is handled separately by the
        ``freq`` setter,
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
        freq : str, pd.Period, pd.Index, or None, default=None
            Optional fallback frequency. Used when ``values`` is a
            DatetimeIndex without freq. Extracted via ``extract_freq``.
        is_relative : bool or None, default=None
            User-supplied relativity flag, passed through from the
            ``ForecastingHorizon`` constructor. Only consulted for
            ``pd.RangeIndex`` and ``pd.Index`` with integer dtype, since
            these types are ambiguous (compatible with both relative and
            absolute). For these two cases, if provided, the value is
            returned as the inferred is_relative; if None, defaults to True.
            All other input types (``pd.PeriodIndex``, ``pd.DatetimeIndex``,
            ``pd.TimedeltaIndex``, ``pd.Index`` with timedelta dtype, scalar
            and list inputs) have unambiguous semantics and ignore this
            parameter entirely.

        Returns
        -------
        tuple of (np.ndarray, bool, str or None, bool)
            (values, is_relative, freq, values_are_nanos).

        Raises
        ------
        TypeError
            If ``values`` type is not supported.
        ValueError
            If DatetimeIndex is provided without freq and no fallback.
        """
        # pandas Timedelta scalar
        if isinstance(values, pd.Timedelta):
            freq_str = PandasFHConverter._extract_freq_str(values)
            if freq_str is not None:
                steps = PandasFHConverter.nanos_to_steps(
                    np.array([values.value], dtype=np.int64), freq_str
                )
                return (steps, True, freq_str, False)
            else:
                arr = np.array([values.value], dtype=np.int64)
                return (arr, True, None, True)

        # pandas offset scalar
        if isinstance(values, pd.offsets.BaseOffset):
            td = pd.Timedelta(values)
            freq_str = PandasFHConverter._offset_to_freq_str(values)
            if freq_str is not None:
                steps = PandasFHConverter.nanos_to_steps(
                    np.array([td.value], dtype=np.int64), freq_str
                )
                return (steps, True, freq_str, False)
            else:
                arr = np.array([td.value], dtype=np.int64)
                return (arr, True, None, True)

        # PeriodIndex -> ordinals (already integer steps)
        if isinstance(values, pd.PeriodIndex):
            arr = values.asi8.copy()
            freq_str = PandasFHConverter._freqstr(values)
            return (arr, False, freq_str, False)

        # DatetimeIndex -> convert to period ordinals
        if isinstance(values, pd.DatetimeIndex):
            freq_str = PandasFHConverter._freqstr(values)
            if freq_str is None and freq is not None:
                freq_str = PandasFHConverter.extract_freq(freq)
            elif freq_str is not None and freq is not None:
                # Sparse DatetimeIndex (e.g. 2 elements from fh=[2,5] on daily
                # data) can have pandas-inferred freq that is a multiple of the
                # true data freq (e.g. '3D' vs 'D'). When an explicit freq is
                # provided and shares the same base and the inferred freq is an
                # exact multiple, prefer the explicit (narrower) freq.
                explicit_freq = PandasFHConverter.extract_freq(freq)
                if explicit_freq is not None and freq_str != explicit_freq:
                    _, inf_base, inf_sfx = _parse_freq(freq_str)
                    _, exp_base, exp_sfx = _parse_freq(explicit_freq)
                    inf_mult = PandasFHConverter.freq_multiplier(freq_str)
                    exp_mult = PandasFHConverter.freq_multiplier(explicit_freq)
                    if (
                        inf_base == exp_base
                        and inf_sfx == exp_sfx
                        and inf_mult % exp_mult == 0
                    ):
                        freq_str = explicit_freq
                    else:
                        raise ValueError(
                            f"DatetimeIndex freq {freq_str!r} conflicts with "
                            f"explicit freq {explicit_freq!r} and cannot be "
                            f"resolved (different base or non-divisible "
                            f"multiple)."
                        )
            if freq_str is None:
                raise ValueError(
                    "DatetimeIndex without freq is not supported. "
                    "Provide freq explicitly via "
                    "ForecastingHorizon(values, freq=...) or use a "
                    "DatetimeIndex with freq set."
                )
            arr = values.to_period(freq_str).asi8.copy()
            return (arr, False, freq_str, False)

        # TimedeltaIndex -> steps (with freq) or nanos (without freq)
        if isinstance(values, pd.TimedeltaIndex):
            freq_str = PandasFHConverter._freqstr(values)
            if freq_str is not None:
                steps = PandasFHConverter.nanos_to_steps(values.asi8.copy(), freq_str)
                return (steps, True, freq_str, False)
            else:
                arr = values.asi8.copy()
                return (arr, True, None, True)

        # RangeIndex -> plain integers
        if isinstance(values, pd.RangeIndex):
            arr = values.to_numpy().astype(np.int64)
            is_rel = is_relative if is_relative is not None else True
            return (arr, is_rel, None, False)

        # generic pd.Index
        if isinstance(values, pd.Index):
            if pd.api.types.is_integer_dtype(values.dtype):
                arr = values.to_numpy().astype(np.int64)
                is_rel = is_relative if is_relative is not None else True
                return (arr, is_rel, None, False)
            if pd.api.types.is_timedelta64_dtype(values.dtype):
                arr = values.to_numpy().view(np.int64).copy()
                return (arr, True, None, True)
            raise TypeError(
                f"pd.Index with dtype {values.dtype} is not supported. "
                f"Expected integer or timedelta dtype."
            )

        # lists of pandas/timedelta scalars
        if isinstance(values, list):
            return PandasFHConverter._list_to_internal(values)

        raise TypeError(
            f"Unsupported type for forecasting horizon values: "
            f"{type(values).__name__}. When passing pandas objects, "
            f"following are expected types: pd.PeriodIndex, "
            f"pd.DatetimeIndex, pd.TimedeltaIndex, pd.RangeIndex, "
            f"pd.Index (integer or timedelta dtype), pd.Timedelta, "
            f"pd.offsets.BaseOffset, or list of pandas scalars."
        )

    @staticmethod
    def _list_to_internal(values: list) -> tuple:
        """Convert list of pandas/temporal scalar types to internal tuple.

        Dispatches on the type of the first element. All elements must be
        of the same type (enforced via ``_check_list_homogeneity``).

        Supported element types and their conversion:

        - ``pd.Timedelta``, ``np.timedelta64``, ``datetime.timedelta``:
        Converted to ``pd.TimedeltaIndex``. If freq is inferrable,
        normalized to integer steps; otherwise stored as nanoseconds
        with ``values_are_nanos=True``.
        - ``pd.Period``: Converted to ``pd.PeriodIndex``, ordinals
        extracted via ``.asi8``. Always absolute.
        - ``pd.Timestamp``: Converted to ``pd.DatetimeIndex``, then to
        ``pd.PeriodIndex`` via ``.to_period(freq)``. Freq must be
        inferrable, otherwise raises. Always absolute.
        - ``pd.offsets.BaseOffset``: Each offset is converted to
        ``pd.Timedelta``, then follows the timedelta path.

        Note: ``list[int]`` is not handled here — it is handled by
        ``ForecastingHorizon._coerce_canonical`` before reaching the
        converter.

        Parameters
        ----------
        values : list
            Non-empty list of homogeneous pandas/temporal scalars.

        Returns
        -------
        tuple of (np.ndarray, bool, str or None, bool)
            (values, is_relative, freq, values_are_nanos).
            Same format as ``to_internal``.

        Raises
        ------
        ValueError
            If ``values`` is empty, or if list of Timestamps has no
            inferrable freq.
        TypeError
            If element type is not supported, or if list elements are
            not homogeneous.
        """
        from datetime import timedelta as _timedelta

        if len(values) == 0:
            raise ValueError("Forecasting horizon values must not be empty.")

        # timedelta types -> TimedeltaIndex path
        # pd.Timedelta, np.timedelta64, and stdlib datetime.timedelta are
        # combined into a single case because they all represent the same
        # concept and pd.TimedeltaIndex accepts all three.
        _timedelta_types = (pd.Timedelta, np.timedelta64, _timedelta)
        if isinstance(values[0], _timedelta_types):
            PandasFHConverter._check_list_homogeneity(values, _timedelta_types)
            idx = pd.TimedeltaIndex(values)
            freq_str = PandasFHConverter._freqstr(idx)
            if freq_str is not None:
                steps = PandasFHConverter.nanos_to_steps(idx.asi8.copy(), freq_str)
                return (steps, True, freq_str, False)
            else:
                return (idx.asi8.copy(), True, None, True)

        # period values -> ordinals via PeriodIndex
        if isinstance(values[0], pd.Period):
            PandasFHConverter._check_list_homogeneity(values, pd.Period)
            idx = pd.PeriodIndex(values)
            arr = idx.asi8.copy()
            freq_str = PandasFHConverter._freqstr(idx)
            return (arr, False, freq_str, False)

        # timestamp values -> ordinals via DatetimeIndex -> PeriodIndex
        if isinstance(values[0], pd.Timestamp):
            PandasFHConverter._check_list_homogeneity(values, pd.Timestamp)
            idx = pd.DatetimeIndex(values)
            freq_str = PandasFHConverter._freqstr(idx)
            if freq_str is None:
                raise ValueError(
                    "List of Timestamps without inferrable freq is not "
                    "supported. Provide freq explicitly via "
                    "ForecastingHorizon(values, freq=...)."
                )
            arr = idx.to_period(freq_str).asi8.copy()
            return (arr, False, freq_str, False)

        # offset objects -> timedelta path
        if isinstance(values[0], pd.offsets.BaseOffset):
            PandasFHConverter._check_list_homogeneity(values, pd.offsets.BaseOffset)
            tds = [pd.Timedelta(v) for v in values]
            idx = pd.TimedeltaIndex(tds)
            freq_str = PandasFHConverter._freqstr(idx)
            if freq_str is not None:
                steps = PandasFHConverter.nanos_to_steps(idx.asi8.copy(), freq_str)
                return (steps, True, freq_str, False)
            else:
                return (idx.asi8.copy(), True, None, True)

        raise TypeError(
            f"List with element type {type(values[0]).__name__} is not supported."
        )

    @staticmethod
    def to_pandas_index(
        values: np.ndarray,
        is_relative: bool,
        freq: str | None = None,
        values_are_nanos: bool = False,
    ) -> pd.Index:
        """Convert internal FH state to a pandas Index.

        Reconstructs a pandas Index from the four internal attributes
        of a ForecastingHorizon. The output type depends on the state:

        - ``values_are_nanos=True``: returns ``pd.TimedeltaIndex``.
        Values are raw nanoseconds from a freq-less TimedeltaIndex
        that hasn't been converted to integer steps yet.
        - ``is_relative=False`` and ``freq is not None``: returns
        ``pd.PeriodIndex``. Values are period ordinals, and freq is
        needed to reconstruct the periods.
        - All other cases: returns plain ``pd.Index`` with integer
        dtype. This covers relative integer FH (with or without
        freq), and absolute integer FH without freq.

        Note: relative FH that originated from a TimedeltaIndex (with
        freq) is normalized to integer steps at construction and will
        be returned as plain integer Index, not TimedeltaIndex. The
        original input type is not preserved. This is by design —
        output type reconstruction for prediction indices is handled
        by ``to_absolute_index`` using cutoff context.

        Parameters
        ----------
        values : np.ndarray
            Int64 numpy array of horizon values. Contains period
            ordinals (absolute with freq), raw nanoseconds
            (values_are_nanos), or integer step counts (all other).
        is_relative : bool
            Whether values are relative to a training cutoff. Only
            affects output type when ``freq`` is also set:
            - absolute with freq produces PeriodIndex,
            - relative with freq produces integer Index.
        freq : str or None
            Frequency string (e.g. ``"M"``, ``"D"``). Required to
            reconstruct PeriodIndex for absolute values. Ignored for
            relative values and nanos.
        values_are_nanos : bool
            If True, values are raw nanoseconds pending freq
            assignment. Takes precedence over all other parameters
            for determining output type. Should only be True when
            ``freq`` is None.

        Returns
        -------
        pd.TimedeltaIndex, pd.PeriodIndex, or pd.Index
            Pandas Index matching the semantic state of the FH.
        """
        if values_are_nanos:
            td_arr = values.copy().view("timedelta64[ns]")
            return pd.TimedeltaIndex(td_arr)

        if not is_relative and freq is not None:
            return pd.PeriodIndex.from_ordinals(values.copy(), freq=freq)

        return pd.Index(values.copy(), dtype=int)

    # this function handles conversion of period ordinals back to DatetimeIndex when
    # cutoff is datetime-based and involves timezone handling for DST-aware timezones.
    # Should be reviewed carefully to ensure correctness,
    # especially around DST boundaries and the potential loss of
    # original UTC offset information due to the period-ordinal round-trip.
    # tests for this should also be included in the test suite
    # to cover edge cases around DST transitions.
    @staticmethod
    def steps_to_datetime(values, freq, tz=None, sub_period_offset=None):
        """Convert integer step values (period ordinals) to DatetimeIndex.

        Used by ``to_absolute_index`` when the cutoff is a datetime
        type, to reconstruct DatetimeIndex output from period ordinals.

        Conversion path: ordinals -> PeriodIndex -> DatetimeIndex via
        ``to_timestamp()``, which returns the **start** of each period
        (e.g. ``Period("2020-01", "M")`` -> ``Timestamp("2020-01-01")``).

        If ``sub_period_offset`` is provided, it is added to the
        tz-naive DatetimeIndex before timezone handling. This preserves
        sub-period precision from the cutoff (e.g. a cutoff at 12:00
        with daily freq produces timestamps at 12:00, not midnight).
        Refer issue #5186.

        If ``tz`` is provided, the tz-naive DatetimeIndex is first
        localized to UTC (which has no DST transitions), then
        converted to the target timezone via ``tz_convert``. This
        avoids ``AmbiguousTimeError`` / ``NonExistentTimeError`` that
        would occur with direct ``tz_localize`` for DST-aware
        timezones at DST boundaries. The trade-off is that for
        timestamps at DST boundaries, the reconstructed UTC offset
        may differ from the original by up to 1 hour, since the
        period-ordinal round-trip loses the original offset.

        Parameters
        ----------
        values : np.ndarray
            Int64 period ordinals.
        freq : str
            Frequency string. Must not be None.
        tz : str or None
            Timezone to localize the output DatetimeIndex.
            None produces a tz-naive DatetimeIndex.
        sub_period_offset : pd.Timedelta or None
            Offset within a period to add to the reconstructed
            timestamps. Computed from the cutoff's position within
            its period (e.g. 12 hours for a cutoff at noon with
            daily freq). Applied before timezone handling.

        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex reconstructed from ordinals.

        Raises
        ------
        ValueError
            If ``freq`` is None (from ``PeriodIndex.from_ordinals``).
        """
        period_idx = pd.PeriodIndex.from_ordinals(values.copy(), freq=freq)
        dt_idx = period_idx.to_timestamp()
        if sub_period_offset is not None and sub_period_offset > pd.Timedelta(0):
            dt_idx = dt_idx + sub_period_offset
        if tz is not None:
            # localize to UTC first (no DST ambiguity), then convert
            # to target tz. Direct tz_localize(tz) would raise
            # AmbiguousTimeError/NonExistentTimeError for timestamps
            # at DST boundaries in DST-aware timezones.
            dt_idx = dt_idx.tz_localize("UTC").tz_convert(tz)
        return dt_idx

    # cutoff conversion

    # self note: below method added after vignette exploration
    @staticmethod
    def cutoff_to_steps(cutoff, freq=None):
        """Convert cutoff to an integer step value (period ordinal).

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, pd.Index, or np.integer
            Cutoff value. If pd.Index, the last element is used.
            For ``pd.Period``, if ``freq`` is provided and differs from
            the Period's own freq, the Period is converted via
            ``asfreq(freq)`` to produce the correct ordinal.
            For ``pd.Timestamp`` (including tz-aware), the timestamp is
            converted to a Period via ``to_period(freq)``. Timezone is
            handled correctly by pandas.
        freq : str or None
            Frequency string. Required for ``pd.Timestamp`` cutoff.
            Used for ``pd.Period`` cutoff to ensure the ordinal matches
            the FH's frequency coordinate system.

        Returns
        -------
        np.int64
            Cutoff as integer step (period ordinal for temporal types).

        Raises
        ------
        ValueError
            If cutoff is a Timestamp and freq is not provided.
        TypeError
            If cutoff type is not supported.
        """
        # unwrap pd.Index to scalar
        if isinstance(cutoff, pd.Index):
            if len(cutoff) == 0:
                raise ValueError("Cutoff index is empty.")
            scalar = cutoff[-1]
            # extract freq from index if not provided
            if freq is None and hasattr(cutoff, "freq") and cutoff.freq is not None:
                freq = PandasFHConverter._freqstr(cutoff)
            return PandasFHConverter.cutoff_to_steps(scalar, freq=freq)

        if isinstance(cutoff, pd.Period):
            cutoff_freq = PandasFHConverter.normalize_freq(cutoff.freqstr)
            if freq is not None and cutoff_freq != freq:
                cutoff = cutoff.asfreq(freq)
            return np.int64(cutoff.ordinal)

        if isinstance(cutoff, pd.Timestamp):
            if freq is None:
                raise ValueError(
                    "freq is required to convert Timestamp cutoff to "
                    "integer steps. Provide freq on the ForecastingHorizon "
                    "or use a PeriodIndex cutoff."
                )
            period = cutoff.to_period(freq)
            return np.int64(period.ordinal)

        if isinstance(cutoff, (int, np.integer)):
            return np.int64(cutoff)

        raise TypeError(
            f"Unsupported cutoff type: {type(cutoff).__name__}. "
            f"Expected pd.Period, pd.Timestamp, int, or pd.Index."
        )

    @staticmethod
    def cutoff_is_dti_ts(cutoff) -> bool:
        """Check if cutoff is a datetime-based type.

        Returns True for ``pd.DatetimeIndex`` and ``pd.Timestamp``.
        Used by ``to_absolute_index`` to decide whether to produce
        DatetimeIndex output.

        Parameters
        ----------
        cutoff : pd.DatetimeIndex, pd.Timestamp, or other
            Cutoff value to check.

        Returns
        -------
        bool
            True if cutoff is ``pd.DatetimeIndex`` or
            ``pd.Timestamp``, False otherwise.
        """
        if isinstance(cutoff, pd.DatetimeIndex):
            return True
        if isinstance(cutoff, pd.Timestamp):
            return True
        return False

    @staticmethod
    def cutoff_tz(cutoff) -> str | None:
        """Extract timezone from cutoff, if present.

        Checks ``pd.DatetimeIndex`` and ``pd.Timestamp`` for a ``.tz``
        attribute. All other types return None.

        Parameters
        ----------
        cutoff : pd.DatetimeIndex, pd.Timestamp, or other
            Cutoff value. Only ``pd.DatetimeIndex`` and
            ``pd.Timestamp`` with a non-None ``.tz`` will return a
            timezone string.  All other types return None.

        Returns
        -------
        str or None
            Timezone string (e.g. ``"UTC"``, ``"US/Eastern"``), or
            None if cutoff has no timezone or is not a datetime type.
        """
        if isinstance(cutoff, pd.DatetimeIndex) and cutoff.tz is not None:
            return str(cutoff.tz)
        if isinstance(cutoff, pd.Timestamp) and cutoff.tz is not None:
            return str(cutoff.tz)
        return None

    @staticmethod
    def cutoff_sub_period_offset(cutoff, freq):
        """Compute sub-period offset of a datetime cutoff within its period.

        For a cutoff at ``2025-03-02 12:00:00`` with ``freq="D"``, the
        period start is midnight, so the offset is ``Timedelta("12h")``.
        For a cutoff exactly on a period boundary the offset is zero.

        Used by ``to_absolute_index`` to preserve sub-period precision
        that would otherwise be lost in the period-ordinal round-trip.
        Refer issue #5186.

        Parameters
        ----------
        cutoff : pd.DatetimeIndex or pd.Timestamp
            Datetime cutoff. If ``pd.DatetimeIndex``, the last element
            is used.
        freq : str
            Frequency string for the period grid.

        Returns
        -------
        pd.Timedelta
            Offset within the period (>= 0). Zero when the cutoff
            falls exactly on a period boundary.
        """
        cutoff_ts = cutoff[-1] if isinstance(cutoff, pd.Index) else cutoff
        if cutoff_ts.tzinfo is not None:
            cutoff_naive = cutoff_ts.tz_localize(None)
        else:
            cutoff_naive = cutoff_ts
        period_start = cutoff_naive.to_period(freq).to_timestamp()
        return cutoff_naive - period_start

    @staticmethod
    def nanos_to_steps(nanos: np.ndarray, freq: str) -> np.ndarray:
        """Convert nanosecond values to integer steps using freq.

        Parameters
        ----------
        nanos : np.ndarray of int64
            Nanosecond values.
        freq : str
            Frequency string (must be a fixed-length frequency like
            "D", "h", "s", not variable-length like "M", "Y").

        Returns
        -------
        np.ndarray of int64
            Integer steps.

        Raises
        ------
        ValueError
            If freq is a variable-length frequency (M, Q, Y) that cannot
            be converted to a fixed nanosecond count, or if nanos are not
            evenly divisible by the freq.
        """
        from pandas.tseries.frequencies import to_offset

        offset = to_offset(_resolve_pandas_freq(freq))
        try:
            freq_nanos = offset.nanos
        except ValueError:
            raise ValueError(
                f"Cannot convert nanosecond timedeltas to steps with "
                f"non-fixed frequency {freq!r}. Variable-length frequencies "
                f"like 'M', 'Q', 'Y' do not have a fixed nanosecond count."
            )

        remainder = nanos % freq_nanos
        if np.any(remainder != 0):
            raise ValueError(
                f"Timedelta values are not evenly divisible by frequency "
                f"{freq!r}. This means the timedeltas do not represent "
                f"integer multiples of the frequency."
            )
        return (nanos // freq_nanos).astype(np.int64)

    # frequency helper functions

    @staticmethod
    def freq_multiplier(freq: str | None) -> int:
        """Return the step multiplier for a frequency string.

        For multi-step frequencies like ``"2D"`` or ``"3M"``, period ordinals
        have diffs equal to the multiplier per period. This method extracts
        that multiplier from the frequency string via ``to_offset(freq).n``.

        Used at conversion boundaries (``to_relative``, ``to_absolute``,
        ``to_absolute_int``, arithmetic, ``_is_contiguous``) to convert
        between ordinal diffs and step counts.

        Parameters
        ----------
        freq : str or None
            Frequency string. If None, returns 1 (no scaling).

        Returns
        -------
        int
            The multiplier ``n`` from the offset. 1 for simple frequencies
            (``"D"``, ``"M"``), >1 for multi-step (``"2D"``, ``"3M"``).
        """
        if freq is None:
            return 1
        from pandas.tseries.frequencies import to_offset

        return to_offset(_resolve_pandas_freq(freq)).n

    # final check pending for this function
    @staticmethod
    def extract_freq(obj) -> str | None:
        """Extract and normalize a frequency string from a pandas object or a string.

        Handles pd.Index, pd.Period, pd.offsets.BaseOffset, strings,
        and objects with a ``cutoff`` attribute (e.g. sktime forecasters).

        Parameters
        ----------
        obj : str, pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex,
            pd.Index, pd.Period, pd.Timestamp, pd.offsets.BaseOffset,
            or forecaster object carrying frequency information.
            Types that always carry freq (``pd.Period``, ``pd.offsets.BaseOffset``)
            always return a string.
            Types that may carry freq (``pd.PeriodIndex``, ``pd.DatetimeIndex``,
            ``pd.TimedeltaIndex``) return a string only if ``.freq`` is set.
            Types that never carry freq (``pd.Timestamp``, integer ``pd.Index``,
            ``pd.RangeIndex``) always return None.

        Returns
        -------
         str or None
             Normalized frequency string, or None if no frequency
             could be extracted.
        """
        if isinstance(obj, str):
            try:
                from pandas.tseries.frequencies import to_offset

                resolved = _resolve_pandas_freq(obj)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    offset = to_offset(resolved)
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

        if isinstance(obj, (pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex)):
            return PandasFHConverter._freqstr(obj)

        # generic pd.Index, no freq attribute: return None

        return None

    @staticmethod
    def normalize_freq(freq_str: str | None) -> str | None:
        """Normalize frequency string to our canonical form.

        Maps any known pandas alias to our canonical name using
        the dynamic resolution system. Handles multiplied
        (``"2ME"`` → ``"2M"``) and anchored (``"QE-DEC"`` →
        ``"Q-DEC"``) forms.

        Parameters
        ----------
        freq_str : str or None
            Raw frequency string.

        Returns
        -------
        str or None
            Normalized frequency string.
        """
        return _normalize_freq(freq_str)
        """
        Old code with hardcoded alias map,
        kept for reference pending final check of the new dynamic system:
        # <check>
        # 1. check for unsupported frequencies and raise informative errors
        # 2. check for completeness of the alias map and add any missing aliases
        # 3. is there way to leverage pandas frequency parsing/normalization logic
        #    instead of hardcoding an alias map here?
        # 4. if we keep the alias map, make it more comprehensive and robust,
        #    and add tests for it.
        #    For example, handling both "M" and "ME" for month-end frequencies,
        #    and ensuring that all common aliases are covered.
        # 5. whether to use pandas.tseries.frequencies.to_offset to validate and
        #    normalize freq strings, which would leverage pandas'
        #    internal logic and ensure consistency with pandas behavior.
        #    one way of achieving 1. and 2. without hardcoding an alias map
        #    if it succeeds,
        #    use the resulting offset's name as the normalized freq string.
        # 6. Make the alias map a frozenset to prevent accidental modifications
        # </check>
        alias_map = {
            "ME": "M",
            "QE": "Q",
            "YE": "Y",
            "BQE": "BQ",
            "BYE": "BY",
            "SME": "SM",
        }
        return alias_map.get(freq_str, freq_str)"""

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
    # copied from old _fh.py
    # in order to serve import needs of `sktime/forecasting/compose/reduce.py`
    if cutoff is None:
        raise ValueError("`cutoff` must be given, but found none.")
    if isinstance(index, pd.PeriodIndex):
        assert isinstance(cutoff, (pd.Period, pd.PeriodIndex))
        assert index.freqstr == cutoff.freqstr

    if isinstance(index, pd.DatetimeIndex):
        assert isinstance(cutoff, (pd.Timestamp, pd.DatetimeIndex))


def _index_range(relative, cutoff):
    """Return Index Range relative to cutoff."""
    # copied from old _fh.py
    # in order to serve import needs of `sktime/forecasting/compose/reduce.py`
    _check_cutoff(cutoff, relative)
    is_timestamp = isinstance(cutoff, pd.DatetimeIndex)

    if is_timestamp:
        # coerce to pd.Period for reliable arithmetic operations and
        # computations of time deltas
        cutoff = cutoff.to_period(cutoff.freqstr)

    if isinstance(cutoff, pd.Index):
        cutoff = cutoff[[0] * len(relative)]

    absolute = cutoff + relative

    if is_timestamp:
        # coerce back to DatetimeIndex after operation
        absolute = absolute.to_timestamp(cutoff.freqstr)
    return absolute

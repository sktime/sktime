# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
ForecastingHorizonV2: pandas-agnostic forecasting horizon implementation.

All pandas-specific logic (type conversions, frequency handling, version detection)
is delegated to the _fh_utils module.
"""

__all__ = ["ForecastingHorizonV2"]

import numpy as np

from sktime.forecasting.base._fh_utils import PandasFHConverter
from sktime.forecasting.base._fh_values import FHValues, FHValueType

# <check></check>
# this is the marker left to mark all delayed checks
# all occurences must be removed/addressed before merging this code


class ForecastingHorizonV2:
    """Forecasting horizon with pandas-decoupled internals.

    Parameters
    ----------
    values : pd.Index, pd.TimedeltaIndex, np.array, list, pd.Timedelta, or int
        Values of forecasting horizon.
    is_relative : bool, optional (default=None)
        If True, a relative ForecastingHorizon is created:
        values are relative to end of training series.
        If False, an absolute ForecastingHorizon is created:
        values are absolute.
        If None, the flag is determined automatically:
        relative - if values are of supported relative type
        absolute - if values are of supported absolute type
    freq : str, pd.Index, pandas offset, or sktime forecaster, optional (default=None)
        Object carrying frequency information on values
        Ignored unless values lack inferable freq.

    Examples
    --------
    >>> from sktime.forecasting.base._fh_v2 import ForecastingHorizonV2
    >>> fh = ForecastingHorizonV2([1, 2, 3])
    >>> fh.is_relative
    True
    >>> fh.to_numpy()
    numpy.ndarray([1, 2, 3])
    """

    def __init__(
        self,
        values=None,
        is_relative: bool | None = None,
        freq=None,
    ):
        # convert input to internal representation
        self.fhvalues = PandasFHConverter.to_internal(values, freq)
        # above conversion would need to be bypassed when input is already in internal
        # FHValues representation,
        # for example when creating modified copies internally with the _new constructor

        if self.fhvalues.freq is None and freq is not None:
            # set freq from input if not already set
            # this stores normalized freq string
            # not the pandas freq object
            self.fhvalues.freq = PandasFHConverter.extract_freq(freq)

        # if is_relative is provided, validate compatibility
        # of passed is_relative with value type
        if is_relative is not None:
            if not isinstance(is_relative, bool):
                raise TypeError("`is_relative` must be a boolean or None")
            self.is_relative = is_relative
            if is_relative and not self.fhvalues.is_relative_type():
                # if is_relative is passed as True,
                # then values must be of a type that can be relative
                raise TypeError(
                    f"`values` type {self.fhvalues.value_type.name} is "
                    f"not compatible with `is_relative=True`."
                )
            if not is_relative and not self.fhvalues.is_absolute_type():
                # opposite for absolute
                raise TypeError(
                    f"`values` type {self.fhvalues.value_type.name} is "
                    f"not compatible with `is_relative=False`."
                )
        # determine is_relative if not provided
        else:
            # Infer from value type
            vtype = self.fhvalues.value_type
            if vtype in (FHValueType.TIMEDELTA,):
                self.is_relative = True
            elif vtype in (FHValueType.PERIOD, FHValueType.DATETIME):
                self.is_relative = False
            elif vtype == FHValueType.INT:
                # INT can be either relative or absolute
                # in line 306 code block in _fh.py, the default for this case
                # is set to relative, hence using the same here
                # if this handling is ok, then this elif can be merged into the
                # 1st if block above
                self.is_relative = True
            else:
                raise TypeError(f"Cannot infer is_relative for value type {vtype.name}")
        # <check>
        # above code assumes fhvalues.is_relative_type and
        # fhvalues.is_absolute_type to be implemented.
        # Currently they are not implemented.</check>

    # this class would also need a copy constructor
    # because the conversion from pandas types to internal representation
    # is done as first step in __init__,
    # and the resulting FHValues instance is stored as an attribute.
    # internal methods during various operations would need to create new FHValues
    # instances with modified values. So a copy constructor that can take
    # an existing FHValues instance and create a new one with modified values
    # but same metadata would be needed
    # to avoid having to go through the full conversion process again

    # methods for operating on the internal FHValues instance,

    # methods for converting back to pandas types when needed for interoperability

    # added dummy comment to test skipping of CI runs on every subsequent commit
    # until all checks are removed and code is ready for review/tests

    @property
    def is_relative(self) -> bool:
        """Whether forecasting horizon is relative to the end of the training series.

        Returns
        -------
        is_relative : bool
        """
        return self.is_relative

    @property
    def freq(self) -> str | None:
        """Frequency string, or None."""
        return self.fhvalues.freq

    @freq.setter
    def freq(self, obj) -> None:
        """Set frequency from string, pd.Index, pd.offset, or forecaster.

        Parameters
        ----------
        obj : str, pd.Index, pd.offsets.BaseOffset, or forecaster
            Object carrying frequency information.

        Raises
        ------
        ValueError
            If freq is already set and conflicts with new value.
        """
        new_freq = PandasFHConverter.extract_freq(obj)
        old_freq = self.fhvalues.freq

        if old_freq is not None and new_freq is not None and old_freq != new_freq:
            raise ValueError(
                f"Frequencies do not match: current={old_freq}, new={new_freq}"
            )
        if new_freq is not None:
            self.fhvalues = self.fhvalues._new(freq=new_freq)

    # core conversion methods

    # <check>
    # to_relative requires cutoff to be specified
    # but for a drop-in replacement for the old FH,
    # we want to allow users to call to_relative without cutoff
    # and use the same default cutoff as the old FH, which is end of training series
    # so we would need to add logic to determine the default cutoff
    # when cutoff is not provided
    # </check>
    def to_relative(self, cutoff=None):
        """Return relative version of forecasting horizon.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, or pd.Index, optional
            Cutoff value required for conversion.

        Returns
        -------
        ForecastingHorizonV2
            Relative representation of forecasting horizon.
        """
        if self.is_relative:
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
        vtype = self.fhvalues.value_type
        vals = self.fhvalues.values

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
            # another place where _new is needed to create a new FHValues instance
            # with modified values but same metadata

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
        ForecastingHorizonV2
            Absolute representation of forecasting horizon.
        """
        if not self.is_relative:
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
        vtype = self.fhvalues.value_type
        vals = self.fhvalues.values

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

    # Dunders -> Arithmatic operators

    def __add__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            result = self.fhvalues.values + other.fhvalues.values
        else:
            result = self.fhvalues.values + np.int64(other)
        fhv = self.fhvalues._new(values=result)
        return self._new(fhvalues=fhv)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            result = self.fhvalues.values - other.fhvalues.values
        else:
            result = self.fhvalues.values - np.int64(other)
        fhv = self.fhvalues._new(values=result)
        return self._new(fhvalues=fhv)

    def __rsub__(self, other):
        # not checking if other is FH here
        # because __rsub__ is mostly called
        # when other does not support the operation with FH,
        # in which case we want to treat other as a scalar.
        # If other is FH, then other minus self
        # would have been handled by other.__sub__
        # and this method would not be called
        result = np.int64(other) - self.fhvalues.values
        fhv = self.fhvalues._new(values=result)
        return self._new(fhvalues=fhv)

    def __mul__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            result = self.fhvalues.values * other.fhvalues.values
        else:
            result = self.fhvalues.values * np.int64(other)
        fhv = self.fhvalues._new(values=result)
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
        if isinstance(other, ForecastingHorizonV2):
            return self.fhvalues.values == other.fhvalues.values
        return self.fhvalues.values == np.int64(other)

    def __ne__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            return self.fhvalues.values != other.fhvalues.values
        return self.fhvalues.values != np.int64(other)

    def __lt__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            return self.fhvalues.values < other.fhvalues.values
        return self.fhvalues.values < np.int64(other)

    def __le__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            return self.fhvalues.values <= other.fhvalues.values
        return self.fhvalues.values <= np.int64(other)

    def __gt__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            return self.fhvalues.values > other.fhvalues.values
        return self.fhvalues.values > np.int64(other)

    def __ge__(self, other):
        if isinstance(other, ForecastingHorizonV2):
            return self.fhvalues.values >= other.fhvalues.values
        return self.fhvalues.values >= np.int64(other)

    # Dunders -> container methods len, getitem, max, min
    def __len__(self):
        return len(self.fhvalues)

    def __getitem__(self, key):
        result = self.fhvalues[key]
        if isinstance(result, FHValues):
            return self._new(fhvalues=result)
        # scalar — return as-is
        return result

    def max(self):
        """Return the maximum value."""
        return self.fhvalues.max()

    def min(self):
        """Return the minimum value."""
        return self.fhvalues.min()

    # Below method computes a hash for the ForecastingHorizonV2 instance,
    # which allows it to be used in sets and as dictionary keys.
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
    # ForecastingHorizonV2, or
    # Move element-wise comparison to a separate method
    # and keep __eq__ consistent with __hash__.
    # Need to consider this in th context of forecasting horizon usage.
    # </check>
    def __hash__(self):
        return hash((self.fhvalues, self.is_relative))

    def __repr__(self):
        class_name = type(self).__name__
        vals = self.fhvalues
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

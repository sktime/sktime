# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
ForecastingHorizonV2: pandas-agnostic forecasting horizon implementation.

All pandas-specific logic (type conversions, frequency handling, version detection)
is delegated to the _fh_utils module.
"""

from sktime.forecasting.base._fh_values import FHValueType
from sktime.forecasting.base._pandas_fh_converter import PandasFHConverter

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

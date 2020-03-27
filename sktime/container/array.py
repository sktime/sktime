import numpy as np
import pandas as pd

import numbers
from collections.abc import Iterable
from typing import Any, Callable, Optional, Type

from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype
)

from sktime.container import TimeBase
from sktime.container.base import check_data_index, ensure_2d


# -----------------------------------------------------------------------------
# Extension Type
# -----------------------------------------------------------------------------
class TimeDtype(ExtensionDtype):
    type = TimeBase
    name = "timeseries"
    kind = 'O'
    na_value = None

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a 'TimeDtype' from '{}'".format(string))

    @classmethod
    def construct_array_type(cls):
        return TimeArray

# Register the datatype with pandas
register_extension_dtype(TimeDtype)


# -----------------------------------------------------------------------------
# Constructors / converters to other formats
# -----------------------------------------------------------------------------

def from_list(data):
    n = len(data)

    out_d = []
    out_i = []
    widths = np.full((n, ), np.nan, dtype=np.double)

    for idx in range(n):
        ts = data[idx]

        # TODO: think about other constructors that might make sense
        if isinstance(ts, (TimeBase, TimeArray)):
            check_data_index(ts.data, ts.time_index)
            out_d.append(ts.data)
            out_i.append(ts.time_index)
            widths[idx] = ts.data.shape[1]
        elif isinstance(ts, (tuple, list, np.ndarray)):
            out_d.append(ensure_2d(np.array(ts[0])))
            out_i.append(ensure_2d(np.array(ts[1])))
            widths[idx] = out_d[idx].shape[1]
        elif isinstance(ts, pd.Series):
            out_d.append(ts.values)
            out_i.append(ts.index.values)
            widths[idx] = ts.values.size
        elif ts is None:
            out_d.append(None)
            out_i.append(None)
        else:
            raise TypeError("Input must be valid timeseries objects: {0}".format(ts))

    # Deal with missing
    mask = np.isnan(widths)
    if np.all(mask):
        if n == 0:
            return None
        return np.array([None for _ in range(n)])
    elif np.any(mask):
        # TODO: is there a better way to do this?
        w = out_d[int(np.where(~mask)[0][0])].shape[1]
        out_d = [np.full((1, w), np.nan, np.double) if m else d for m, d in zip(mask, out_d)]
        out_i = [np.full((1, w), np.nan, np.double) if m else i for m, i in zip(mask, out_i)]

    # Ensure that all the widths are equal
    if not np.all(widths[~mask] == widths[~mask][0]):
        raise ValueError("The width of each timeseries must be the equal.")

    return TimeArray(np.vstack(out_d), np.vstack(out_i))


def from_pandas(data):
    if isinstance(data, pd.Series):
        return from_list(data)
    elif isinstance(data, pd.DataFrame):
        time_index = data.columns.values[np.newaxis, :]
        time_index = np.repeat(time_index, data.shape[0], axis=0)
        return TimeArray(data.to_numpy(), time_index)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame: {0}".format(data))



# ----------------------------------------------------------------------------------------------------------------------
# Extension Container
# ----------------------------------------------------------------------------------------------------------------------

# TODO: update docstrings to reflect recent refactoring

class TimeArray(ExtensionArray):
    """
    Class wrapping a numpy array of time series and
    holding the array-based implementations.
    """

    # A note on the internal data layout: TimeArray implements a
    # collection of equal length time series in which rows correspond to
    # a single observed time series. We use a Numpy array to store the
    # individual data points of each time series, i.e. columns relate to
    # specific points in time. The time indices are stored in a separate
    # Numpy array, either of dimension 1xT for a common index across
    # time series or NxT for a separate index for each time series.
    _dtype = TimeDtype()
    _can_hold_na = True

    @property
    def _constructor(self) -> Type["TimeArray"]:
        return TimeArray

    def _choose_time_index(self, from_param, from_data):
        if from_param is None:
            return from_data
        return from_param

    def _make_seq(self, data):
        return np.vstack([np.arange(data.shape[1]) for _ in range(data.shape[0])])


    def __init__(self, data, time_index=None):
        # TODO: add copying functionality
        """
        Initialise a new TimeArray containing equal length time series

        Parameters
        ----------
        data :
        time_index :
        """

        index = None

        if isinstance(data, self.__class__):
            index = data.time_index
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise TypeError("'data' should be array of equal length timeseries. Use from_list, "
                            "to construct a TimeArray.")
        elif not data.ndim == 2:
            raise ValueError("'data' should be a 2-dimensional array of timeseries, where rows correspond"
                             "to series and columns to individual observations.")

        self.data = data

        if time_index is not None:
            self.time_index = time_index
        elif index is not None:
            self.time_index = index
        else:
            self.time_index = self._make_seq(self.data)

        check_data_index(self.data, self.time_index)

    # -------------------------------------------------------------------------
    # Class methods
    # -------------------------------------------------------------------------

    @classmethod
    def _from_ndarray(cls, data, copy=False):
        if copy:
            data = data.copy()
        new = TimeArray([])
        new.data = data
        return new

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if isinstance(scalars, TimeArray):
            return scalars
        return from_list(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError("Reconstruction of TimeArray after factorization has not been implemented "
                                  "yet.")

    @classmethod
    def _concat_same_type(cls, to_concat):
        ref_width = to_concat[0].data.shape[1]
        ti = []

        for ta in to_concat:
            if not isinstance(ta, TimeArray):
                raise TypeError(f"Only TimeArrays can be concatenated, got {type(ta.dtype)}")

            ta_width = ta.data.shape[1]
            if not ta_width == ref_width:
                raise ValueError(f"Lengths of concatenated TimeArrays must be compatible, got {ref_width} "
                                 f"and {ta_width}.")

            if ta.time_index.ndim == 1:
                ti.append(np.broadcast_to(ta.time_index, ta.data.shape))
            else:
                ti.append(ta.time_index)

        data = np.vstack([ta.data for ta in to_concat])

        return TimeArray(data, np.vstack(ti))

    # -------------------------------------------------------------------------
    # Interfaces
    # -------------------------------------------------------------------------
    def _get_time_index_at(self, row):
        if self.time_index.ndim == 1:
            return self.time_index
        else:
            return self.time_index[row]

    def __getitem__(self, idx):
        # validate and convert IntegerArray / BooleanArray
        # keys to numpy array, pass-through non-array-like indexers
        idx = pd.api.indexers.check_array_indexer(self, idx)

        if isinstance(idx, (int, np.int, numbers.Integral)):
            if np.all(np.isnan(self.data[idx])) and np.all(np.isnan(self.time_index[idx])):
                # Return the missing type if both data and time_index are completely missing
                return None

            return TimeBase(self.data[idx], self.time_index[idx])
        elif isinstance(idx, (Iterable, slice)):
            return self._constructor(self.data[idx], self.time_index[idx])
        else:
            raise TypeError("Index type not supported", idx)


    def __setitem__(self, key, value):
        # validate and convert IntegerArray / BooleanArray
        # keys to numpy array, pass-through non-array-like indexers
        key = pd.api.indexers.check_array_indexer(self, key)

        if not isinstance(value, (TimeBase, TimeArray)):
            value = from_list(value)

        # TODO: add dimensionality check

        if value is None or \
                (isinstance(value, np.ndarray) and np.all([x is None for x in value])):
            # This is setting all `key` elements to missing
            self.data[key] = np.empty((1,), dtype=self.data.dtype)
            self.time_index[key] = np.empty((1,), dtype=self.data.dtype)
        else:
            self.data[key] = value.data
            self.time_index[key] = value.time_index

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return (self.data.shape[0], )

    @property
    def ndim(self):
        return 1

    @property
    def nbytes(self):
        return self.data.nbytes


    # ------------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------------

    def isna(self):
        # TODO: revisit missingness
        return self.isna_row(self.data)

    def isna_row(self, arr):
        return np.apply_over_axes(np.all, self.isna_grid(arr), 1).flatten()

    def isna_grid(self, arr):
        # TODO: revisit missingness
        return np.isnan(arr)

    def astype(self, dtype, copy=True):
        if isinstance(dtype, TimeDtype):
            if copy:
                return self.copy()
            else:
                return self
        elif dtype is str or (hasattr(dtype, 'kind') and dtype.kind in {'U', 'S'}):
            return np.array(self).astype(dtype)
        elif dtype is object or (hasattr(dtype, 'kind') and dtype.kind in {'O'}):
            return np.array(self)
        raise ValueError(f"TimeArray can only be cast to numpy array with type object",
                         f"got type {dtype}")

    def unique(self):
        from pandas import factorize

        rows, _ = factorize(np.array(self).astype(str))

        return self[np.unique(rows)]

    # -------------------------------------------------------------------------
    # general array like compat
    # -------------------------------------------------------------------------

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None):
        return np.array([TimeBase(self.data[i], self.time_index[i]) for i in range(self.data.shape[0])])

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any):
        raise NotImplementedError("__array_ufunc__ not implemented yet")

    def __eq__(self, other):
        if isinstance(other, pd.Series):
            comp_data = np.broadcast_to(other.values[np.newaxis, :], self.data.shape)
            comp_index = np.broadcast_to(other.index.values[np.newaxis, :], self.time_index.shape)
        else:
            comp_data = other.data
            comp_index = other.time_index
        # TODO: sense check for other types

        return (np.all(self.data == comp_data)) & (np.all(self.time_index == comp_index))

    def __add__(self, o):
        if np.all(self.time_index != o.time_index):
            raise ValueError("The time indices of two TimeArrays that should be added must be identical.")

        return TimeArray(self.data + o.data, time_index=self.time_index)

    def copy(self, *args, **kwargs):
        return TimeArray(self.data.copy(), self.time_index.copy())

    def take(self, indices, allow_fill=False, fill_value=None):
        # Use the take implementation from pandas to get the takes separately
        # for the data and the time indices
        from pandas.api.extensions import take
        data = take(self.data, indices, allow_fill=allow_fill)
        time_index = take(self.time_index, indices, allow_fill=allow_fill)

        if allow_fill and isinstance(fill_value, TimeBase):
            # Fill those indices that were out of range with the fill TimeBase
            indices = np.asarray(indices, dtype=np.intp)
            out_of_range = (indices < 0) | (indices >= data.shape[0])

            data[self.isna_row(data) & out_of_range] = fill_value.data
            time_index[self.isna_row(time_index) & out_of_range] = fill_value.time_index
        elif allow_fill and fill_value is not None:
            TypeError(f"Only TimeBase are allowed as fill values, got {type(fill_value)}")

        if np.all(np.isnan(data)) and np.all(np.isnan(time_index)):
            # If the take resulted in a completely missing TimeArray
            # TODO: put in a separate function if needed elsewhere
            return np.array([None for _ in range(data.shape[0])])

        return self._constructor(data, time_index)

    # ------------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        from pandas.io.formats.printing import format_object_summary

        template = "{class_name}{data}\nLength: {length}, dtype: {dtype}"
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False, is_justify=False
        ).rstrip(", \n")
        class_name = "<{}>\n".format(self.__class__.__name__)

        return template.format(
            class_name=class_name, data=data, length=len(self), dtype=self.dtype
        )

    def _formatter(self, boxed: bool = False) -> Callable[[Any], Optional[str]]:
        if boxed:
            return str
        return repr


    # -------------------------------------------------------------------------
    # time series functionality
    # -------------------------------------------------------------------------

    def tabularise(self, name=None, return_array=False):
        if name is None:
            name = "dim"
        if return_array:
            return self.data
        # TODO: throw a naming error when time indes isn't equal in each row
        return pd.DataFrame(self.data, columns=[name + "_" + str(i) for i in self.time_index[0]])

    def tabularize(self, return_array=False):
        return self.tabularise(return_array)

    def check_equal_index(self):
        if self.time_index.ndim == 1 or self.time_index.shape[0] == 0:
            # TODO: change if one time index isn't allowed anymore
            self._equal_index = True
        else:
            self._equal_index = (self.time_index == self.time_index[0]).all()

        return self._equal_index

    def slice_time(self, time_index):
        if len(self.time_index.shape) == 1:
            sel = np.isin(self.time_index.to_numpy(), time_index)
            return TimeArray(self.data[:, sel], time_index=self.time_index[sel])
        elif self._equal_index:
            sel = np.isin(self.time_index.to_numpy()[0, :], time_index)
            return TimeArray(self.data[:, sel], time_index=self.time_index[:, sel])
        else:
            raise NotImplementedError("Time slicing that results in unequal lengths has not been implemented yet.")
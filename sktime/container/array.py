import numpy as np
import pandas as pd

import numbers
from collections.abc import Iterable

# TODO: add typing
from typing import (
    Any,
    Callable,
    Sequence,
    Tuple,
    Type
)

from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype
)

from sktime.container import TimeBase
from sktime.container.base import check_data_index, ensure_2d


# ------------------------------------------------------------------------------
# Extension Type
# ------------------------------------------------------------------------------


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
            raise TypeError(f"Cannot construct a 'TimeDtype' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return TimeArray


# Register the datatype with pandas
register_extension_dtype(TimeDtype)


# ------------------------------------------------------------------------------
# Constructors / converters to other formats
# ------------------------------------------------------------------------------

def from_list(data):
    """
    Create TimeArray object from list of timeseries-like objects

    Parameters
    ----------
    data : list
        each element in `data` represents a single timeseries that will be
        converted into a row in the TimeArray. Elements of `data` need to
        provide a time index and a corresponding value. This can be provided in
        the form of objects with internal value/index representations (TimeBase,
        TimeArray, pd.Series), or through a length 2 tuple/list/np.ndarray
        object of np.ndarrays in which the first entry will be interpreted as
        values and the second entry as indices.

    Returns
    -------
    TimeArray
    """
    n = len(data)

    out_d = []
    out_i = []
    widths = np.full((n, ), np.nan, dtype=np.float)

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
            raise TypeError("Input must be valid timeseries objects: {ts}")

    # Deal with missing
    mask = np.isnan(widths)
    if np.all(mask):
        # All list elements represented missing timeseries
        return TimeArray(empty((n, 0)))
    elif np.any(mask):
        # Some list elements represented missing timeseries
        # TODO: is there a better way to do this?
        w = out_d[int(np.where(~mask)[0][0])].shape[1]
        out_d = [np.full((1, w), np.nan, np.float) if m else d
                 for m, d in zip(mask, out_d)]
        out_i = [np.full((1, w), np.nan, np.float) if m else i
                 for m, i in zip(mask, out_i)]

    # Ensure that all the widths are equal
    if not np.all(widths[~mask] == widths[~mask][0]):
        raise ValueError("The width of each timeseries must be the equal.")

    return TimeArray(np.vstack(out_d), np.vstack(out_i))


def from_pandas(data):
    """
    Create TimeArray object from a nested pd.Series or from a pd.DataFrame

    If data is a nested pd.Series, each sub-series will be interpreted as a row
    in the TimeArray. If data is a pd.Dataframe, each row in the pd.Dataframe
    will be interpreted as a row in the TimeArray and the column index will be
    used as the index.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame

    Returns
    -------
    TimeArray
    """
    if isinstance(data, pd.Series):
        return from_list(data)
    elif isinstance(data, pd.DataFrame):
        time_index = data.columns.values[np.newaxis, :]
        time_index = np.repeat(time_index, data.shape[0], axis=0)
        return TimeArray(data.to_numpy(), time_index)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame: {data}")


def from_ts(ts):
    # Note: this is a very(!) preliminary implementation to convert to ts
    #       format, used solely to allow for factorization and pass pandas
    #       implementation tests. This will be extended in the future, after
    #       TimeArray compatibility with pandas has been resolved.
    # TODO: transfer a full implementation from
    #       sktime.utils.load_data.load_from_tsfile_to_dataframe
    def parse_line(line):
        line = line.strip("()")
        points = line.split("),(")
        index, data = zip(*[x.split(",") for x in points])
        return np.array(data, dtype=np.float), np.array(index, dtype=np.float)

    return from_list([TimeBase(*parse_line(l)) for l in ts])


def to_ts(obj, include_header=True):
    # Note: this is a very preliminary implementation to convert to ts format.
    #       This will be extended in the future, after TimeArray compatibility
    #       with pandas has been resolved.
    # TODO: extend to cover all cases
    if include_header:
        raise NotImplementedError("Conversion to ts format with headers not "
                                  "supported yet.")

    if not isinstance(obj, (TimeBase, TimeArray)):
        raise TypeError(f"Can only convert TimeBase or TimeArray objects, "
                        f"got {type(obj)}")
    elif isinstance(obj, TimeArray):
        obj = obj.astype(object)
    else:
        obj = [obj]

    # TODO: this returns a list which is needed for factorise but we might want
    #       to move thisinto a separate function and return a string here
    return ["(" + "),(".join([str(i) + "," + str(d)
                              for i, d in zip(x.data[0], x.time_index[0])])+")"
            for x in obj]


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def empty(shape, dtype=np.float):
    """
    Create an empty (i.e. np.nan or dtype equivalent) np.ndarray of given shape

    Parameters
    ----------
    shape : Tuple
    dtype : type

    Returns
    -------
    np.ndarray
    """
    return np.full(shape, np.nan, dtype=dtype)


def rows_na(arr, sum_func=np.all, axis=0):
    """
    Summarise the presence of missing values along an axis in np.ndarray

    Parameters
    ----------
    arr : np.ndarray
    sum_func : function, optional
        Summary function like np.all, np.any or np.sum
    axis : int
        axis along which to summarise

    Returns
    -------
    a boolean array
    """
    return np.apply_over_axes(sum_func, np.isnan(arr), axis).flatten()


# ------------------------------------------------------------------------------
# Extension Container
# ------------------------------------------------------------------------------


class TimeArray(ExtensionArray):
    """
    Holder for multiple timeseries objects with an equal number of
    observations.

    TimeArray is a container for storing and manipulating timeseries data. Each
    observation in TimeArray is a distinct timeseries, i.e. a series of key/
    value pairs that represent a time point and a corresponding measurement.

    A note on the internal data layout: TimeArray implements a
    collection of equal length timeseries in which rows correspond to
    a single observed timeseries. We use a Numpy array to store the
    individual data points of each time series, i.e. columns relate to
    specific points in time. The time indices are stored in a separate
    Numpy array.

    Note on compatibility with pandas:
    TimeArray aims to to satisfy pandas' extension array interface so it can be
    stored inside :class:`pandas.Series` and :class:`pandas.DataFrame`. While
    this works for basic indexing already, many of the more intricate
    functionality of pandas (e.g. apply or groupby) has not been integrated yet.

    Parameters
    ----------
    data : TimeArray or np.ndarray
        The measurements at certain time points (columns) for one or more
        timeseries (rows).
    time_index : np.ndarray, optional
        A time index for each entry in 'data'. Must be of the same shape as
        data. If None, the time index stored in 'data' will be used if 'data'
        is a TimeArray object, or a default time index of [0, 1, ..., N] will be
        generated for each row if 'data' is a np.ndarray.
    copy : bool, default False
        If True, copy the underlying data.
    """
    _dtype = TimeDtype()
    _can_hold_na = True

    @property
    def _constructor(self) -> Type["TimeArray"]:
        return TimeArray

    def __init__(self, data, time_index=None, copy=False):
        tidx = None

        # Initialise the data
        if data is None:
            # TODO: what if data is None by index is passed?
            data = np.full((1,0), np.nan, dtype=np.float)
        elif isinstance(data, self.__class__):
            tidx = data.time_index
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                f"'data' should be TimeArray or numpy.ndarray, got "
                f"{type(data)}. Use from_list, to construct a TimeArray from "
                f"list-like objects."
            )
        elif not data.ndim == 2:
            raise ValueError(
                "'data' should be a 2-dimensional, where rows correspond"
                "to timeseries and columns to individual observations."
            )

        # Initialise the time index
        if time_index is not None:
            if not isinstance(time_index, np.ndarray):
                raise TypeError(f"'time_index' should be numpy.ndarray, "
                                f"got {type(time_index)}.")
            tidx = time_index
        elif tidx is None:
            if data.shape[0] == 1:
                tidx = np.arange(data.shape[1])[np.newaxis, :]
            elif data.shape[0] > 1:
                tidx = np.vstack([np.arange(data.shape[1], dtype=np.float)
                                  for _ in range(data.shape[0])])
            else:
                tidx = data.copy()

        if copy:
            data = data.copy()
            tidx = tidx.copy()

        self.data = data
        self.time_index = tidx
        check_data_index(self.data, self.time_index)

    # -------------------------------------------------------------------------
    # Class methods
    # -------------------------------------------------------------------------

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False) -> Type[None]:
        """
        Construct a new TimeArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element should be a timeseries-like object,
            see from_list() for a detailed description.
        dtype : type, optional
            Construct for this particular dtype. This is currently ignored
            and only included for compatibility
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        TimeArray
        """
        if isinstance(scalars, TimeArray):
            return scalars
        return from_list(scalars)

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        """Return an array and missing value suitable for factorization.

        Note: strings following the ts file format are used to make
              TimeArrays suitable for factorization

        Returns
        -------
        values : np.ndarray
            An array of type String suitable for factorization.
        na_value : object
            The value in `values` to consider missing. Not implemented
            yet.
        """
        # TODO: also consider na_value
        vals = to_ts(self, False)
        return np.array(vals), None

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct a TimeArray after factorization.

        Parameters
        ----------
        values : np.ndarray
            An integer ndarray with the factorized values.
        original : TimeArray
            The original TimeArray that factorize was called on.

        See Also
        --------
        factorize
        ExtensionArray.factorize
        """
        # TODO: look into how the original array could be better utilised to
        #       avoid duplication
        return from_ts(values)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate an array/list of TimeArrays.

        All TimeArrays must have compatible data/index shapes, i.e. if the data
        structure is NxD, all D's must be equal. The arrays are concatenated
        along axis 0, the final array will have the shape (N1+...+Nc)xD, where
        c is the number of arrays.

        Parameters
        ----------
        to_concat : array/list of TimeArrays

        Returns
        -------
        TimeArray
            A single array
        """
        if not np.all([isinstance(x, TimeArray) for x in to_concat]):
            # TODO: should we also allow to concatenate a np.array of None
            #       (i.e. missing)?
            raise TypeError("Only TimeArrays can be concatenated.")

        widths = np.array([x.data.shape[1] for x in to_concat])
        ref_width = np.unique(widths[widths > 0])

        if len(ref_width) == 0:
            ref_width = 0
        elif len(ref_width) == 1:
            ref_width = ref_width[0]
        else:
            raise ValueError("Lengths of TimeArrays must be equal.")

        data = [empty((x.shape[0], ref_width)) if w == 0 else x.data
                for w, x in zip(widths, to_concat)]
        tidx = [empty((x.shape[0], ref_width)) if w == 0 else x.time_index
                for w, x in zip(widths, to_concat)]

        return cls(np.vstack(data), np.vstack(tidx))

    # --------------------------------------------------------------------------
    # Interfaces
    # --------------------------------------------------------------------------

    def __getitem__(self, idx):
        """
        Return an item.
        """
        # validate and convert IntegerArray / BooleanArray
        # keys to numpy array, pass-through non-array-like indexers
        idx = pd.api.indexers.check_array_indexer(self, idx)

        sel_data = self.data[idx]
        sel_indx = self.time_index[idx]

        if isinstance(idx, (int, np.int, numbers.Integral)):
            if np.all(np.isnan(sel_data)) and np.all(np.isnan(sel_indx)):
                # Return the missing type if both data and time_index are
                # completely missing
                return None

            return TimeBase(sel_data, sel_indx)
        elif isinstance(idx, (Iterable, slice)):
            if np.all(np.isnan(sel_data)) and np.all(np.isnan(sel_indx)):
                # Return the missing type if both data and time_index are
                # completely missing
                return from_list([None for _ in range(sel_data.shape[0])])

            return self._constructor(sel_data, sel_indx)
        else:
            raise TypeError("Index type not supported", idx)

    def __setitem__(self, key, value):
        """
        Item assignment.
        """
        # validate and convert IntegerArray / BooleanArray
        # keys to numpy array, pass-through non-array-like indexers
        key = pd.api.indexers.check_array_indexer(self, key)

        if value is not None and not isinstance(value, (TimeBase, TimeArray)):
            value = from_list(value)

        # TODO: add dimensionality check
        # TODO: set underlying array to width 0 when all rows are missing
        #       as a result of calling __setitem__ to conform with the behaviour
        #       of __getitem__ when only missing rows are selected

        if value is None or np.all(value.isna()):
            # This is setting all `key` elements to missing
            self.data[key] = np.full((1, ), np.nan, dtype=np.float)
            self.time_index[key] = np.full((1, ), np.nan, dtype=np.float)
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
        # Note that a TimeArray's shape is one-dimensional (to conform with
        # pandas' requirements) and equal to the number of timeseries
        # TODO: shall we add a second property that returns NxD?
        return (self.data.shape[0], )

    @property
    def ndim(self):
        return 1

    @property
    def nbytes(self):
        return self.data.nbytes

    # --------------------------------------------------------------------------
    # general array like compatibility functions
    # --------------------------------------------------------------------------

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Parameters
        ----------
        dtype :
            only included for compatibility reasons with pandas and ignored.

        Returns
        -------
        np.ndarray
            A numpy array of TimeBase objects
        """
        return np.array([None if m else
                         TimeBase(self.data[i], self.time_index[i])
                         for m, i in zip(self.isna(), range(self.data.shape[0]))])

    def __array_ufunc__(self,
                        ufunc: Callable,
                        method: str,
                        *inputs: Any,
                        **kwargs: Any):
        raise NotImplementedError("__array_ufunc__ not implemented yet")

    def __eq__(self, other):
        if isinstance(other, pd.Series):
            comp_data = np.broadcast_to(other.values[np.newaxis, :],
                                        self.data.shape)
            comp_index = np.broadcast_to(other.index.values[np.newaxis, :],
                                         self.time_index.shape)
        else:
            comp_data = other.data
            comp_index = other.time_index

        # TODO: sense check for other types
        # TODO: revisit comparison of TimeArrays with

        return np.all(self.data == comp_data) & \
            np.all(self.time_index == comp_index)

    def __add__(self, o):
        if not isinstance(o, TimeArray):
            return NotImplemented

        if np.all(self.time_index != o.time_index):
            raise ValueError("The time indices of two TimeArrays that should "
                             "be added must be identical.")

        return self._constructor(self.data + o.data, time_index=self.time_index)

    # --------------------------------------------------------------------------
    # pandas compatibility functions
    # --------------------------------------------------------------------------

    def isna(self):
        """
        Detect missing rows

        Missing rows (i.e. missing information on all data and indices) are
        detected.

        Returns
        -------
        a boolean array of whether rows (i.e. data and index) are completely
        empty
        """
        # TODO: revisit definition of missingness
        return rows_na(self.data, axis=1) & rows_na(self.time_index, axis=1)

    def hasna(self):
        """
        Detect missing values

        Missing values (i.e. missing information on **data**) are detected.

        Returns
        -------
        a boolean array of whether any data items in a row are empty
        """
        # TODO: revisit definition of missingness
        return rows_na(self.data, sum_func=np.any, axis=1)

    def astype(self, dtype, copy=True):
        """
        Coerce this type to another dtype

        Parameters
        ----------
        dtype : numpy dtype or pandas type
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and dtype is categorical, the original
            object is returned.
        """
        if isinstance(dtype, TimeDtype):
            if copy:
                return self.copy()
            else:
                return self
        elif dtype is str or \
                (hasattr(dtype, 'kind') and dtype.kind in {'U', 'S'}):
            return np.array(self).astype(dtype)
        elif dtype is object or \
                (hasattr(dtype, 'kind') and dtype.kind in {'O'}):
            return np.array(self)
        raise ValueError(f"TimeArray can only be cast to numpy array with type "
                         f"'object', got type {dtype}")

    def copy(self, *args, **kwargs):
        """
        Copy constructor.
        """
        return self._constructor(self.data.copy(), self.time_index.copy())

    def unique(self):
        """
        Compute the TimeArray of unique timeseries.
        Returns
        -------
        uniques : TimeArray
        """
        # TODO: review, does it make sense to keep such a function in?
        from pandas import factorize
        rows, _ = factorize(np.array(self).astype(str))
        return self[np.unique(rows)]

    def value_counts(self, dropna=True):
        """
        Return a Series containing counts of each unique timeseries.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        # TODO: review, does it make sense to keep such a function in?
        if (dropna):
            factorised, _ = self.dropna()._values_for_factorize()
        else:
            factorised, _ = self._values_for_factorize()

        return pd.Series(factorised).value_counts()

    def take(self, indices, allow_fill=False, fill_value=None):
        """
        Take elements (=rows) from the TimeArray.

        Parameters
        ----------
        indexer : sequence of int
            The indices in `self` to take. The meaning of negative values in
            `indexer` depends on the value of `allow_fill`.
        allow_fill : bool, default False
            How to handle negative values in `indexer`.
            * False: negative values in `indices` indicate positional indices
              from the right. This is similar to
              :func:`numpy.take`.
            * True: negative values in `indices` indicate missing values
              (the default). These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.
        fill_value : object
            The value to use for `indices` that are missing (-1), when
            ``allow_fill=True``.

        Returns
        -------
        TimeArray

        See Also
        --------
        Series.take : Similar method for Series.
        numpy.ndarray.take : Similar method for NumPy arrays.
        """
        # TODO: revisit and simplify
        # Use the take implementation from pandas to get the takes separately
        # for the data and the time indices
        from pandas.api.extensions import take

        if not allow_fill and self.data.shape[0] == 0:
            # Quick-fix to raise the correct error when shape (0,0)
            # TODO: look into why np.ndarray.take() raises the wrong error when
            #       it has shape (0,0)
            raise IndexError("cannot do a non-empty take from an empty axes.")

        data = take(self.data, indices, allow_fill=allow_fill)
        time_index = take(self.time_index, indices, allow_fill=allow_fill)

        if allow_fill and isinstance(fill_value, TimeBase):
            # Fill those indices that were out of range with the fill TimeBase
            indices = np.asarray(indices, dtype=np.intp)
            out_of_range = (indices < 0) | (indices >= data.shape[0])

            data[rows_na(data, axis=1) & out_of_range] = fill_value.data
            time_index[rows_na(time_index, axis=1) & out_of_range] = \
                fill_value.time_index
        elif allow_fill and fill_value is not None:
            TypeError(f"Only TimeBase are allowed as fill values, "
                      f"got {type(fill_value)}")

        if data.shape[0] != 0 and np.all(np.isnan(data)):
            # If the take resulted in a missing TimeArray (note, this is
            # different from a length 0 TimeArray)
            # TODO: put in a separate function if needed elsewhere
            return self._constructor(empty((data.shape[0],0)))

        return self._constructor(data, time_index)

    # --------------------------------------------------------------------------
    # Printing
    # --------------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation.
        """
        from pandas.io.formats.printing import format_object_summary

        template = "{class_name}{data}\nLength: {length}, dtype: {dtype}"
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False, is_justify=False
        ).rstrip(", \n")
        class_name = "<{}>\n".format(self.__class__.__name__)

        return template.format(
            class_name=class_name, data=data, length=len(self), dtype=self.dtype
        )

    # --------------------------------------------------------------------------
    # Time series functionality
    # --------------------------------------------------------------------------

    def tabularise(self, prefix=None, return_array=False):
        """
        Convert TimeArray into a 2-dimensional table

        Parameters
        ----------
        prefix : str, optional
            stub used in the column names, of the form stub + index[i]
        return_array : boolean, default False
            shall the result be returned as np.ndarray

        Returns
        -------
        table
            if `return_array` is False a pd.DataFrame, else np.ndarray
        """
        if prefix is None:
            prefix = "dim"
        if return_array:
            return self.data
        # TODO: throw a naming error when time index isn't equal in each row
        return pd.DataFrame(self.data, columns=[prefix + "_" + str(i)
                                                for i in self.time_index[0]])

    tabularize = tabularise

    def slice_time(self, time_index):
        """
        Slice a TimeArray across the time axis.

        Parameters
        ----------
        time_index : list, np.ndarray, pd.Index
            indices to be included in the slice

        Returns
        -------
        TimeArray

        Raises
        ------
        ValueError
            If the slice results in a ragged Array because different numbers are
            selected across rows
        """
        # TODO: this silently assumes that time indices are the same. Decide how
        #       this function should work if they are not (and the result can be
        #       a ragged/non-equal length array)
        # TODO: currently, this only allows to slice with one condition for all
        #       rows. We might want to expand that to allow for a separate
        #       slice condition per row
        # TODO: add slice as an option for the parameter
        mask = np.isin(self.time_index, time_index)
        width = np.unique(np.sum(mask, axis=1))
        if not len(width) == 1:
            raise ValueError("slicing resulted in unequal number of elements "
                             "between rows; must be equal.")
        data = self.data[mask].reshape(self.shape[0], width[0])
        indx = self.time_index[mask].reshape(self.shape[0], width[0])
        return self._constructor(data, indx)

    def sort_time(self, inplace=False):
        """
        Sort rows by their time index

        If two elements in one row have the same index, sort them by also by
        value.

        Parameters
        ----------
        inplace : boolean, default False
            shall the TimeArray be sorted in memory

        Returns
        -------
        TimeArray
        """
        # TODO: add options to sort in reverse order
        d_ord = np.argsort(self.data, axis=1)
        data = np.take_along_axis(self.data, d_ord, axis=1)
        time_index = np.take_along_axis(self.time_index, d_ord, axis=1)

        t_ord = np.argsort(time_index, axis=1, kind='stable')
        data = np.take_along_axis(data, t_ord, axis=1)
        time_index = np.take_along_axis(time_index, t_ord, axis=1)

        if inplace:
            self.data = data
            self.time_index = time_index
            return self
        else:
            return self._constructor(data, time_index)
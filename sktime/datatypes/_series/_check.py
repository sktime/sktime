"""Machine type checkers for Series scitype.

Exports checkers for Series scitype:

check_dict: dict indexed by pairs of str
  1st element = mtype - str
  2nd element = scitype - str
elements are checker/validation functions for mtype

Function signature of all elements
check_dict[(mtype, scitype)]

Parameters
----------
obj - object to check
return_metadata - bool, optional, default=False
    if False, returns only "valid" return
    if True, returns all three return objects
    if str, list of str, metadata return dict is subset to keys in return_metadata
var_name: str, optional, default="obj" - name of input in error messages

Returns
-------
valid: bool - whether obj is a valid object of mtype/scitype
msg: str - error message if object is not valid, otherwise None
        returned only if return_metadata is True
metadata: dict - metadata about obj if valid, otherwise None
        returned only if return_metadata is True
    fields:
        "is_univariate": bool, True iff series has one variable
        "is_equally_spaced": bool, True iff series index is equally spaced
        "is_empty": bool, True iff series has no variables or no instances
        "has_nans": bool, True iff the series contains NaN values
        "n_features": int, number of variables in series
        "feature_names": list of int or object, names of variables in series
"""

import numpy as np
import pandas as pd

from sktime.datatypes._base._common import _req
from sktime.datatypes._base._common import _ret as ret
from sktime.datatypes._dtypekind import (
    _get_feature_kind,
    _get_series_dtypekind,
    _pandas_dtype_to_kind,
)
from sktime.datatypes._series._base import ScitypeSeries
from sktime.utils.validation.series import is_in_valid_index_types

VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)

# whether the checks insist on freq attribute is set
FREQ_SET_CHECK = False


class SeriesPdDataFrame(ScitypeSeries):
    """Data type: pandas.DataFrame based specification of single time series.

    Name: ``"pd.DataFrame"``

    Short description:

    a uni- or multivariate ``pandas.DataFrame``,
    with rows = time points, cols = variables

    Long description:

    The ``"pd.DataFrame"`` :term:`mtype` is a concrete specification
    that implements the ``Series`` :term:`scitype`, i.e., the abstract
    type of a single time series.

    An object ``obj: pandas.DataFrame`` follows the specification iff:

    * structure convention: ``obj.index`` must be monotonic,
      and one of ``Int64Index``, ``RangeIndex``, ``DatetimeIndex``, ``PeriodIndex``.
    * variables: columns of ``obj`` correspond to different variables
    * variable names: column names ``obj.columns``
    * time points: rows of ``obj`` correspond to different, distinct time points
    * time index: ``obj.index`` is interpreted as the time index.

    Capabilities:

    * cannot represent multivariate series
    * can represent unequally spaced series
    * can represent missing values

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "pd.DataFrame",  # any string
        "name_python": "series_pd_df",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
        "python_type": "pandas.DataFrame",
        "capability:multivariate": True,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        return _check_pddataframe_series(obj, return_metadata, var_name)


def _check_pddataframe_series(obj, return_metadata=False, var_name="obj"):
    """Check if obj is a pandas.DataFrame based specification of single time series."""
    metadata = dict()

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pandas.DataFrame, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # check to delineate from nested_univ mtype (Panel)
    # pd.DataFrame mtype allows object dtype,
    # but if we allow object dtype with pd.Series entries,
    # the mtype becomes ambiguous, i.e., non-delineable from nested_univ
    if np.prod(obj.shape) > 0 and isinstance(obj.iloc[0, 0], (pd.Series, pd.DataFrame)):
        msg = f"{var_name} cannot contain nested pd.Series or pd.DataFrame"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj.columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns.to_list()
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "pd.DataFrame")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_series_dtypekind(obj, "pd.DataFrame")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    # check that columns are unique
    if not obj.columns.is_unique:
        msg = f"{var_name} must have unique column indices, but found {obj.columns}"
        return ret(False, msg, None, return_metadata)

    # check whether the time index is of valid type
    if not is_in_valid_index_types(index):
        msg = (
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} or integer index instead."
        )
        return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not index.is_monotonic_increasing:
        msg = (
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )
        return ret(False, msg, None, return_metadata)

    if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
        if index.freq is None:
            msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
            return ret(False, msg, None, return_metadata)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = _index_equally_spaced(index)
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    return ret(True, None, metadata, return_metadata)


class SeriesPdSeries(ScitypeSeries):
    """Data type: pandas.Series based specification of single time series.

    Name: ``"pd.Series"``

    Short description:

    a (univariate) ``pandas.Series``,
    with entries corresponding to different time points

    Long description:

    The ``"pd.Series"`` :term:`mtype` is a concrete specification
    that implements the ``Series`` :term:`scitype`, i.e., the abstract
    type of a single time series.

    An object ``obj: pandas.Series`` follows the specification iff:

    * structure convention: ``obj.index`` must be monotonic,
      and one of ``Int64Index``, ``RangeIndex``, ``DatetimeIndex``, ``PeriodIndex``.
    * variables: there is a single variable, corresponding to the values of ``obj``.
      Only univariate series can be represented.
    * variable names: by default, there is no column name.
      If needed, a variable name can be provided as ``obj.name``.
    * time points: entries of ``obj`` correspond to different, distinct time points
    * time index: ``obj.index`` is interpreted as a time index.

    Capabilities:

    * cannot represent multivariate series
    * can represent unequally spaced series
    * can represent missing values

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "pd.Series",  # any string
        "name_python": "series_pd_sr",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
        "python_type": "pandas.Series",
        "capability:multivariate": False,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        metadata = dict()

        if not isinstance(obj, pd.Series):
            msg = f"{var_name} must be a pandas.Series, found {type(obj)}"
            return ret(False, msg, None, return_metadata)

        # we now know obj is a pd.Series
        index = obj.index
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(index) < 1
        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = True
        if _req("n_features", return_metadata):
            metadata["n_features"] = 1
        if _req("feature_names", return_metadata):
            if not hasattr(obj, "name") or obj.name is None:
                metadata["feature_names"] = [0]
            else:
                metadata["feature_names"] = [obj.name]
        if _req("dtypekind_dfip", return_metadata):
            metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "pd.Series")
        if _req("feature_kind", return_metadata):
            dtype_kind = _get_series_dtypekind(obj, "pd.Series")
            metadata["feature_kind"] = _get_feature_kind(dtype_kind)

        # check whether the time index is of valid type
        if not is_in_valid_index_types(index):
            msg = (
                f"{type(index)} is not supported for {var_name}, use "
                f"one of {VALID_INDEX_TYPES} or integer index instead."
            )
            return ret(False, msg, None, return_metadata)

        # Check time index is ordered in time
        if not index.is_monotonic_increasing:
            msg = (
                f"The (time) index of {var_name} must be sorted monotonically "
                f"increasing, but found: {index}"
            )
            return ret(False, msg, None, return_metadata)

        if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
            if index.freq is None:
                msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
                return ret(False, msg, None, return_metadata)

        # check whether index is equally spaced or if there are any nans
        #   compute only if needed
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = _index_equally_spaced(index)
        if _req("has_nans", return_metadata):
            metadata["has_nans"] = obj.isna().values.any()

        return ret(True, None, metadata, return_metadata)


class SeriesNp2D(ScitypeSeries):
    """Data type: 2D np.ndarray based specification of single time series.

    Name: ``"np.ndarray"``

    Short description:

    a 2D ``numpy.ndarray``, with rows = time points, cols = variables

    Long description:

    The ``"np.ndarray"`` :term:`mtype` is a concrete specification
    that implements the ``Series`` :term:`scitype`, i.e., the abstract
    type of a single time series.

    An object ``obj: numpy.ndarray`` follows the specification iff:

    * structure convention: ``obj`` must be 2D, i.e., ``obj.shape`` must have length 2.
      This is also true for univariate time series.
    * variables: variables correspond to columns of ``obj``.
    * variable names: the ``"np.ndarray"`` mtype cannot represent variable names.
    * time points: the rows of ``obj`` correspond to different, distinct time points.
    * time index: The time index is implicit and by-convention.
      The ``i``-th row (for an integer ``i``) is interpreted as an observation
      at the time point ``i``. That is, the index is always interpreted as zero-indexed
      integer.

    Capabilities:

    * can represent multivariate series
    * cannot represent unequally spaced series
    * can represent missing values

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "np.ndarray",  # any string
        "name_python": "series_pd_np",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "numpy",
        "python_type": "numpy.ndarray",
        "capability:multivariate": True,
        "capability:unequally_spaced": False,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        metadata = dict()

        if not isinstance(obj, np.ndarray):
            msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
            return ret(False, msg, None, return_metadata)

        if len(obj.shape) == 2:
            # we now know obj is a 2D np.ndarray
            if _req("is_empty", return_metadata):
                metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
            if _req("is_univariate", return_metadata):
                metadata["is_univariate"] = obj.shape[1] < 2
            if _req("n_features", return_metadata):
                metadata["n_features"] = obj.shape[1]
            if _req("feature_names", return_metadata):
                metadata["feature_names"] = list(range(obj.shape[1]))
            if _req("dtypekind_dfip", return_metadata):
                metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "numpy")
            if _req("feature_kind", return_metadata):
                dtype_kind = _get_series_dtypekind(obj, "numpy")
                metadata["feature_kind"] = _get_feature_kind(dtype_kind)
        elif len(obj.shape) == 1:
            # we now know obj is a 1D np.ndarray
            if _req("is_empty", return_metadata):
                metadata["is_empty"] = len(obj) < 1
            if _req("is_univariate", return_metadata):
                metadata["is_univariate"] = True
            if _req("n_features", return_metadata):
                metadata["n_features"] = 1
            if _req("feature_names", return_metadata):
                metadata["feature_names"] = [0]
            if _req("dtypekind_dfip", return_metadata):
                metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "numpy")
            if _req("feature_kind", return_metadata):
                dtype_kind = _get_series_dtypekind(obj, "numpy")
                metadata["feature_kind"] = _get_feature_kind(dtype_kind)
        else:
            msg = (
                f"{var_name} must be 1D or 2D numpy.ndarray, "
                f"but found {len(obj.shape)}D"
            )
            return ret(False, msg, None, return_metadata)

        # np.arrays are considered equally spaced by assumption
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = True

        # check whether there any nans; compute only if requested
        if _req("has_nans", return_metadata):
            metadata["has_nans"] = pd.isnull(obj).any()

        return ret(True, None, metadata, return_metadata)


def _index_equally_spaced(index):
    """Check whether pandas.index is equally spaced.

    Parameters
    ----------
    index: pandas.Index. Must be one of:
        pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex

    Returns
    -------
    equally_spaced: bool - whether index is equally spaced
    """
    if not is_in_valid_index_types(index):
        raise TypeError(f"index must be one of {VALID_INDEX_TYPES} or integer index")

    # empty, single and two-element indices are equally spaced
    if len(index) < 3:
        return True

    # RangeIndex is always equally spaced
    if isinstance(index, pd.RangeIndex):
        return True

    if isinstance(index, pd.PeriodIndex):
        return index.is_full

    # we now treat a necessary condition for being equally spaced:
    # the first two spaces are equal. From now on, we know this.
    if index[1] - index[0] != index[2] - index[1]:
        return False

    # another necessary condition for equally spaced:
    # index span is number of spaces times first space
    n = len(index)
    if index[n - 1] - index[0] != (n - 1) * (index[1] - index[0]):
        return False

    # fallback for all other cases:
    # in general, we need to compute all differences and check explicitly
    # CAVEAT: this has a comparabily long runtime and high memory usage
    diffs = np.diff(index)
    all_equal = np.all(diffs == diffs[0])

    return all_equal


class SeriesXarray(ScitypeSeries):
    """Data type: xarray based specification of single time series.

    Name: ``xr.DataArray``

    Short description:

    An ``xarray.DataArray`` representing a single time series, where:

    - Each row corresponds to a time point.
    - Columns represent variables or features.
    - Coordinates provide additional metadata for the time index and variables.

    Long description:

    The ``xr.DataArray`` :term:``mtype`` is a concrete specification
    that implements the ``Series`` :term:``scitype``, i.e., the abstract
    type for time series data.

    An object ``obj: xarray.DataArray`` follows the specification iff:

    * structure convention:

      - ``obj`` is a 2D array-like structure with shape ``(n_timepoints, n_features)``.
      - ``obj.coords`` must include:

        - A time-like index (``dim_0``) which is either ``Int64Index``, ``RangeIndex``,
          ``DatetimeIndex``, or ``PeriodIndex``, and it must be monotonic.
        - A variable-like index (``dim_1``) for feature/variable names (optional).

    * time index:

      - The ``dim_0`` coordinate is interpreted as the time index.

    * time points:

      - Each row of ``obj`` represents a single time point.
      - Rows with the same ``dim_0`` value correspond to the same time point.

    * variables:

      - Columns represent different variables (or features).
      - Column names are stored in ``dim_1`` if present.

    * variable names:

      - The variable names are the column names (``dim_1``), if present.

    * metadata:

      - Additional metadata (e.g., attributes) may be included in ``obj.attrs``.

    Capabilities:

    * can represent univariate or multivariate time series
    * requires equally spaced time points (if time index is specified)
    * supports missing values
    * cannot represent series with differing sets of variables

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "xr.DataArray",  # any string
        "name_python": "series_xarray",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "xarray",
        "python_type": "xarray.DataArray",
        "capability:multivariate": True,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        import xarray as xr

        metadata = {}

        if not isinstance(obj, xr.DataArray):
            msg = f"{var_name} must be a xarray.DataArray, found {type(obj)}"
            return ret(False, msg, None, return_metadata)

        # we now know obj is a xr.DataArray
        if len(obj.dims) > 2:  # Without multi indexing only two dimensions are possible
            msg = f"{var_name} must have two or less dimension, found {type(obj.dims)}"
            return ret(False, msg, None, return_metadata)

        # The first dimension is the index of the time series in sktimelen
        index = obj.indexes[obj.dims[0]]

        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(index) < 1 or len(obj.values) < 1
        # The second dimension is the set of columns
        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = len(obj.dims) == 1 or len(obj[obj.dims[1]]) < 2
        if len(obj.dims) == 1:
            if _req("n_features", return_metadata):
                metadata["n_features"] = 1
            if _req("feature_names", return_metadata):
                metadata["feature_names"] = [0]
        else:
            if _req("n_features", return_metadata):
                metadata["n_features"] = len(obj[obj.dims[1]])
            if _req("feature_names", return_metadata):
                metadata["feature_names"] = obj.indexes[obj.dims[1]].to_list()

        if _req("dtypekind_dfip", return_metadata):
            metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "xarray")
        if _req("feature_kind", return_metadata):
            dtype_kind = _get_series_dtypekind(obj, "xarray")
            metadata["feature_kind"] = _get_feature_kind(dtype_kind)

        # check that columns are unique
        if not len(obj.dims) == len(set(obj.dims)):
            msg = f"{var_name} must have unique column indices, but found {obj.dims}"
            return ret(False, msg, None, return_metadata)

        # check whether the time index is of valid type
        if not is_in_valid_index_types(index):
            msg = (
                f"{type(index)} is not supported for {var_name}, use "
                f"one of {VALID_INDEX_TYPES} or integer index instead."
            )
            return ret(False, msg, None, return_metadata)

        # Check time index is ordered in time
        if not index.is_monotonic_increasing:
            msg = (
                f"The (time) index of {var_name} must be sorted "
                f"monotonically increasing, but found: {index}"
            )
            return ret(False, msg, None, return_metadata)

        if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
            if index.freq is None:
                msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
                return ret(False, msg, None, return_metadata)

        # check whether index is equally spaced or if there are any nans
        #   compute only if needed
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = _index_equally_spaced(index)
        if _req("has_nans", return_metadata):
            metadata["has_nans"] = obj.isnull().values.any()

        return ret(True, None, metadata, return_metadata)


class SeriesDask(ScitypeSeries):
    """Data type: dask.DataFrame based specification of single time series.

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "dask_series",  # any string
        "name_python": "series_dask",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "dask",
        "python_type": "dask.dataframe",
        "capability:multivariate": True,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        from sktime.datatypes._adapter.dask_to_pd import check_dask_frame

        return check_dask_frame(
            obj=obj,
            return_metadata=return_metadata,
            var_name=var_name,
            freq_set_check=FREQ_SET_CHECK,
            scitype="Series",
        )


class SeriesPolarsEager(ScitypeSeries):
    """Data type: polars.DataFrame based specification of single time series.

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "pl.DataFrame",  # any string
        "name_python": "series_polars_eager",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "polars",
        "python_type": "polars.DataFrame",
        "capability:multivariate": True,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        from sktime.datatypes._adapter.polars import check_polars_frame

        metadict = check_polars_frame(
            obj=obj,
            return_metadata=return_metadata,
            var_name=var_name,
            scitype="Series",
        )

        if isinstance(metadict, tuple) and metadict[0]:
            # update dict with Series specific keys
            if _req("is_equally_spaced", return_metadata):
                metadict[2]["is_equally_spaced"] = "NA"

        return metadict


class SeriesGluontsList(ScitypeSeries):
    """Data type: gluonts ListDataset based specification of single time series.

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "gluonts_ListDataset_series",  # any string
        "name_python": "series_gluonts_listdataset",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "gluonts",
        "python_type": "list",
        "capability:multivariate": True,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        from sktime.datatypes._dtypekind import DtypeKind

        metadata = dict()

        if (
            not isinstance(obj, list)
            or not isinstance(obj[0], dict)
            or "target" not in obj[0]
            or len(obj[0]["target"]) > 1
        ):
            msg = f"{var_name} must be a gluonts.ListDataset, found {type(obj)}"
            return ret(False, msg, None, return_metadata)

        # Check if there are no time series in the ListDataset
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(obj) < 1

        if _req("is_univariate", return_metadata):
            # Check first if the ListDataset is empty
            if len(obj) < 1:
                metadata["is_univariate"] = True

            # Check the first time-series for total features
            else:
                metadata["is_univariate"] = obj[0]["target"].shape[1] == 1

        req_n_feat = ["n_features", "feature_names", "feature_kind", "dtypekind_dfip"]
        if _req(req_n_feat, return_metadata):
            # Check first if the ListDataset is empty
            if len(obj) < 1:
                n_features = 0
            else:
                n_features = obj[0]["target"].shape[1]

        if _req("n_features", return_metadata):
            metadata["n_features"] = n_features

        if _req(["dtypekind_dfip", "feature_kind"], return_metadata):
            dtypes = []

            # Each entry in a ListDataset is formed with an ndarray.
            # Basing off definitions in _dtypekind, assigning values of FLOAT

            dtypes.extend([DtypeKind.FLOAT] * len(obj))

            if _req("dtypekind_dfip", return_metadata):
                metadata["dtypekind_dfip"] = dtypes

            if _req("feature_kind", return_metadata):
                metadata["feature_kind"] = _get_feature_kind(dtypes)

        if _req("n_instances", return_metadata):
            metadata["n_instances"] = 1

        if _req("feature_names", return_metadata):
            metadata["feature_names"] = [f"value_{i}" for i in range(n_features)]

        for series in obj:
            # check that no dtype is object
            if series["target"].dtype == "object":
                msg = f"{var_name} should not have column of 'object' dtype"
                return ret(False, msg, None, return_metadata)

        # Check if a valid Frequency is set
        if FREQ_SET_CHECK and len(obj) >= 1:
            if obj[0].freq is None:
                msg = f"{var_name} has no freq attribute set."
                return ret(False, msg, None, return_metadata)

        # For a GluonTS ListDataset, only a start date and frequency is set
        # so everything should thus be equally spaced
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = True

        if _req("has_nans", return_metadata):
            for series in obj:
                metadata["has_nans"] = pd.isnull(ScitypeSeries["target"]).any()

                # Break out if at least 1 time series has NaN values
                if metadata["has_nans"]:
                    break

        return ret(True, None, metadata, return_metadata)


class SeriesGluontsPandas(ScitypeSeries):
    """Data type: gluonts PandasDataset based specification of single time series.

    Parameters
    ----------
    is_univariate: bool
        True iff series has one variable
    is_equally_spaced: bool
        True iff series index is equally spaced
    is_empty: bool
        True iff series has no variables or no instances
    has_nans: bool
        True iff the series contains NaN values
    n_features: int
        number of variables in series
    feature_names: list of int or object
        names of variables in series
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Series",
        "name": "gluonts_PandasDataset_series",  # any string
        "name_python": "series_gluonts_pandasdataset",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "gluonts",
        "python_type": "gluonts.PandasDataset",
        "capability:multivariate": True,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
    }

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        from gluonts.dataset.pandas import PandasDataset

        metadata = dict()

        # Check for type correctness
        if not isinstance(obj, PandasDataset):
            msg = f"{var_name} must be a gluonts.PandasDataset, found {type(obj)}"
            return ret(False, msg, None, return_metadata)

        # Convert to a pandas DF for easier checks
        df = obj._data_entries.iterable

        # Checking if the DataFrame is stored in the appropriate place
        if (
            not isinstance(df, list)
            or not isinstance(df[0], tuple)
            or not isinstance(df[0][1], pd.DataFrame)
        ):
            msg = f"{var_name} was not formed with a single-instance pandas DataFrame"
            return ret(False, msg, None, return_metadata)

        df = df[0][1]

        # Check if there are no values
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(obj._data_entries) == 0

        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = len(df.columns) == 1

        if _req("n_features", return_metadata):
            metadata["n_features"] = 1

        if _req("n_instances", return_metadata):
            metadata["n_instances"] = 1

        if _req("n_panels", return_metadata):
            metadata["n_panels"] = 1

        if _req("feature_names", return_metadata):
            metadata["feature_names"] = df.columns

        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = True

        if _req("has_nans", return_metadata):
            metadata["has_nans"] = df.isna().any().any()

        if _req("dtypekind_dfip", return_metadata):
            index_cols_count = len(df.columns)

            # slicing off additional index columns
            dtype_list = df.dtypes.to_list()[index_cols_count:]

            metadata["dtypekind_dfip"] = _pandas_dtype_to_kind(dtype_list)

        if _req("feature_kind", return_metadata):
            dtype_list = df.dtypes.to_list()[index_cols_count:]
            dtype_kind = _pandas_dtype_to_kind(dtype_list)
            metadata["feature_kind"] = _get_feature_kind(dtype_kind)

        return ret(True, None, metadata, return_metadata)

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
        "is_univariate": bool, True iff all series in panel have one variable
        "is_equally_spaced": bool, True iff all series indices are equally spaced
        "is_equal_length": bool, True iff all series in panel are of equal length
        "is_empty": bool, True iff one or more of the series in the panel are empty
        "is_one_series": bool, True iff there is only one series in the panel
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances in the panel
        "n_features": int, number of variables in series
        "feature_names": list of int or object, names of variables in series
"""

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import is_nested_object

from sktime.datatypes._base._common import _req, _ret
from sktime.datatypes._dtypekind import (
    _get_feature_kind,
    _get_panel_dtypekind,
    _pandas_dtype_to_kind,
)
from sktime.datatypes._panel._base import ScitypePanel
from sktime.datatypes._series._check import (
    _check_pddataframe_series,
    _index_equally_spaced,
)
from sktime.utils.validation.series import is_in_valid_index_types, is_integer_index

VALID_MULTIINDEX_TYPES = (pd.RangeIndex, pd.Index)
VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)


def is_in_valid_multiindex_types(x) -> bool:
    """Check that the input type belongs to the valid multiindex types."""
    return isinstance(x, VALID_MULTIINDEX_TYPES) or is_integer_index(x)


def _list_all_equal(obj):
    """Check whether elements of list are all equal.

    Parameters
    ----------
    obj: list - assumed, not checked

    Returns
    -------
    bool, True if elements of obj are all equal
    """
    if len(obj) < 2:
        return True

    return np.all([s == obj[0] for s in obj])


class PanelDfList(ScitypePanel):
    """Data type: list-of-pandas.DataFrame based specification of panel of time series.

    Name: ``"df-list"``

    Short description:

    a ``list`` of ``pandas.DataFrame``,
    with list index = instances, data frame rows = time points,
    data frame cols = variables

    Long description:

    The ``"df-list"`` :term:`mtype` is a concrete specification
    that implements the ``Panel`` :term:`scitype`, i.e., the abstract
    type of a collection of time series.

    An object ``obj: list`` follows the specification iff:

    * structure convention: ``obj`` must be a list of ``pandas.DataFrames``.
      Individual list elements of `obj` must follow the ``"pd.DataFrame"``
      mtype convention for the ``"Series"`` scitype.
    * instances: instances correspond to different list elements of ``obj``.
    * instance index: the instance index of an instance is the list index at which
      it is located in ``obj``. That is, the data at ``obj[i]``
      correspond to observations of the instance with index ``i``.
    * time points: rows of ``obj[i]`` correspond to different, distinct time points,
      at which instance ``i`` is observed.
    * time index: ``obj[i].index`` is interpreted as the time index for instance ``i``.
    * variables: columns of ``obj[i]`` correspond to different variables available
      for instance ``i``.
    * variable names: column names ``obj[i].columns`` are the names of variables
      available for instance ``i``.

    Capabilities:

    * can represent panels of multivariate series
    * can represent unequally spaced series
    * can represent panels of unequally supported series
    * can represent panels of series with different sets of variables
    * can represent missing values

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "df-list",  # any string
        "name_python": "panel_df_list",  # lower_snake_case
        "name_aliases": [],
        "description": "list of pd.DataFrame",
        "python_version": None,
        "python_dependencies": "pandas",
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
        return _check_dflist_panel(
            obj, return_metadata=return_metadata, var_name=var_name
        )


def _check_dflist_panel(obj, return_metadata=False, var_name="obj"):
    """Check if obj is a list of pandas.DataFrame according to df-list specification."""
    if not isinstance(obj, list):
        msg = f"{var_name} must be list of pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    n = len(obj)

    bad_inds = [i for i in range(n) if not isinstance(obj[i], pd.DataFrame)]

    if len(bad_inds) > 0:
        msg = f"{var_name}[i] must pd.DataFrame, but found other types at i={bad_inds}"
        return _ret(False, msg, None, return_metadata)

    check_res = [_check_pddataframe_series(s, return_metadata=True) for s in obj]
    bad_inds = [i for i in range(n) if not check_res[i][0]]

    if len(bad_inds) > 0:
        msg = f"{var_name}[i] must be Series of mtype pd.DataFrame, not at i={bad_inds}"
        return _ret(False, msg, None, return_metadata)

    metadata = dict()
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = np.all(
            [res[2]["is_univariate"] for res in check_res]
        )
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = np.all(
            [res[2]["is_equally_spaced"] for res in check_res]
        )
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = _list_all_equal([len(s) for s in obj])
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = np.any([res[2]["is_empty"] for res in check_res])
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = np.any([res[2]["has_nans"] for res in check_res])
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = n == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = n
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj[0].columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj[0].columns.to_list()
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_panel_dtypekind(obj, "df-list")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_panel_dtypekind(obj, "df-list")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    return _ret(True, None, metadata, return_metadata)


class PanelNp3D(ScitypePanel):
    """Data type: 3D np.ndarray based specification of panel of time series.

    Name: ``"numpy3D"``

    Short description:

    a 3D ``numpy.ndarray``,
    with axis 0 = instances, axis 1 = variables, axis 2 = time points

    Long description:

    The ``"numpy3D"`` :term:`mtype` is a concrete specification
    that implements the ``Panel`` :term:`scitype`, i.e., the abstract
    type of a collection of time series.

    An object ``obj: numpy.ndarray`` follows the specification iff:

    * structure convention: ``obj`` must be 3D, i.e., ``obj.shape`` must have length 3.
    * instances: instances correspond to axis 0 elements of ``obj``.
    * instance index: the instance index is implicit and by-convention.
      The ``i``-th element of axis 0 (for an integer ``i``) is interpreted as indicative
      of observing instance `i`.
    * variables: variables correspond to axis 1 elements of ``obj``.
    * variable names: the ``"numpy3D"`` mtype cannot represent variable names.
    * time points: time points correspond to axis 2 elements of ``obj``.
    * time index: the time index is implicit and by-convention.
      The ``i``-th element of axis 2 (for an integer ``i``) is interpreted
      as an observation at the time point ``i``.

    Capabilities:

    * can represent panels of multivariate series
    * cannot represent unequally spaced series
    * cannot represent panels of unequally supported series
    * cannot represent panels of series with different sets of variables
    * can represent missing values

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "numpy3D",  # any string
        "name_python": "panel_np_3d",  # lower_snake_case
        "name_aliases": [],
        "description": "3D np.array of format (n_instances, n_columns, n_timepoints)",
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
        return _check_numpy3d_panel(
            obj, return_metadata=return_metadata, var_name=var_name
        )


def _check_numpy3d_panel(obj, return_metadata=False, var_name="obj"):
    """Check if obj is a numpy.ndarray according to numpy3D specification."""
    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not len(obj.shape) == 3:
        msg = f"{var_name} must be a 3D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 3D np.ndarray
    metadata = dict()
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1 or obj.shape[2] < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.shape[1] < 2
    # np.arrays are considered equally spaced and equal length by assumption
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = True
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = True

    if _req("n_instances", return_metadata):
        metadata["n_instances"] = obj.shape[0]
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = obj.shape[0] == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True
    if _req("n_features", return_metadata):
        metadata["n_features"] = obj.shape[1]
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = list(range(obj.shape[1]))
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_panel_dtypekind(obj, "numpy3D")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_panel_dtypekind(obj, "numpy3D")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    # check whether there any nans; only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()

    return _ret(True, None, metadata, return_metadata)


class PanelPdMultiIndex(ScitypePanel):
    """Data type: MultiIndex-ed pd.DataFrame specification of panel of time series.

    Name: ``"pd-multiindex"``

    Short description:

    a ``pandas.DataFrame``, with row multi-index (instances, time), cols = variables

    Long description:

    The ``"pd-multiindex"`` :term:`mtype` is a concrete specification
    that implements the ``Panel`` :term:`scitype`, i.e., the abstract
    type of a collection of time series.

    An object ``obj: pd.DataFrame`` follows the specification iff:

    * structure convention: ``obj.index`` must be a pair multi-index of
      type ``(Index, t)``, where ``t`` is one of ``Int64Index``, ``RangeIndex``,
      ``DatetimeIndex``, ``PeriodIndex`` and monotonic.
      ``obj.index`` must have two levels (can be named or not).
    * instance index: the first element of pairs in ``obj.index`` (0-th level value)
      is interpreted as an instance index, we call it "instance index" below.
    * instances: rows with the same "instance index" index value correspond to the same
      instance; rows with different "instance index" values correspond
      to different instances.
    * time index: the second element of pairs in ``obj.index`` (1-st level value)
      is interpreted as a time index, we call it "time index" below.
    * time points: rows of ``obj`` with the same "time index" value correspond
      correspond to the same time point; rows of `obj` with different "time index"
      index correspond to the different time points.
    * variables: columns of ``obj`` correspond to different variables
    * variable names: column names ``obj.columns``

    Capabilities:

    * can represent panels of multivariate series
    * can represent unequally spaced series
    * can represent panels of unequally supported series
    * cannot represent panels of series with different sets of variables
    * can represent missing values

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "pd-multiindex",  # any string
        "name_python": "panel_pd_df",  # lower_snake_case
        "name_aliases": [],
        "description": "pd.DataFrame with multi-index (instances, timepoints)",
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
        return _check_pdmultiindex_panel(
            obj, return_metadata=return_metadata, var_name=var_name
        )


def _check_pdmultiindex_panel(obj, return_metadata=False, var_name="obj", panel=True):
    """Check if obj is a pandas.DataFrame according to pd-multiindex specification."""
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not isinstance(obj.index, pd.MultiIndex):
        msg = f"{var_name} must have a MultiIndex, found {type(obj.index)}"
        return _ret(False, msg, None, return_metadata)

    # check to delineate from nested_univ mtype (Hierarchical)
    # pd.DataFrame mtype allows object dtype,
    # but if we allow object dtype with Panel entries,
    # the mtype becomes ambiguous, i.e., non-delineable from nested_univ
    if np.prod(obj.shape) > 0 and isinstance(obj.iloc[0, 0], (pd.Series, pd.DataFrame)):
        msg = f"{var_name} cannot contain nested pd.Series or pd.DataFrame"
        return _ret(False, msg, None, return_metadata)

    index = obj.index

    # check that columns are unique
    col_names = obj.columns
    if not col_names.is_unique:
        msg = f"{var_name} must have unique column indices, but found {col_names}"
        return _ret(False, msg, None, return_metadata)

    # check that there are precisely two index levels
    nlevels = index.nlevels
    if panel is True and not nlevels == 2:
        msg = f"{var_name} must have a MultiIndex with 2 levels, found {nlevels}"
        return _ret(False, msg, None, return_metadata)
    elif panel is False and not nlevels > 2:
        msg = (
            f"{var_name} must have a MultiIndex with 3 or more levels, found {nlevels}"
        )
        return _ret(False, msg, None, return_metadata)

    # check whether the time index is of valid type
    if not is_in_valid_index_types(index.levels[-1]):
        msg = (
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} or integer index instead."
        )
        return _ret(False, msg, None, return_metadata)

    # check instance index being integer or range index
    inst_inds = index.levels[0]
    if not is_in_valid_multiindex_types(inst_inds):
        msg = (
            f"instance index (first/highest index) must be {VALID_MULTIINDEX_TYPES}, "
            f"integer index, but found {type(inst_inds)}"
        )
        return _ret(False, msg, None, return_metadata)

    # check if time index is monotonic increasing for each group
    if not index.is_monotonic_increasing:
        index_frame = index.to_frame()
        if (
            not index_frame.groupby(level=list(range(index.nlevels - 1)), sort=False)[
                index_frame.columns[-1]
            ]
            .is_monotonic_increasing.astype(bool)
            .all()
        ):
            msg = (
                f"The (time) index of {var_name} must be sorted monotonically "
                f"increasing. Use {var_name}.sort_index() to sort the index, or "
                f"{var_name}.duplicated() to find duplicates."
            )
            return _ret(False, msg, None, return_metadata)

    metadata = dict()

    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj.columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns.to_list()
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_panel_dtypekind(obj, "pd-multiindex")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_panel_dtypekind(obj, "pd-multiindex")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    requires_series_grps = [
        "n_instances",
        "is_one_series",
        "is_equal_length",
        "is_equally_spaced",
    ]
    if _req(requires_series_grps, return_metadata):
        levels = list(range(index.nlevels - 1))
        if len(levels) == 1:
            levels = levels[0]
        series_groups = obj.groupby(level=levels, sort=False)
        n_series = series_groups.ngroups

        if _req("n_instances", return_metadata):
            metadata["n_instances"] = n_series
        if _req("is_one_series", return_metadata):
            metadata["is_one_series"] = n_series == 1
        if _req("is_equal_length", return_metadata):
            metadata["is_equal_length"] = _list_all_equal(
                series_groups.size().to_numpy()
            )
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = all(
                _index_equally_spaced(group.index.get_level_values(-1))
                for _, group in series_groups
            )

    requires_panel_grps = ["n_panels", "is_one_panel"]
    if _req(requires_panel_grps, return_metadata):
        if panel:
            n_panels = 1
        else:
            panel_groups = obj.groupby(level=list(range(index.nlevels - 2)), sort=False)
            n_panels = panel_groups.ngroups

        if _req("n_panels", return_metadata):
            metadata["n_panels"] = n_panels
        if _req("is_one_panel", return_metadata):
            metadata["is_one_panel"] = n_panels == 1

    return _ret(True, None, metadata, return_metadata)


def are_columns_nested(X):
    """Check whether any cells have nested structure in each DataFrame column.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to check for nested data structures.

    Returns
    -------
    any_nested : bool
        If True, at least one column is nested.
        If False, no nested columns.
    """
    any_nested = any(is_nested_object(series) for _, series in X.items())
    return any_nested


def _nested_dataframe_has_unequal(X: pd.DataFrame) -> bool:
    """Check whether an input nested DataFrame of Series has unequal length series.

    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series

    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    rows = len(X)
    cols = len(X.columns)
    s = X.iloc[0, 0]
    length = len(s)

    for i in range(0, rows):
        for j in range(0, cols):
            s = X.iloc[i, j]
            temp = len(s)
            if temp != length:
                return True
    return False


def _nested_dataframe_has_nans(X: pd.DataFrame) -> bool:
    """Check whether an input pandas of Series has nans.

    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series

    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    cases = len(X)
    dimensions = len(X.columns)
    for i in range(cases):
        for j in range(dimensions):
            s = X.iloc[i, j]
            if hasattr(s, "size"):
                for k in range(s.size):
                    if pd.isna(s.iloc[k]):
                        return True
            elif pd.isna(s):
                return True
    return False


class PanelNestedDf(ScitypePanel):
    """Data type: nested pandas.DataFrame based specification of panel of time series.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "nested_univ",  # any string
        "name_python": "panel_pd_nested",  # lower_snake_case
        "name_aliases": [],
        "description": "pd.DataFrame with one column per variable, pd.Series in cells",
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
        return _is_nested_dataframe(
            obj, return_metadata=return_metadata, var_name=var_name
        )


def _is_nested_dataframe(obj, return_metadata=False, var_name="obj"):
    """Check whether the input is a nested DataFrame.

    To allow for a mixture of nested and primitive columns types the
    the considers whether any column is a nested np.ndarray or pd.Series.

    Column is consider nested if any cells in column have a nested structure.

    Parameters
    ----------
    X: Input that is checked to determine if it is a nested DataFrame.

    Returns
    -------
    bool: Whether the input is a nested DataFrame
    """
    # If not a DataFrame we know is_nested_dataframe is False
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)
    # Otherwise we'll see if any column has a nested structure in first row
    else:
        if not all([i == "object" for i in obj.dtypes]):
            msg = f"{var_name} All columns must be object, found {type(obj)}"
            return _ret(False, msg, None, return_metadata)

        if not are_columns_nested(obj):
            msg = f"{var_name} entries must be pd.Series"
            return _ret(False, msg, None, return_metadata)

    # check that columns are unique
    if not obj.columns.is_unique:
        msg = f"{var_name} must have unique column indices, but found {obj.columns}"
        return _ret(False, msg, None, return_metadata)

    # Check instance index is unique
    if not obj.index.is_unique:
        msg = (
            f"The instance index of {var_name} must be unique, "
            f"but found duplicates. Use {var_name}.duplicated() "
            f"to find the duplicates."
        )
        return _ret(False, msg, None, return_metadata)

    metadata = dict()
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.shape[1] < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = len(obj) == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = _nested_dataframe_has_nans(obj)
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = not _nested_dataframe_has_unequal(obj)
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj.columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns.to_list()
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_panel_dtypekind(obj, "nested_univ")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_panel_dtypekind(obj, "nested_univ")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    # todo: this is temporary override, proper is_empty logic needs to be added
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = False
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = True
    # end hacks

    return _ret(True, None, metadata, return_metadata)


class PanelNumpyFlat(ScitypePanel):
    """Data type: flat 2D np.ndarray based specification of panel of time series.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "numpyflat",  # any string
        "name_python": "panel_np_flat",  # lower_snake_case
        "name_aliases": [],
        "description": (
            "WARNING: only for internal use, not a fully supported Panel mtype. "
            "2D np.array of format (n_instances, n_columns*n_timepoints)"
        ),
        "python_version": None,
        "python_dependencies": "numpy",
        "python_type": "numpy.ndarray",
        "capability:multivariate": False,
        "capability:unequally_spaced": False,
        "capability:missing_values": True,
        "internal_only": True,
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
        return _check_numpyflat_Panel(
            obj, return_metadata=return_metadata, var_name=var_name
        )


def _check_numpyflat_Panel(obj, return_metadata=False, var_name="obj"):
    """Check if obj is a numpy.ndarray according to numpyflat specification."""
    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not len(obj.shape) == 2:
        msg = f"{var_name} must be a 2D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 2D np.ndarray
    metadata = dict()
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    if _req("n_features", return_metadata):
        metadata["n_features"] = 1
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = [0]
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_panel_dtypekind(obj, "numpyflat")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_panel_dtypekind(obj, "numpyflat")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)
    # np.arrays are considered equally spaced, equal length, by assumption
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = True
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = True
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = obj.shape[0]
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = obj.shape[0] == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = np.isnan(obj).any()

    return _ret(True, None, metadata, return_metadata)


class PanelDask(ScitypePanel):
    """Data type: dask data frame based specification of panel of time series.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "dask_panel",  # any string
        "name_python": "panel_dask",  # lower_snake_case
        "name_aliases": [],
        "description": "dask DataFrame based panel of time series",
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
            scitype="Panel",
        )


class PanelPolarsEager(ScitypePanel):
    """Data type: polars.DataFrame based specification of panel of time series.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "polars_panel",  # any string
        "name_python": "panel_polars",  # lower_snake_case
        "name_aliases": [],
        "description": "polars.DataFrame based panel of time series",
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
        return _check_polars_panel(
            obj=obj, return_metadata=return_metadata, var_name=var_name, scitype="Panel"
        )


def _check_polars_panel(obj, return_metadata=False, var_name="obj", scitype="Panel"):
    """Check if obj is a polars.DataFrame according to polars specification."""
    import polars as pl

    from sktime.datatypes._adapter.polars import check_polars_frame, get_mi_cols

    metadict = check_polars_frame(
        obj=obj,
        return_metadata=return_metadata,
        var_name=var_name,
        scitype=scitype,
    )

    if isinstance(metadict, tuple) and metadict[0]:
        requires_series_grps = [
            "n_instances",
            "is_one_series",
            "is_equal_length",
            "is_equally_spaced",
        ]

        if _req(requires_series_grps, return_metadata):
            if isinstance(obj, pl.LazyFrame):
                for key in requires_series_grps:
                    metadict[1][key] = "NA"
            else:
                mi_cols = get_mi_cols(obj)
                levels = mi_cols[:-1]
                if len(mi_cols) == 1:
                    levels = levels[0]
                series_groups = obj.group_by(levels).agg(
                    [
                        pl.col(mi_cols[-1]).alias(
                            "level_values"
                        ),  # index values in each level
                        pl.len().alias("size"),  # Calculate the size of each group
                    ]
                )
                n_series = series_groups.shape[0]

                if _req("n_instances", return_metadata):
                    metadict[2]["n_instances"] = n_series
                if _req("is_one_series", return_metadata):
                    metadict[2]["is_one_series"] = n_series == 1
                if _req("is_equal_length", return_metadata):
                    metadict[2]["is_equal_length"] = _list_all_equal(
                        series_groups["size"].to_numpy()
                    )
                if _req("is_equally_spaced", return_metadata):
                    metadict[2]["is_equally_spaced"] = "NA"

        requires_panel_grps = ["n_panels", "is_one_panel"]
        if _req(requires_panel_grps, return_metadata):
            if isinstance(obj, pl.LazyFrame):
                for key in requires_panel_grps:
                    metadict[1][key] = "NA"
            else:
                if scitype == "Panel":
                    n_panels = 1
                else:
                    mi_cols = get_mi_cols(obj)
                    panel_groups = obj.group_by(mi_cols[:-2]).agg([pl.col(mi_cols[-2])])
                    n_panels = panel_groups.shape[0]

                if _req("n_panels", return_metadata):
                    metadict[2]["n_panels"] = n_panels
                if _req("is_one_panel", return_metadata):
                    metadict[2]["is_one_panel"] = n_panels == 1

    return metadict


class PanelGluontsList(ScitypePanel):
    """Data type: gluonTS representation of univariate and multivariate time series.

    Name: ``"gluonts_ListDataset_panel"``

    Short description:

    A ``list`` of ``dict``,
    with list index = instances, ``dict['target']`` rows = time points,
    ``dict['target']`` cols = variables, and ``dict['start']`` marking the interval.
    Identical to ``gluonts.dataset.common.ListDataset``.

    Long description:

    The ``"gluonts_ListDataset_panel"`` :term:`mtype` is a concrete specification
    that implements the ``Panel`` :term:`scitype`, i.e., the abstract
    type of a collection of time series.

    An object ``obj: list`` follows the specification iff:

    * structure convention: ``obj`` must be a ``list`` of ``dict``.
      Each ``dict`` must contain a key ``"target"``which
      maps to a 1D ``numpy.ndarray`` for a univariate, and
      2D ``numpy.ndarray`` for a multivariate time series.
      Optionally, it may also contain a key ``"start"``
      that maps to a ``pandas.Period`` object.
      eg: ``pandas.Period("2024-01-01", freq="D")`` for a time series starting
      on 2024-01-01 and sampled daily.
    * instances: instances correspond to different list elements of ``obj``.
    * instance index: the instance index of an instance is the list index at which
      it is located in ``obj``. That is, the data at ``obj[i]``
      correspond to observations of the instance with index ``i``.
    * time points: rows of ``obj[i]['target']`` correspond to
      different, distinct time points, at which
      instance ``i`` is observed.
    * time index: the time index is implicit and by-convention.
      The ``j``-th element (for an integer ``j``) of instance ``i`` is interpreted
      as an observation at the time point ``j``.
      If ``"start"`` key is present, the time index for
      the ``j``-th element of instance ``i`` is ``obj[i]['start'] + j``.
    * variables: columns of ``obj[i]['target']`` correspond to different variables
      available for instance ``i``.
    * variable names: ``numpy`` mtypes cannot represent variable names. If required,
      then variable names are assigned ``"value_{k}``
      where ``k`` is the feature column index.

    Capabilities:

    * can represent panels of multivariate series
    * can not represent panels of unequally spaced series
    * can represent panels of unequally supported series
    * can represent panels of series with different sets of variables
    * can represent missing values

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "gluonts_ListDataset_panel",  # any string
        "name_python": "panel_gluonts_list",  # lower_snake_case
        "name_aliases": [],
        "description": "gluonts list based panel of time series",
        "python_version": None,
        "python_dependencies": None,
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
            or len(obj[0]["target"]) <= 1
        ):
            msg = f"{var_name} must be a listDataset, found {type(obj)}"
            return _ret(False, msg, None, return_metadata)

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
            metadata["n_instances"] = len(obj)

        if _req("n_panels", return_metadata):
            metadata["n_panels"] = len(obj)

        if _req("feature_names", return_metadata):
            # Check first if the ListDataset is empty
            if len(obj) < 1:
                metadata["feature_names"] = []
            else:
                metadata["feature_names"] = [f"value_{i}" for i in range(n_features)]

        for series in obj:
            # check that no dtype is object
            if series["target"].dtype == "object":
                msg = f"{var_name} should not have column of 'object' dtype"
                return _ret(False, msg, None, return_metadata)

        # For a GluonTS ListDataset, only a start date and frequency is set
        # so everything should thus be equally spaced
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = True

        if _req("has_nans", return_metadata):
            for series in obj:
                metadata["has_nans"] = pd.isnull(series["target"]).any()

                # Break out if at least 1 time series has NaN values
                if metadata["has_nans"]:
                    break

        return _ret(True, None, metadata, return_metadata)


class PanelGluontsPandas(ScitypePanel):
    """Data type: polars.DataFrame based specification of panel of time series.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "gluonts_PandasDataset_panel",  # any string
        "name_python": "panel_gluonts_pandas",  # lower_snake_case
        "name_aliases": [],
        "description": "gluonts PandasDataset based panel of time series",
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
        # Importing required libraries
        from gluonts.dataset.pandas import PandasDataset

        metadata = dict()

        # Check for type correctness
        if not isinstance(obj, PandasDataset):
            msg = f"{var_name} must be a gluonts.PandasDataset, found {type(obj)}"
            return _ret(False, msg, None, return_metadata)

        if not hasattr(obj._data_entries.iterable, "iterable"):
            msg = (
                f"{var_name} must be formed with a multiindex DataFrame to "
                "be a valid `pandasDataset_panel`"
            )
            return _ret(False, msg, None, return_metadata)

        # Convert to a pandas DF for easier checks
        df = obj._data_entries.iterable.iterable

        # Check if there are no values
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(obj._data_entries) == 0

        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = len(df[0][1].columns) == 1

        req_n_feat = ["n_features", "feature_kind", "dtypekind_dfip"]

        if _req(req_n_feat, return_metadata):
            n_features = len(df[0][1].columns)

        if _req("n_features", return_metadata):
            metadata["n_features"] = n_features

        if _req("dtypekind_dfip", return_metadata):
            index_cols_count = len(df[0][1].columns)

            # slicing off additional index columns
            dtype_list = df[0][1].dtypes.to_list()[index_cols_count:]

            metadata["dtypekind_dfip"] = _pandas_dtype_to_kind(dtype_list)

        if _req("feature_kind", return_metadata):
            dtype_list = df[0][1].dtypes.to_list()[index_cols_count:]
            dtype_kind = _pandas_dtype_to_kind(dtype_list)
            metadata["feature_kind"] = _get_feature_kind(dtype_kind)

        if _req("n_instances", return_metadata):
            metadata["n_instances"] = len(df)

        if _req("n_panels", return_metadata):
            metadata["n_panels"] = len(df)

        if _req("feature_names", return_metadata):
            metadata["feature_names"] = df[0][1].columns

        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = True

        if _req("has_nans", return_metadata):
            metadata["has_nans"] = False

        return _ret(True, None, metadata, return_metadata)


class PanelPdWide(ScitypePanel):
    """Data type: panel of time series in pandas wide format.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "pd-wide",  # any string
        "name_python": "panel_pd_wide",  # lower_snake_case
        "name_aliases": [],
        "description": (
            "WARNING: only for internal use, not a fully supported Panel mtype. "
            "pd.DataFrame in wide format, cols = (instance*timepoints)"
        ),
        "python_version": None,
        "python_dependencies": "pandas",
        "python_type": "pandas.DataFrame",
        "capability:multivariate": False,
        "capability:unequally_spaced": False,
        "capability:missing_values": True,
        "internal_only": True,
        "skip_in_checks": True,
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
        # no checker defined, only used in conversions
        pass


class PanelPdLong(ScitypePanel):
    """Data type: panel of time series in pandas long format.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_equally_spaced : bool
        True iff series index is equally spaced
    is_equal_length: bool
        True iff all series in panel are of equal length
    is_empty: bool
        True iff table has no variables or no instances
    is_one_series: bool
        True iff there is only one series in the panel of time series
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the panel of time series
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    dtypekind_dfip: list of DtypeKind enum
        list of DtypeKind enum values for each feature in the panel,
        following the data frame interface protocol
    feature_kind: list of str
        list of feature kind strings for each feature in the panel,
        coerced to FLOAT or CATEGORICAL type
    """

    _tags = {
        "scitype": "Panel",
        "name": "pd-long",  # any string
        "name_python": "panel_pd_long",  # lower_snake_case
        "name_aliases": [],
        "description": (
            "WARNING: only for internal use, not a fully supported Panel mtype. "
            "pd.DataFrame in long format, cols = (index, time_index, column)"
        ),
        "python_version": None,
        "python_dependencies": "pandas",
        "python_type": "pandas.DataFrame",
        "capability:multivariate": True,
        "capability:unequally_spaced": True,
        "capability:missing_values": True,
        "internal_only": True,
        "skip_in_checks": True,
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
        # no checker defined, only used in conversions
        pass

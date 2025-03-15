"""Machine type checkers for Table scitype.

Exports checkers for Table scitype:

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
var_name: str, optional, default="obj" - name of input in error messages

Returns
-------
valid: bool - whether obj is a valid object of mtype/scitype
msg: str - error message if object is not valid, otherwise None
        returned only if return_metadata is True
metadata: dict - metadata about obj if valid, otherwise None
        returned only if return_metadata is True
    fields:
        "is_univariate": bool, True iff table has one variable
        "is_empty": bool, True iff table has no variables or no instances
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances/rows in the table
        "n_features": int, number of variables in table
        "feature_names": list of int or object, names of variables in table
"""

import numpy as np
import pandas as pd

from sktime.datatypes._base._common import _req, _ret
from sktime.datatypes._dtypekind import _get_feature_kind, _get_table_dtypekind
from sktime.datatypes._table._base import ScitypeTable

PRIMITIVE_TYPES = (float, int, str)


class TablePdDataFrame(ScitypeTable):
    """Data type: pandas.DataFrame based specification of tabular data.

    Name: ``"pd_DataFrame_Table"``

    Short description:

    a pandas ``DataFrame`` representing tabular data,
    with rows = instances, cols = features

    Long description:

    The ``"pd_DataFrame_Table"`` :term:`mtype` is a concrete specification
    that implements the ``Table`` :term:`scitype`, i.e., the abstract
    type of tabular data.

    An object ``obj: pandas.DataFrame`` follows the specification iff:

    * structure convention: ``obj.index`` can be any valid pandas index.
    * features: columns of ``obj`` correspond to different features
    * feature names: column names ``obj.columns``
    * instances: rows of ``obj`` correspond to different instances
    * instance index: ``obj.index`` is interpreted as the instance index

    Capabilities:

    * can represent multivariate data
    * can represent missing values

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
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
        "scitype": "Table",
        "name": "pd_DataFrame_Table",  # any string
        "name_python": "table_pd_df",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
        "python_type": "pandas.DataFrame",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": True,
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
        return _check_pddataframe_table(obj, return_metadata, var_name)


def _check_pddataframe_table(obj, return_metadata=False, var_name="obj"):
    """Check if obj is a pandas.DataFrame based specification of single time series."""
    metadata = dict()

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pandas.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(index)
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj.columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns.to_list()
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_table_dtypekind(obj, "pd.DataFrame")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_table_dtypekind(obj, "pd.DataFrame")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    return _ret(True, None, metadata, return_metadata)


class TablePdSeries(ScitypeTable):
    """Data type: pandas.Series based specification of tabular data.

    Name: ``"pd_Series_Table"``

    Short description:

    a pandas ``Series`` representing tabular data,
    with rows = instances, single feature

    Long description:

    The ``"pd_Series_Table"`` :term:`mtype` is a concrete specification
    that implements the ``Table`` :term:`scitype`, i.e., the abstract
    type of tabular data.

    An object ``obj: pandas.Series`` follows the specification iff:

    * structure convention: ``obj.index`` can be any valid pandas index.
    * feature: the series ``obj`` represents a single feature
    * feature name: the ``name`` attribute of the ``pd.Series`` object
    * instances: rows of ``obj`` correspond to different instances
    * instance index: ``obj.index`` is interpreted as the instance index

    Capabilities:

    * cannot represent multivariate data
    * can represent missing values

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
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
        "scitype": "Table",
        "name": "pd_Series_Table",  # any string
        "name_python": "table_pd_series",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
        "python_type": "pandas.Series",
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:index": True,
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
        return _check_pdseries_table(obj, return_metadata, var_name)


def _check_pdseries_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, pd.Series):
        msg = f"{var_name} must be a pandas.Series, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a pd.Series
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(index)
    if _req("n_features", return_metadata):
        metadata["n_features"] = 1
    if _req("feature_names", return_metadata):
        if not hasattr(obj, "name") or obj.name is None:
            metadata["feature_names"] = [0]
        else:
            metadata["feature_names"] = [obj.name]
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_table_dtypekind(obj, "pd.Series")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_table_dtypekind(obj, "pd.Series")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    return _ret(True, None, metadata, return_metadata)


class TableNp1D(ScitypeTable):
    """Data type: 1D np.ndarray based specification of data frame table.

    Name: ``"numpy1D"``

    Short description:

    a 1D numpy ``ndarray`` representing a univariate data table,
    with elements as instances of single feature

    Long description:

    The ``"numpy1D"`` :term:`mtype` is a concrete specification
    that implements the ``Table`` :term:`scitype`, i.e., the abstract
    type of tabular data.

    An object ``obj: np.ndarray`` follows the specification iff:

    * structure convention: ``obj`` is a 1D numpy array.
    * feature: the array ``obj`` represents a single feature
    * instances: elements of ``obj`` correspond to different instances
    * instance index: The instance index is implicit and by-convention.
      The ``i``-th entry (for an integer ``i``) is interpreted as the ``i``-th instance.
      That is, the index is always interpreted as zero-indexed integer.

    Capabilities:

    * cannot represent multivariate data
    * can represent missing values

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
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
        "scitype": "Table",
        "name": "numpy1D",  # any string
        "name_python": "table_numpy1d",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "numpy",
        "python_type": "numpy.ndarray",
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:index": False,
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
        return _check_numpy1d_table(obj, return_metadata, var_name)


def _check_numpy1d_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 1:
        msg = f"{var_name} must be 1D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 1D np.ndarray
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)
    # 1D numpy arrays are considered univariate
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    # check whether there any nans; compute only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()
    # 1D numpy arrays are considered univariate, with one feature named 0 (integer)
    if _req("n_features", return_metadata):
        metadata["n_features"] = 1
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = [0]
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_table_dtypekind(obj, "numpy1D")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_table_dtypekind(obj, "numpy1D")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    return _ret(True, None, metadata, return_metadata)


class TableNp2D(ScitypeTable):
    """Data type: 2D np.ndarray based specification of data frame table.

    Name: ``"numpy2D"``

    Short description:

    a 2D numpy array representing tabular data,
    with rows = instances, cols = features

    Long description:

    The ``"numpy2D"`` :term:`mtype` is a concrete specification
    that implements the ``Table`` :term:`scitype`, i.e., the abstract
    type of tabular data.

    An object ``obj: numpy.ndarray`` follows the specification iff:

    * structure convention: ``obj.shape`` is (n_instances, n_features)
    * features: columns of ``obj`` correspond to different features
    * feature names: feature names are integers 0, 1, ..., n_features-1
    * instances: rows of ``obj`` correspond to different instances

    Capabilities:

    * can represent multivariate data
    * can represent missing values

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
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
        "scitype": "Table",
        "name": "numpy2D",  # any string
        "name_python": "table_numpy2d",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "numpy",
        "python_type": "numpy.ndarray",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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
        return _check_numpy2d_table(obj, return_metadata, var_name)


def _check_numpy2d_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 2:
        msg = f"{var_name} must be 1D or 2D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 2D np.ndarray
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.shape[1] < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = obj.shape[0]
    # check whether there any nans; compute only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()
    # 1D numpy arrays are considered univariate, with integer feature names
    if _req("n_features", return_metadata):
        metadata["n_features"] = obj.shape[1]
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = list(range(obj.shape[1]))
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_table_dtypekind(obj, "numpy2D")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_table_dtypekind(obj, "numpy2D")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    return _ret(True, None, metadata, return_metadata)


class TableListOfDict(ScitypeTable):
    """Data type: list of dict based specification of data frame table.

    Name: ``"list_of_dict"``

    Short description:

    a list of dict representing tabular data,
    with rows = instances, keys = features

    Long description:

    The ``"list_of_dict"`` :term:`mtype` is a concrete specification
    that implements the ``Table`` :term:`scitype`, i.e., the abstract
    type of tabular data.

    An object ``obj: list`` follows the specification iff:

    * structure convention: ``obj`` is a list of dict
    * features: keys of dict correspond to different features
    * feature names: keys of dict
    * instances: elements of ``obj`` correspond to different instances

    Capabilities:

    * can represent multivariate data
    * can represent missing values

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
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
        "scitype": "Table",
        "name": "list_of_dict",  # any string
        "name_python": "table_list_of_dict",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "numpy",
        "python_type": "list",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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
        return _check_list_of_dict_table(obj, return_metadata, var_name)


def _check_list_of_dict_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, list):
        msg = f"{var_name} must be a list of dict, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not np.all([isinstance(x, dict) for x in obj]):
        msg = (
            f"{var_name} must be a list of dict, but elements at following "
            f"indices are not dict: {np.where(not isinstance(x, dict) for x in obj)}"
        )
        return _ret(False, msg, None, return_metadata)

    for i, d in enumerate(obj):
        for key in d.keys():
            if not isinstance(d[key], PRIMITIVE_TYPES):
                msg = (
                    "all entries must be of primitive type (str, int, float), but "
                    f"found {type(d[key])} at index {i}, key {key}"
                )

    # we now know obj is a list of dict
    # check whether there any nans; compute only if requested
    if _req("is_univariate", return_metadata):
        multivariate_because_one_row = np.any([len(x) > 1 for x in obj])
        if not multivariate_because_one_row:
            all_keys = np.unique([key for d in obj for key in d.keys()])
            multivariate_because_keys_different = len(all_keys) > 1
            multivariate = multivariate_because_keys_different
        else:
            multivariate = multivariate_because_one_row
        metadata["is_univariate"] = not multivariate
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = np.any(
            [pd.isnull(d[key]) for d in obj for key in d.keys()]
        )
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or np.all([len(x) < 1 for x in obj])
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)

    # this can be expensive, so compute only if requested
    if _req("n_features", return_metadata) or _req("feature_names", return_metadata):
        all_keys = np.unique([key for d in obj for key in d.keys()])
        if _req("n_features", return_metadata):
            metadata["n_features"] = len(all_keys)
        if _req("feature_names", return_metadata):
            metadata["feature_names"] = all_keys.tolist()
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_table_dtypekind(obj, "list_of_dict")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_table_dtypekind(obj, "list_of_dict")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    return _ret(True, None, metadata, return_metadata)


class TablePolarsEager(ScitypeTable):
    """Data type: eager polars DataFrame based specification of data frame table.

    Name: ``"TablePolarsEager"``

    Short description:
        A specification for a data table backed by an eager Polars DataFrame,
        supporting both univariate and multivariate data.

    Long description:
        The ``"TablePolarsEager"`` :term:`mtype` is a concrete specification
        that implements the ``Table`` :term:`scitype`, representing a data table
        with an eager Polars DataFrame.
        An object ``obj: Polars DataFrame`` follows the specification iff:

        * structure convention: ``obj`` is a Polars DataFrame.
        * feature: the DataFrame can have multiple features (columns).
        * instances: rows of the DataFrame represent individual instances.
        * instance index: The index is implicit, with each row corresponding to
          a unique instance (zero-indexed by default).

    Capabilities:
        * supports multivariate data with multiple features.
        * can handle missing values (NaNs) and empty tables.
        * includes metadata like feature names, data types, and feature kinds.

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
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
        "scitype": "Table",
        "name": "polars_eager_table",  # any string
        "name_python": "table_polars_eager",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
        "python_type": "polars.DataFrame",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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

        return check_polars_frame(obj, return_metadata, var_name, lazy=False)


class TablePolarsLazy(ScitypeTable):
    """Data type: lazy polars DataFrame based specification of data frame table.

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
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
        "scitype": "Table",
        "name": "polars_lazy_table",  # any string
        "name_python": "table_polars_lazy",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
        "python_type": "polars.LazyFrame",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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

        return check_polars_frame(obj, return_metadata, var_name, lazy=True)

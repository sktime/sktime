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
        "is_univariate": bool, True iff all series in hier.panel have one variable
        "is_equally_spaced": bool, True iff all series indices are equally spaced
        "is_equal_length": bool, True iff all series in panel are of equal length
        "is_empty": bool, True iff one or more of the series in the panel are empty
        "is_one_series": bool, True iff there is only one series in the hier.panel
        "is_one_panel": bool, True iff there is only one flat panel in the hier.panel
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances in the hierarchical panel
        "n_panels": int, number of flat panels in the hierarchical panel
        "n_features": int, number of variables in series
        "feature_names": list of int or object, names of variables in series
"""

import numpy as np

from sktime.datatypes._hierarchical._base import ScitypeHierarchical


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


class HierarchicalPdMultiIndex(ScitypeHierarchical):
    """Data type: pandas.DataFrame based specification of hierarchical series.

    Name: ``"pd_multiindex_hier"``

    Short description:
    a ``pandas.DataFrame``, with row multi-index,
    last level interpreted as time, others as hierarchy, cols = variables

    Long description:

    The ``"pd_multiindex_hier"`` :term:`mtype` is a concrete specification
    that implements the ``Hierarchical`` :term:`scitype`, i.e., the abstract
    type of a hierarchically indexed collection of time series.

    An object ``obj: pandas.DataFrame`` follows the specification iff:

    * structure convention: ``obj.index`` must be a 3 or more level multi-index of type
      ``(Index, ..., Index, t)``, where ``t`` is one of ``Int64Index``, ``RangeIndex``,
      ``DatetimeIndex``, ``PeriodIndex`` and monotonic.
      We call the last index the "time-like" index.
    * hierarchy level: rows with the same non-time-like index values correspond to the
      same hierarchy unit; rows with different non-time-like index combination
      correspond to different hierarchy unit.
    * hierarchy: the non-time-like indices in ``obj.index`` are interpreted as a
      hierarchy identifying index.
    * time index: the last element of tuples in ``obj.index`` is interpreted
      as a time index.
    * time points: rows of ``obj`` with the same ``"timepoints"`` index correspond
      to the same time point; rows of ``obj`` with different ``"timepoints"`` index
      correspond to the different time points.
    * variables: columns of ``obj`` correspond to different variables
    * variable names: column names ``obj.columns``

    Capabilities:

    * can represent multivariate hierarchical series
    * can represent unequally spaced hierarchical series
    * can represent unequally supported hierarchical series
    * cannot represent hierarchical series with different sets of variables
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
        True iff there is only one series in the hierarchical collection.
    is_one_panel: bool
        True iff there is only one flat panel in the hierarchical collection,
        i.e., the collection has a flat hierarchy, plus additional hierarchy levels
        that have only a single value.
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the hierarchical collection
    n_panels: int
        number of flat panels in the hierarchical collection
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
        "scitype": "Hierarchical",
        "name": "pd_multiindex_hier",  # any string
        "name_python": "hier_pd_df",  # lower_snake_case
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
        from sktime.datatypes._panel._check import _check_pdmultiindex_panel

        ret = _check_pdmultiindex_panel(
            obj, return_metadata=return_metadata, var_name=var_name, panel=False
        )

        return ret


class HierarchicalDask(ScitypeHierarchical):
    """Data type: dask frame based specification of hierarchical series.

    Name: ``"dask_hierarchical"``

    Short description:
    A ``dask.DataFrame`` where hierarchy is represented using explicit columns
    instead of a traditional index, and time is recorded in a dedicated column.

    Long description:
    The ``"dask_hierarchical"`` :term:`mtype` is a concrete specification of the
    ``Hierarchical`` :term:`scitype`, which represents a hierarchically structured
    collection of time series.

    An object ``obj: dask.DataFrame`` follows the specification iff:

    * structure convention: ``obj`` must have at least three index columns, where:

      - The first ``n-1`` index columns define the hierarchy.
      - The last index column represents time.
      - All index columns must be explicitly named following the pattern ``__index__*``,
        such as ``__index__0``, ``__index__1``, ..., ``__index__N-1``.

    * hierarchy level: rows with the same values in the hierarchy columns belong
      to the same hierarchy unit, while different hierarchy values correspond to
      different hierarchy units.
    * hierarchy: the hierarchy structure is explicitly encoded in columns rather
      than an index.
    * time index: the last column in the index set is interpreted as a time column.
      It must be one of ``Int64``, ``RangeIndex``, ``DatetimeIndex`` or ``PeriodIndex``,
      and must be monotonically increasing.
    * time points: rows with the same value in the time column correspond to
      the same time point.
    * variables: columns excluding the hierarchy and time columns correspond
      to variables.
    * variable names: column names are taken from ``obj.columns``.

    Capabilities:
    * can represent multivariate hierarchical series.
    * can represent unequally spaced hierarchical series.
    * can represent unequally supported hierarchical series.
    * cannot represent hierarchical series with different sets of variables.
    * can represent missing values.

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
        True iff there is only one series in the hierarchical collection.
    is_one_panel: bool
        True iff there is only one flat panel in the hierarchical collection,
        i.e., the collection has a flat hierarchy, plus additional hierarchy levels
        that have only a single value.
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the hierarchical collection
    n_panels: int
        number of flat panels in the hierarchical collection
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
        "scitype": "Hierarchical",
        "name": "dask_hierarchical",  # any string
        "name_python": "hier_dask",  # lower_snake_case
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
            scitype="Hierarchical",
        )


class HierarchicalPolarsEager(ScitypeHierarchical):
    """Data type: polars DataFrame frame based specification of hierarchical series.

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
        True iff there is only one series in the hierarchical collection.
    is_one_panel: bool
        True iff there is only one flat panel in the hierarchical collection,
        i.e., the collection has a flat hierarchy, plus additional hierarchy levels
        that have only a single value.
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances in the hierarchical collection
    n_panels: int
        number of flat panels in the hierarchical collection
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
        "scitype": "Hierarchical",
        "name": "polars_hierarchical",  # any string
        "name_python": "hier_polars_eager",  # lower_snake_case
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
        from sktime.datatypes._panel._check import _check_polars_panel

        return _check_polars_panel(
            obj=obj,
            return_metadata=return_metadata,
            var_name=var_name,
            scitype="Hierarchical",
        )

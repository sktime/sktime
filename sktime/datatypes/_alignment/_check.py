"""Machine type checkers for Alignment scitype.

Exports checkers for Alignment scitype:

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
        is_multiple : boolean, whether align_df has multiple (3 or more) time series
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.datatypes._alignment._base import ScitypeAlignment
from sktime.datatypes._base._common import _req, _ret


def check_align(align_df, name="align_df", index="iloc", return_metadata=False):
    """Check whether the object is a data frame in alignment format.

    Parameters
    ----------
    align_df : any object
        check passes if it follows alignment format, as follows:

        pandas.DataFrame with column names 'ind'+str(i) for integers i, as follows:
        all integers i between 0 and some natural number n must be present

    name : string, optional, default="align_df"
        variable name that is printed in ValueError-s

    index : string, optional, one of "iloc" (default), "loc", "either"
        whether alignment to check is "loc" or "iloc"

    return_metadata - bool, str, or list of str, optional, default=False

        * if False, returns only ``valid`` return. No metadata is returned.
        * if True, returns all three return objects. All metadata fields are returned.
        * if str, list of str, metadata return dict is subset to keys in
        ``return_metadata``. This allows selective return of metadata fields,
        to avoid unnecessary computation.

    Returns
    -------
    valid : boolean, whether align_df is a valid alignment data frame
    msg : error message if align_df is invalid
    metadata : dict
        is_multiple : boolean, whether align_df has multiple (3 or more) time series
    """
    if not isinstance(align_df, pd.DataFrame):
        msg = f"{name} is not a pandas DataFrame"
        return False, msg, {}

    cols = align_df.columns
    n = len(cols)

    correctcols = {f"ind{i}" for i in range(n)}

    if not set(cols) == set(correctcols):
        msg = f"{name} index columns must be named 'ind0', 'ind1', ... 'ind{n}'"
        return False, msg, {}

    if index == "iloc":
        # checks whether df columns are of integer (numpy or pandas nullable) type
        dtypearr = np.array([str(x) for x in align_df[cols].dtypes])
        allowedtypes = np.array(
            [
                "int",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "UInt8",
                "UInt16",
                "UInt32",
                "UInt64",
            ]
        )
        if not np.all(np.isin(dtypearr, allowedtypes)):
            msg = f"columns of {name} must have dtype intX, uintX, IntX, or UIntX"
            return False, msg, {}
    # no additional restrictions apply if loc or either, so no elif

    metadata = {}
    if _req("is_multiple", return_metadata):
        metadata["is_multiple"] = n >= 3

    return True, "", metadata


class AlignmentIloc(ScitypeAlignment):
    """Data type: pandas.DataFrame based specification of an alignment, iloc references.

    Name: ``"alignment"``

    Short description:

    ``pandas.DataFrame``, with cols = aligned series, entries = iloc references,
    rows = ordered alignment pairs/tuples

    Long description:

    The ``"alignment"`` :term:`mtype` is a concrete specification
    that implements the ``Alignment`` :term:`scitype`, i.e., the abstract
    type of an alignment of two or more time series.

    An object ``obj: pandas.DataFrame`` follows the specification iff:

    * structure convention: ``obj.index`` must be ``RangeIndex``
    * column names must be strings of the form ``"ind0"``, ``"ind1"``, ..., ``"indn"``,
        where ``n`` is the number of columns
    * column types must be integers, either numpy or pandas nullable integer types
    * all columns must be monotonic increasing, not necessarily strictly, or by 1

    Parameters
    ----------
    is_multiple : boolean
        True iff is multiple alignment, i.e., between 3 or more time series
    """

    _tags = {
        "scitype": "Alignment",
        "name": "alignment",  # any string
        "name_python": "alignment_iloc",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
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
        valid, msg, metadata = check_align(
            obj, name=var_name, index="iloc", return_metadata=return_metadata
        )

        return _ret(valid, msg, metadata, return_metadata)


class AlignmentLoc(ScitypeAlignment):
    """Data type: pandas.DataFrame based specification of an alignment, loc references.

    Name: ``"alignment_loc"``

    Short description:

    ``pandas.DataFrame``, with cols = aligned series, entries = loc references,
    rows = ordered alignment pairs/tuples

    Long description:

    The ``"alignment"`` :term:`mtype` is a concrete specification
    that implements the ``Alignment`` :term:`scitype`, i.e., the abstract
    type of an alignment of two or more time series.

    An object ``obj: pandas.DataFrame`` follows the specification iff:

    * structure convention: ``obj.index`` must be ``RangeIndex``
    * column names must be strings of the form ``"ind0"``, ``"ind1"``, ..., ``"indn"``,
        where ``n`` is the number of columns
    * column types must be valid vor loc indexing into
    ``Int64Index``, ``RangeIndex``, ``DatetimeIndex``, or ``PeriodIndex``.
    * all columns must be monotonic increasing, not necessarily strictly, or regulraly

    Parameters
    ----------
    is_multiple : boolean
        True iff is multiple alignment, i.e., between 3 or more time series
    """

    _tags = {
        "scitype": "Alignment",
        "name": "alignment_loc",  # any string
        "name_python": "alignment_loc",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
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
        valid, msg, metadata = check_align(
            obj, name=var_name, index="loc", return_metadata=return_metadata
        )

        return _ret(valid, msg, metadata, return_metadata)

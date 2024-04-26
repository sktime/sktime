"""Testing utility to compare equality in value for nested objects.

Objects compared can have one of the following valid types:
    types compatible with != comparison
    pd.Series, pd.DataFrame, np.ndarray
    lists, tuples, or dicts of a valid type (recursive)
"""

__author__ = ["fkiraly"]

__all__ = ["deep_equals"]

from inspect import isclass

import numpy as np
import pandas as pd


# todo 0.29.0: check whether scikit-base>=0.6.1 lower bound is 0.6.1 or higher
# if yes, remove this legacy function and use the new one from sktime.utils.deep_equals
def deep_equals(x, y, return_msg=False):
    """Test two objects for equality in value.

    Correct if x/y are one of the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)
        delayed types that result in the above when calling .compute(), e.g., dask df

    Important note:
        this function will return "not equal" if types of x,y are different
        for instant, bool and numpy.bool are *not* considered equal

    Parameters
    ----------
    x : object
    y : object
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
            concatenation of the following strings:
            .type - type is not equal
            .len - length is not equal
            .value - value is not equal
            .keys - if dict, keys of dict are not equal
                    if class/object, names of attributes and methods are not equal
            .dtype - dtype of pandas or numpy object is not equal
            .index - index of pandas object is not equal
            .series_equals, .df_equals, .index_equals - .equals of pd returns False
            [i] - if tuple/list: i-th element not equal
            [key] - if dict: value at key is not equal
            [colname] - if pandas.DataFrame: column with name colname is not equal
            != - call to generic != returns False
    """
    from sktime.utils.validation._dependencies import _check_soft_dependencies
    from sktime.utils.warnings import warn

    deprec_explain = (
        "The legacy deep_equals from sktime.utils._testing.deep_equals is "
        "deprecated and should be replaced by the new deep_equals,"
        " from sktime.utils.deep_equals, which requires scikit-base>=0.6.1. "
    )

    removal_schedule = (
        "The legacy deep_equals is not scheduled for removal yet, this "
        "warning will change to specify a removal date when it is scheduled."
    )

    if _check_soft_dependencies(
        "scikit-base>=0.6.1",
        package_import_alias={"scikit-base": "skbase"},
    ):
        env_msg = (
            "As you have scikit-base>=0.6.1, please update your imports to use the "
            "new deep_equals utility. "
        )
    else:
        env_msg = (
            "As you have scikit-base<0.6.1, please consider updating your environment "
            "to scikit-base>=0.6.1, and update your imports to use the "
            "new deep_equals utility. "
        )

    msg = deprec_explain + env_msg + removal_schedule
    warn(msg, FutureWarning, stacklevel=2)

    return deep_equals_legacy(x, y, return_msg=return_msg)


def deep_equals_legacy(x, y, return_msg=False):
    """Test two objects for equality in value.

    Correct if x/y are one of the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)
        delayed types that result in the above when calling .compute(), e.g., dask df

    Important note:
        this function will return "not equal" if types of x,y are different
        for instant, bool and numpy.bool are *not* considered equal

    Parameters
    ----------
    x : object
    y : object
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
            concatenation of the following strings:
            .type - type is not equal
            .len - length is not equal
            .value - value is not equal
            .keys - if dict, keys of dict are not equal
                    if class/object, names of attributes and methods are not equal
            .dtype - dtype of pandas or numpy object is not equal
            .index - index of pandas object is not equal
            .series_equals, .df_equals, .index_equals - .equals of pd returns False
            [i] - if tuple/list: i-th element not equal
            [key] - if dict: value at key is not equal
            [colname] - if pandas.DataFrame: column with name colname is not equal
            != - call to generic != returns False
    """

    def ret(is_equal, msg="", string_arguments: list = None):
        string_arguments = [] if string_arguments is None else string_arguments
        if return_msg:
            if is_equal:
                msg = ""
            elif len(string_arguments) > 0:
                msg = msg.format(*string_arguments)
            return is_equal, msg
        else:
            return is_equal

    if type(x) is not type(y):
        return ret(False, f".type, x.type = {type(x)} != y.type = {type(y)}")

    # compute delayed objects (dask)
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(y, "compute"):
        y = y.compute()

    # we now know all types are the same
    # so now we compare values
    if isinstance(x, pd.Series):
        if x.dtype != y.dtype:
            return ret(False, f".dtype, x.dtype= {x.dtype} != y.dtype = {y.dtype}")
        # if columns are object, recurse over entries and index
        if x.dtype == "object":
            index_equal = x.index.equals(y.index)
            values_equal, values_msg = deep_equals(
                list(x.values), list(y.values), return_msg=True
            )
            if not values_equal:
                msg = ".values" + values_msg
            elif not index_equal:
                msg = f".index, x.index: {x.index}, y.index: {y.index}"
            else:
                msg = ""
            return ret(index_equal and values_equal, msg)
        else:
            return ret(x.equals(y), ".series_equals, x = {} != y = {}", [x, y])
    elif isinstance(x, pd.DataFrame):
        # check column names for equality
        if not x.columns.equals(y.columns):
            return ret(
                False, f".columns, x.columns = {x.columns} != y.columns = {y.columns}"
            )
        # check dtypes for equality
        if not x.dtypes.equals(y.dtypes):
            return ret(
                False, f".dtypes, x.dtypes = {x.dtypes} != y.dtypes = {y.dtypes}"
            )
        # check index for equality
        # we are not recursing due to ambiguity in integer index types
        # which may differ from pandas version to pandas version
        # and would upset the type check, e.g., RangeIndex(2) vs Index([0, 1])
        xix = x.index
        yix = y.index
        if hasattr(xix, "dtype") and hasattr(xix, "dtype"):
            if not xix.dtype == yix.dtype:
                return ret(
                    False,
                    ".index.dtype, x.index.dtype = {} != y.index.dtype = {}",
                    [xix.dtype, yix.dtype],
                )
        if hasattr(xix, "dtypes") and hasattr(yix, "dtypes"):
            if not x.dtypes.equals(y.dtypes):
                return ret(
                    False,
                    ".index.dtypes, x.dtypes = {} != y.index.dtypes = {}",
                    [xix.dtypes, yix.dtypes],
                )
        ix_eq = xix.equals(yix)
        if not ix_eq:
            if not len(xix) == len(yix):
                return ret(
                    False,
                    ".index.len, x.index.len = {} != y.index.len = {}",
                    [len(xix), len(yix)],
                )
            if hasattr(xix, "name") and hasattr(yix, "name"):
                if not xix.name == yix.name:
                    return ret(
                        False,
                        ".index.name, x.index.name = {} != y.index.name = {}",
                        [xix.name, yix.name],
                    )
            if hasattr(xix, "names") and hasattr(yix, "names"):
                if not len(xix.names) == len(yix.names):
                    return ret(
                        False,
                        ".index.names, x.index.names = {} != y.index.name = {}",
                        [xix.names, yix.names],
                    )
                if not np.all(xix.names == yix.names):
                    return ret(
                        False,
                        ".index.names, x.index.names = {} != y.index.name = {}",
                        [xix.names, yix.names],
                    )
            elts_eq = np.all(xix == yix)
            return ret(elts_eq, ".index.equals, x = {} != y = {}", [xix, yix])
        # if columns, dtypes are equal and at least one is object, recurse over Series
        if sum(x.dtypes == "object") > 0:
            for c in x.columns:
                is_equal, msg = deep_equals(x[c], y[c], return_msg=True)
                if not is_equal:
                    return ret(False, f'["{c}"]' + msg)
            return ret(True)
        else:
            return ret(x.equals(y), ".df_equals, x = {} != y = {}", [x, y])
    elif isinstance(x, pd.Index):
        if hasattr(x, "dtype") and hasattr(y, "dtype"):
            if not x.dtype == y.dtype:
                return ret(False, f".dtype, x.dtype = {x.dtype} != y.dtype = {y.dtype}")
        if hasattr(x, "dtypes") and hasattr(y, "dtypes"):
            if not x.dtypes.equals(y.dtypes):
                return ret(
                    False, f".dtypes, x.dtypes = {x.dtypes} != y.dtypes = {y.dtypes}"
                )
        return ret(x.equals(y), ".index_equals, x = {} != y = {}", [x, y])
    elif isinstance(x, np.ndarray):
        if x.dtype != y.dtype:
            return ret(False, f".dtype, x.dtype = {x.dtype} != y.dtype = {y.dtype}")
        if x.dtype in ["object", "str"]:
            return ret(np.array_equal(x, y), ".values")
        else:
            return ret(np.array_equal(x, y, equal_nan=True), ".values")
    # recursion through lists, tuples and dicts
    elif isinstance(x, (list, tuple)):
        return ret(*_tuple_equals(x, y, return_msg=True))
    elif isinstance(x, dict):
        return ret(*_dict_equals(x, y, return_msg=True))
    elif _is_np_nan(x):
        return ret(_is_np_nan(y), f"type(x)={type(x)} != type(y)={type(y)}")
    elif isclass(x):
        return ret(x == y, f".class, x={x.__name__} != y={y.__name__}")
    elif type(x).__name__ == "ForecastingHorizon":
        return ret(*_fh_equals(x, y, return_msg=True))
    elif isinstance(x != y, bool) and x != y:
        return ret(False, f" !=, {x} != {y}")
    # csr-matrix must not be compared using np.any(x!=y)
    elif type(x).__name__ == "csr_matrix":  # isinstance(x, csr_matrix):
        if not np.allclose(x.A, y.A):
            return ret(False, f" !=, {x} != {y}")
    elif np.any(x != y):
        return ret(False, f" !=, {x} != {y}")
    return ret(True, "")


def _is_np_nan(x):
    return isinstance(x, float) and np.isnan(x)


def _tuple_equals(x, y, return_msg=False):
    """Test two tuples or lists for equality.

    Correct if tuples/lists contain the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Parameters
    ----------
    x: tuple or list
    y: tuple or list
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
            concatenation of the following elements:
            .len - length is not equal
            [i] - i-th element not equal
    """

    def ret(is_equal, msg):
        if return_msg:
            if is_equal:
                msg = ""
            return is_equal, msg
        else:
            return is_equal

    n = len(x)

    if n != len(y):
        return ret(False, f".len, x.len = {n} != y.len = {len(y)}")

    # we now know dicts are same length
    for i in range(n):
        xi = x[i]
        yi = y[i]

        # recurse through xi/yi
        is_equal, msg = deep_equals(xi, yi, return_msg=True)
        if not is_equal:
            return ret(False, f"[{i}]" + msg)

    return ret(True, "")


def _dict_equals(x, y, return_msg=False):
    """Test two dicts for equality.

    Correct if dicts contain the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Parameters
    ----------
    x: dict
    y: dict
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
            concatenation of the following strings:
            .keys - keys are not equal
            [key] - values at key is not equal
    """

    def ret(is_equal, msg):
        if return_msg:
            if is_equal:
                msg = ""
            return is_equal, msg
        else:
            return is_equal

    xkeys = set(x.keys())
    ykeys = set(y.keys())

    if xkeys != ykeys:
        xmy = xkeys.difference(ykeys)
        ymx = ykeys.difference(xkeys)
        diffmsg = ".keys,"
        if len(xmy) > 0:
            diffmsg += f" x.keys-y.keys = {xmy}."
        if len(ymx) > 0:
            diffmsg += f" y.keys-x.keys = {ymx}."
        return ret(False, diffmsg)

    # we now know that xkeys == ykeys
    for key in xkeys:
        xi = x[key]
        yi = y[key]

        # recurse through xi/yi
        is_equal, msg = deep_equals(xi, yi, return_msg=True)
        if not is_equal:
            return ret(False, f"[{key}]" + msg)

    return ret(True, "")


def _fh_equals(x, y, return_msg=False):
    """Test two forecasting horizons for equality.

    Correct if both x and y are ForecastingHorizon

    Parameters
    ----------
    x: ForecastingHorizon
    y: ForecastingHorizon
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
            concatenation of the following strings:
            .is_relative - x is absolute and y is relative, or vice versa
            .values - values of x and y are not equal
    """

    def ret(is_equal, msg):
        if return_msg:
            if is_equal:
                msg = ""
            return is_equal, msg
        else:
            return is_equal

    if x.is_relative != y.is_relative:
        return ret(False, ".is_relative")

    # recurse through values of x, y
    is_equal, msg = deep_equals(x._values, y._values, return_msg=True)
    if not is_equal:
        return ret(False, ".values" + msg)

    return ret(True, "")

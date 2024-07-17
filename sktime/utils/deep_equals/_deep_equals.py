"""Testing utility to compare equality in value for nested objects.

Objects compared can have one of the following valid types:
    types compatible with != comparison
    pd.Series, pd.DataFrame, np.ndarray
    lists, tuples, or dicts of a valid type (recursive)
    polars.DataFrame, polars.LazyFrame
"""

from skbase.utils.deep_equals._common import _make_ret
from skbase.utils.deep_equals._deep_equals import deep_equals as _deep_equals

__author__ = ["fkiraly"]
__all__ = ["deep_equals"]


def deep_equals(x, y, return_msg=False, plugins=None):
    """Test two objects for equality in value.

    Correct if x/y are one of the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Important note:
        this function will return "not equal" if types of x,y are different
        for instant, bool and numpy.bool are *not* considered equal

    Parameters
    ----------
    x : object
    y : object
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal
    plugins : list, optional, default=None
        optional additional deep_equals plugins to use
        will be appended to the default plugins from ``skbase`` ``deep_equals_custom``
        see ``skbase`` ``deep_equals_custom`` for details of signature of plugins

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
    # call deep_equals_custom with default plugins
    plugins_default = [
        _csr_matrix_equals_plugin,
        _dask_dataframe_equals_plugin,
        _fh_equals_plugin,
        _polars_equals_plugin,
    ]

    if plugins is not None:
        plugins_inner = plugins_default + plugins
    else:
        plugins_inner = plugins_default

    res = _deep_equals(x, y, return_msg=return_msg, plugins=plugins_inner)
    return res


def _fh_equals_plugin(x, y, return_msg=False, deep_equals=None):
    """Test two forecasting horizons for equality.

    Correct if both x and y are ForecastingHorizon.

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
    if type(x).__name__ != "ForecastingHorizon":
        return None

    ret = _make_ret(return_msg)

    if x.is_relative != y.is_relative:
        return ret(False, ".is_relative")

    # recurse through values of x, y
    is_equal, msg = deep_equals(x._values, y._values, return_msg=True)
    if not is_equal:
        return ret(False, ".values" + msg)

    return ret(True, "")


def _csr_matrix_equals_plugin(x, y, return_msg=False, deep_equals=None):
    """Test two scipy csr_matrix for equality.

    Correct if both x and y are csr_matrix.

    Parameters
    ----------
    x: csr_matrix
    y: csr_matrix
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
        if unequal, returns string
    returns None if this function does not apply, i.e., x is not dask
    """
    if type(x).__name__ != "csr_matrix":  # isinstance(x, csr_matrix):
        return None

    ret = _make_ret(return_msg)

    if x.shape != y.shape:
        return ret(False, " !=, {} != {}", [x, y])

    if (x != y).nnz == 0:
        return ret(True, "")

    return ret(False, " !=, {} != {}", [x, y])


def _dask_dataframe_equals_plugin(x, y, return_msg=False, deep_equals=None):
    """Test two dask dataframes for equality.

    Correct if both x and y are dask.dataframe.

    Parameters
    ----------
    x: dask.dataframe
    y: dask.dataframe
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
        if unequal, returns string
    returns None if this function does not apply, i.e., x is not dask.dataframe
    """
    if not hasattr(x, "compute"):
        return None

    from sktime.utils.dependencies import _check_soft_dependencies

    dask_available = _check_soft_dependencies("dask", severity="none")

    if not dask_available:
        return None

    import dask

    if not isinstance(x, dask.dataframe.DataFrame):
        return None

    x = x.compute()
    y = y.compute()

    return deep_equals(x, y, return_msg=return_msg)


def _polars_equals_plugin(x, y, return_msg=False):
    """Test two polars dataframes for equality.

    Correct if both x and y are polars.DataFrame or polars.LazyFrame.

    Parameters
    ----------
    x: polars.DataFrame or polars.LazyFrame
    y: polars.DataFrame or polars.LazyFrame
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
        if unequal, returns string
    returns None if this function does not apply, i.e., x is not polars
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    polars_available = _check_soft_dependencies("polars", severity="none")

    if not polars_available:
        return None

    import polars as pl

    if not isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        return None

    ret = _make_ret(return_msg)

    # compare pl.DataFrame
    if isinstance(x, pl.DataFrame):
        return ret(x.equals(y), ".polars_equals")

    # compare pl.LazyFrame
    if isinstance(x, pl.LazyFrame):
        return ret(x.collect().equals(y.collect()), ".polars_equals")

    return None

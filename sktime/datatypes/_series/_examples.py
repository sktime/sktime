# -*- coding: utf-8 -*-
"""Example generation for testing.

Exports dict of examples, useful for testing as fixtures.

example_dict: dict indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are data objects, considered examples for the mtype
    all examples with same index are considered "same" on scitype content
    if None, indicates that representation is not possible

example_lossy: dict of bool indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are bool, indicate whether representation has information removed
    all examples with same index are considered "same" on scitype content

example_metadata: dict of metadata dict, indexed by pair
  1st element = considered as this scitype - str
  2nd element = int - index of example
  (there is no "mtype" element, as properties are equal for all mtypes)
elements are metadata dict, as returned by check_is_mtype
    used as expected return of check_is_mtype in tests

overall, conversions from non-lossy representations to any other ones
    should yield the element exactly, identidally (given same index)
"""

import numpy as np
import pandas as pd

from sktime.utils.validation._dependencies import _check_soft_dependencies

example_dict = dict()
example_dict_lossy = dict()
example_dict_metadata = dict()

###
# example 0: univariate

s = pd.Series([1, 4, 0.5, -3], dtype=np.float64, name="a")

example_dict[("pd.Series", "Series", 0)] = s
example_dict_lossy[("pd.Series", "Series", 0)] = False

df = pd.DataFrame({"a": [1, 4, 0.5, -3]})

example_dict[("pd.DataFrame", "Series", 0)] = df
example_dict_lossy[("pd.DataFrame", "Series", 0)] = False

arr = np.array([[1], [4], [0.5], [-3]])

example_dict[("np.ndarray", "Series", 0)] = arr
example_dict_lossy[("np.ndarray", "Series", 0)] = True

if _check_soft_dependencies("xarray", severity="none"):
    import xarray as xr

    da = xr.DataArray(
        [[1], [4], [0.5], [-3]],
        coords=[[0, 1, 2, 3], ["a"]],
    )

    example_dict[("xr.DataArray", "Series", 0)] = da
    example_dict_lossy[("xr.DataArray", "Series", 0)] = False


example_dict_metadata[("Series", 0)] = {
    "is_univariate": True,
    "is_equally_spaced": True,
    "is_empty": False,
    "has_nans": False,
}

###
# example 1: multivariate

example_dict[("pd.Series", "Series", 1)] = None
example_dict_lossy[("pd.Series", "Series", 1)] = None

df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})

example_dict[("pd.DataFrame", "Series", 1)] = df
example_dict_lossy[("pd.DataFrame", "Series", 1)] = False

arr = np.array([[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]])

example_dict[("np.ndarray", "Series", 1)] = arr
example_dict_lossy[("np.ndarray", "Series", 1)] = True
if _check_soft_dependencies("xarray", severity="none"):
    import xarray as xr

    da = xr.DataArray(
        [[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]],
        coords=[[0, 1, 2, 3], ["a", "b"]],
    )

    example_dict[("xr.DataArray", "Series", 1)] = da
    example_dict_lossy[("xr.DataArray", "Series", 1)] = False

example_dict_metadata[("Series", 1)] = {
    "is_univariate": False,
    "is_equally_spaced": True,
    "is_empty": False,
    "has_nans": False,
}

###
# example 2: multivariate, positive

example_dict[("pd.Series", "Series", 2)] = None
example_dict_lossy[("pd.Series", "Series", 2)] = None

df = pd.DataFrame({"a": [1, 4, 0.5, 3], "b": [3, 7, 2, 3 / 7]})

example_dict[("pd.DataFrame", "Series", 2)] = df
example_dict_lossy[("pd.DataFrame", "Series", 2)] = False

arr = np.array([[1, 3], [4, 7], [0.5, 2], [3, 3 / 7]])

example_dict[("np.ndarray", "Series", 2)] = arr
example_dict_lossy[("np.ndarray", "Series", 2)] = True

if _check_soft_dependencies("xarray", severity="none"):
    import xarray as xr

    da = xr.DataArray(
        [[1, 3], [4, 7], [0.5, 2], [3, 3 / 7]],
        coords=[[0, 1, 2, 3], ["a", "b"]],
    )

    example_dict[("xr.DataArray", "Series", 2)] = da
    example_dict_lossy[("xr.DataArray", "Series", 2)] = False


example_dict_metadata[("Series", 2)] = {
    "is_univariate": False,
    "is_equally_spaced": True,
    "is_empty": False,
    "has_nans": False,
}

"""Example generation for testing.

Exports dict of examples, useful for testing as fixtures.

example_dict: dict indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are data objects, considered examples for the mtype
    all examples with same index are considered "same" on scitype content
    if None, indicates that representation is not possible

example_lossy: dict of bool indexed by pairs of str
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are bool, indicate whether representation has information removed
    all examples with same index are considered "same" on scitype content

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
# example 0: multivariate, equally sampled

X = np.array(
    [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 55, 6]], [[1, 2, 3], [42, 5, 6]]],
    dtype=np.int64,
)

example_dict[("numpy3D", "Panel", 0)] = X
example_dict_lossy[("numpy3D", "Panel", 0)] = False

example_dict[("numpyflat", "Panel", 0)] = None
example_dict_lossy[("numpyflat", "Panel", 0)] = None

cols = [f"var_{i}" for i in range(2)]
Xlist = [
    pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=cols),
    pd.DataFrame([[1, 4], [2, 55], [3, 6]], columns=cols),
    pd.DataFrame([[1, 42], [2, 5], [3, 6]], columns=cols),
]

example_dict[("df-list", "Panel", 0)] = Xlist
example_dict_lossy[("df-list", "Panel", 0)] = False

cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(2)]

Xlist = [
    pd.DataFrame([[0, 0, 1, 4], [0, 1, 2, 5], [0, 2, 3, 6]], columns=cols),
    pd.DataFrame([[1, 0, 1, 4], [1, 1, 2, 55], [1, 2, 3, 6]], columns=cols),
    pd.DataFrame([[2, 0, 1, 42], [2, 1, 2, 5], [2, 2, 3, 6]], columns=cols),
]
X = pd.concat(Xlist)
X = X.set_index(["instances", "timepoints"])

example_dict[("pd-multiindex", "Panel", 0)] = X
example_dict_lossy[("pd-multiindex", "Panel", 0)] = False

cols = [f"var_{i}" for i in range(2)]
X = pd.DataFrame(columns=cols, index=[0, 1, 2])
X["var_0"] = pd.Series(
    [pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), pd.Series([1, 2, 3])]
)

X["var_1"] = pd.Series(
    [pd.Series([4, 5, 6]), pd.Series([4, 55, 6]), pd.Series([42, 5, 6])]
)

example_dict[("nested_univ", "Panel", 0)] = X
example_dict_lossy[("nested_univ", "Panel", 0)] = False

if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

    df_dask = convert_pandas_to_dask(
        example_dict[("pd-multiindex", "Panel", 0)], npartitions=1
    )

    example_dict[("dask_panel", "Panel", 0)] = df_dask
    example_dict_lossy[("dask_panel", "Panel", 0)] = False

example_dict_metadata[("Panel", 0)] = {
    "is_univariate": False,
    "is_one_series": False,
    "n_panels": 1,
    "is_one_panel": True,
    "is_equally_spaced": True,
    "is_equal_length": True,
    "is_equal_index": True,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 3,
}

###
# example 1: univariate, equally sampled

X = np.array(
    [[[4, 5, 6]], [[4, 55, 6]], [[42, 5, 6]]],
    dtype=np.int64,
)

example_dict[("numpy3D", "Panel", 1)] = X
example_dict_lossy[("numpy3D", "Panel", 1)] = False

X = np.array([[4, 5, 6], [4, 55, 6], [42, 5, 6]], dtype=np.int64)

example_dict[("numpyflat", "Panel", 1)] = X
example_dict_lossy[("numpyflat", "Panel", 1)] = False

cols = [f"var_{i}" for i in range(1)]
Xlist = [
    pd.DataFrame([[4], [5], [6]], columns=cols),
    pd.DataFrame([[4], [55], [6]], columns=cols),
    pd.DataFrame([[42], [5], [6]], columns=cols),
]

example_dict[("df-list", "Panel", 1)] = Xlist
example_dict_lossy[("df-list", "Panel", 1)] = False

cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(1)]

Xlist = [
    pd.DataFrame([[0, 0, 4], [0, 1, 5], [0, 2, 6]], columns=cols),
    pd.DataFrame([[1, 0, 4], [1, 1, 55], [1, 2, 6]], columns=cols),
    pd.DataFrame([[2, 0, 42], [2, 1, 5], [2, 2, 6]], columns=cols),
]
X = pd.concat(Xlist)
X = X.set_index(["instances", "timepoints"])

example_dict[("pd-multiindex", "Panel", 1)] = X
example_dict_lossy[("pd-multiindex", "Panel", 1)] = False

cols = [f"var_{i}" for i in range(1)]
X = pd.DataFrame(columns=cols, index=[0, 1, 2])
X["var_0"] = pd.Series(
    [pd.Series([4, 5, 6]), pd.Series([4, 55, 6]), pd.Series([42, 5, 6])]
)

example_dict[("nested_univ", "Panel", 1)] = X
example_dict_lossy[("nested_univ", "Panel", 1)] = False

if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

    df_dask = convert_pandas_to_dask(
        example_dict[("pd-multiindex", "Panel", 1)], npartitions=1
    )

    example_dict[("dask_panel", "Panel", 1)] = df_dask
    example_dict_lossy[("dask_panel", "Panel", 1)] = False


example_dict_metadata[("Panel", 1)] = {
    "is_univariate": True,
    "is_one_series": False,
    "n_panels": 1,
    "is_one_panel": True,
    "is_equally_spaced": True,
    "is_equal_length": True,
    "is_equal_index": True,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 3,
}

###
# example 2: univariate, equally sampled, one series

X = np.array(
    [[[4, 5, 6]]],
    dtype=np.int64,
)

example_dict[("numpy3D", "Panel", 2)] = X
example_dict_lossy[("numpy3D", "Panel", 2)] = False

X = np.array([[4, 5, 6]], dtype=np.int64)

example_dict[("numpyflat", "Panel", 2)] = X
example_dict_lossy[("numpyflat", "Panel", 2)] = False

cols = [f"var_{i}" for i in range(1)]
Xlist = [
    pd.DataFrame([[4], [5], [6]], columns=cols),
]

example_dict[("df-list", "Panel", 2)] = Xlist
example_dict_lossy[("df-list", "Panel", 2)] = False

cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(1)]

Xlist = [
    pd.DataFrame([[0, 0, 4], [0, 1, 5], [0, 2, 6]], columns=cols),
]
X = pd.concat(Xlist)
X = X.set_index(["instances", "timepoints"])

example_dict[("pd-multiindex", "Panel", 2)] = X
example_dict_lossy[("pd-multiindex", "Panel", 2)] = False

cols = [f"var_{i}" for i in range(1)]
X = pd.DataFrame(columns=cols, index=[0])
X["var_0"] = pd.Series([pd.Series([4, 5, 6])])

example_dict[("nested_univ", "Panel", 2)] = X
example_dict_lossy[("nested_univ", "Panel", 2)] = False

if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

    df_dask = convert_pandas_to_dask(
        example_dict[("pd-multiindex", "Panel", 2)], npartitions=1
    )

    example_dict[("dask_panel", "Panel", 2)] = df_dask
    example_dict_lossy[("dask_panel", "Panel", 2)] = False

example_dict_metadata[("Panel", 2)] = {
    "is_univariate": True,
    "is_one_series": True,
    "n_panels": 1,
    "is_one_panel": True,
    "is_equally_spaced": True,
    "is_equal_length": True,
    "is_equal_index": True,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 1,
}

###
# example 3: univariate, equally sampled, lossy,
# targets #4299 pd-multiindex panel incorrect is_equally_spaced

X_instances = [0, 0, 0, 1, 1, 1, 2, 2, 2]
X_timepoints = pd.to_datetime([0, 1, 2, 4, 5, 6, 9, 10, 11], unit="s")
X_multiindex = pd.MultiIndex.from_arrays(
    [X_instances, X_timepoints], names=["instances", "timepoints"]
)

X = pd.DataFrame(index=X_multiindex, data=list(range(0, 9)), columns=["var_0"])

example_dict[("pd-multiindex", "Panel", 3)] = X
example_dict_lossy[("pd-multiindex", "Panel", 3)] = False

example_dict_metadata[("Panel", 3)] = {
    "is_univariate": True,
    "is_one_series": False,
    "n_panels": 1,
    "is_one_panel": True,
    "is_equally_spaced": True,
    "is_equal_length": True,
    "is_equal_index": False,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 3,
}

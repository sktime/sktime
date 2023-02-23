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

example_lossy: dict of bool indexed by pairs of str
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are bool, indicate whether representation has information removed
    all examples with same index are considered "same" on scitype content

overall, conversions from non-lossy representations to any other ones
    should yield the element exactly, identidally (given same index)
"""

import pandas as pd

from sktime.utils.validation._dependencies import _check_soft_dependencies

example_dict = dict()
example_dict_lossy = dict()
example_dict_metadata = dict()

###
# example 0: multivariate, equally sampled

cols = ["foo", "bar", "timepoints"] + [f"var_{i}" for i in range(2)]

Xlist = [
    pd.DataFrame(
        [["a", 0, 0, 1, 4], ["a", 0, 1, 2, 5], ["a", 0, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["a", 1, 0, 1, 4], ["a", 1, 1, 2, 55], ["a", 1, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["a", 2, 0, 1, 42], ["a", 2, 1, 2, 5], ["a", 2, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 0, 0, 1, 4], ["b", 0, 1, 2, 5], ["b", 0, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 1, 0, 1, 4], ["b", 1, 1, 2, 55], ["b", 1, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 2, 0, 1, 42], ["b", 2, 1, 2, 5], ["b", 2, 2, 3, 6]], columns=cols
    ),
]

# Xlist = [
#     pd.DataFrame(
#         [["__total", "__total", 0, 6, 100], ["__total", 0, 1, 12, 130],
#           ["__total", 0, 2, 18, 36]], columns=cols
#     ),
#     pd.DataFrame(
#         [["a", 0, 0, 1, 4], ["a", 0, 1, 2, 5], ["a", 0, 2, 3, 6]], columns=cols
#     ),
#     pd.DataFrame(
#         [["a", 1, 0, 1, 4], ["a", 1, 1, 2, 55], ["a", 1, 2, 3, 6]], columns=cols
#     ),
#     pd.DataFrame(
#         [["a", 2, 0, 1, 42], ["a", 2, 1, 2, 5], ["a", 2, 2, 3, 6]], columns=cols
#     ),
#     pd.DataFrame(
#         [["a", "__total", 0, 3, 50], ["a", "__total", 1, 6, 65],
#           ["a", "__total", 2, 9, 18]], columns=cols
#     ),
#     pd.DataFrame(
#         [["b", 3, 0, 1, 4], ["b", 3, 1, 2, 5], ["b", 3, 2, 3, 6]], columns=cols
#     ),
#     pd.DataFrame(
#         [["b", 4, 0, 1, 4], ["b", 4, 1, 2, 55], ["b", 4, 2, 3, 6]], columns=cols
#     ),
#     pd.DataFrame(
#         [["b", 5, 0, 1, 42], ["b", 5, 1, 2, 5], ["b", 5, 2, 3, 6]], columns=cols
#     ),
#     pd.DataFrame(
#         [["b", "__total", 0, 3, 50], ["b", "__total", 1, 6, 65],
#           ["b", "__total", 2, 9, 18]], columns=cols
#     ),
# ]
X = pd.concat(Xlist)
X = X.set_index(["foo", "bar", "timepoints"])
# X = X[["var_0"]]

example_dict[("pd_multiindex_hier", "Hierarchical", 0)] = X
example_dict_lossy[("pd_multiindex_hier", "Hierarchical", 0)] = False

if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

    df_dask = convert_pandas_to_dask(
        example_dict[("pd_multiindex_hier", "Hierarchical", 0)], npartitions=1
    )

    example_dict[("dask_hierarchical", "Hierarchical", 0)] = df_dask
    example_dict_lossy[("dask_hierarchical", "Hierarchical", 0)] = False

example_dict_metadata[("Hierarchical", 0)] = {
    "is_univariate": False,
    "is_one_panel": False,
    "is_one_series": False,
    "is_equally_spaced": True,
    "is_equal_length": True,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 6,
    "n_panels": 2,
}


###
# example 1: univariate, equally sampled

cols = ["foo", "bar", "timepoints"] + [f"var_{i}" for i in range(1)]

Xlist = [
    pd.DataFrame([["a", 0, 0, 1], ["a", 0, 1, 2], ["a", 0, 2, 3]], columns=cols),
    pd.DataFrame([["a", 1, 0, 1], ["a", 1, 1, 2], ["a", 1, 2, 3]], columns=cols),
    pd.DataFrame([["a", 2, 0, 1], ["a", 2, 1, 2], ["a", 2, 2, 3]], columns=cols),
    pd.DataFrame([["b", 0, 0, 1], ["b", 0, 1, 2], ["b", 0, 2, 3]], columns=cols),
    pd.DataFrame([["b", 1, 0, 1], ["b", 1, 1, 2], ["b", 1, 2, 3]], columns=cols),
    pd.DataFrame([["b", 2, 0, 1], ["b", 2, 1, 2], ["b", 2, 2, 3]], columns=cols),
]
X = pd.concat(Xlist)
X = X.set_index(["foo", "bar", "timepoints"])

example_dict[("pd_multiindex_hier", "Hierarchical", 1)] = X
example_dict_lossy[("pd_multiindex_hier", "Hierarchical", 1)] = False

if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

    df_dask = convert_pandas_to_dask(
        example_dict[("pd_multiindex_hier", "Hierarchical", 1)], npartitions=1
    )

    example_dict[("dask_hierarchical", "Hierarchical", 1)] = df_dask
    example_dict_lossy[("dask_hierarchical", "Hierarchical", 1)] = False

example_dict_metadata[("Hierarchical", 1)] = {
    "is_univariate": True,
    "is_one_panel": False,
    "is_one_series": False,
    "is_equally_spaced": True,
    "is_equal_length": True,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 6,
    "n_panels": 2,
}

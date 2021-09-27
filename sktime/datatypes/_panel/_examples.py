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
import numpy as np

example_dict = dict()
example_dict_lossy = dict()

###


X = np.array(
    [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 55, 6]], [[1, 2, 3], [42, 5, 6]]],
    dtype=np.int64,
)

example_dict[("numpy3D", "Panel", 0)] = X
example_dict_lossy[("numpy3D", "Panel", 0)] = False

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

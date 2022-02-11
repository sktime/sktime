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
X = pd.concat(Xlist)
X = X.set_index(["foo", "bar", "timepoints"])

example_dict[("pd_multiindex_hier", "Hierarchical", 0)] = X
example_dict_lossy[("pd_multiindex_hier", "Hierarchical", 0)] = False

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

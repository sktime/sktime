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

example_dict = dict()
example_dict_lossy = dict()
example_dict_metadata = dict()

###
# example 0: univariate

df = pd.DataFrame({"a": [1, 4, 0.5, -3]})

example_dict[("pd_DataFrame_Table", "Table", 0)] = df
example_dict_lossy[("pd_DataFrame_Table", "Table", 0)] = False

arr = np.array([[1], [4], [0.5], [-3]])

example_dict[("numpy2D", "Table", 0)] = arr
example_dict_lossy[("numpy2D", "Table", 0)] = True

arr = np.array([1, 4, 0.5, -3])

example_dict[("numpy1D", "Table", 0)] = arr
example_dict_lossy[("numpy1D", "Table", 0)] = True

series = pd.Series([1, 4, 0.5, -3])

example_dict[("pd_Series_Table", "Table", 0)] = series
example_dict_lossy[("pd_Series_Table", "Table", 0)] = True

list_of_dict = [{"a": 1.0}, {"a": 4.0}, {"a": 0.5}, {"a": -3.0}]

example_dict[("list_of_dict", "Table", 0)] = list_of_dict
example_dict_lossy[("list_of_dict", "Table", 0)] = False


example_dict_metadata[("Table", 0)] = {
    "is_univariate": True,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 4,
}

###
# example 1: multivariate

example_dict[("numpy1D", "Table", 1)] = None
example_dict_lossy[("numpy1D", "Table", 1)] = None

df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})

example_dict[("d_DataFrame_Table", "Table", 1)] = df
example_dict_lossy[("pd_DataFrame_Table", "Table", 1)] = False

arr = np.array([[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]])

example_dict[("numpy2D", "Table", 1)] = arr
example_dict_lossy[("numpy2D", "Table", 1)] = True

example_dict[("pd_Series_Table", "Table", 1)] = None
example_dict_lossy[("pd_Series_Table", "Table", 1)] = None

list_of_dict = [
    {"a": 1.0, "b": 3.0},
    {"a": 4.0, "b": 7.0},
    {"a": 0.5, "b": 2.0},
    {"a": -3.0, "b": -3 / 7},
]

example_dict[("list_of_dict", "Table", 1)] = list_of_dict
example_dict_lossy[("list_of_dict", "Table", 1)] = False

example_dict_metadata[("Table", 1)] = {
    "is_univariate": False,
    "is_empty": False,
    "has_nans": False,
    "n_instances": 4,
}

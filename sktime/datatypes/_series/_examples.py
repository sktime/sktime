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

s = pd.Series([1, 4, 0.5, -3], dtype=np.float64, name="a")

example_dict[("pd.Series", "Series", 0)] = s
example_dict_lossy[("pd.Series", "Series", 0)] = False

df = pd.DataFrame({"a": [1, 4, 0.5, -3]})

example_dict[("pd.DataFrame", "Series", 0)] = df
example_dict_lossy[("pd.DataFrame", "Series", 0)] = False

arr = np.array([[1], [4], [0.5], [-3]])

example_dict[("np.ndarray", "Series", 0)] = arr
example_dict_lossy[("np.ndarray", "Series", 0)] = True

###

example_dict[("pd.Series", "Series", 1)] = None
example_dict_lossy[("pd.Series", "Series", 1)] = None

df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})

example_dict[("pd.DataFrame", "Series", 1)] = df
example_dict_lossy[("pd.DataFrame", "Series", 1)] = False

arr = np.array([[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]])

example_dict[("np.ndarray", "Series", 1)] = arr
example_dict_lossy[("np.ndarray", "Series", 1)] = True

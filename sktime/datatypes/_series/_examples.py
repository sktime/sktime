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

import numpy as np
import pandas as pd

from sktime.datatypes._series._registry import SeriesMtype

example_dict = dict()
example_dict_lossy = dict()

###

s = pd.Series([1, 4, 0.5, -3], dtype=np.float64, name="a")

example_dict[(str(SeriesMtype.pd_series), str(SeriesMtype), 0)] = s
example_dict_lossy[(str(SeriesMtype.pd_series), str(SeriesMtype), 0)] = False

df = pd.DataFrame({"a": [1, 4, 0.5, -3]})

example_dict[(str(SeriesMtype.pd_dataframe), str(SeriesMtype), 0)] = df
example_dict_lossy[(str(SeriesMtype.pd_dataframe), str(SeriesMtype), 0)] = False

arr = np.array([[1], [4], [0.5], [-3]])

example_dict[(str(SeriesMtype.np_array), str(SeriesMtype), 0)] = arr
example_dict_lossy[(str(SeriesMtype.np_array), str(SeriesMtype), 0)] = True

###

example_dict[(str(SeriesMtype.pd_series), str(SeriesMtype), 1)] = None
example_dict_lossy[(str(SeriesMtype.pd_series), str(SeriesMtype), 1)] = None

df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})

example_dict[(str(SeriesMtype.pd_dataframe), str(SeriesMtype), 1)] = df
example_dict_lossy[(str(SeriesMtype.pd_dataframe), str(SeriesMtype), 1)] = False

arr = np.array([[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]])

example_dict[(str(SeriesMtype.np_array), str(SeriesMtype), 1)] = arr
example_dict_lossy[(str(SeriesMtype.np_array), str(SeriesMtype), 1)] = True

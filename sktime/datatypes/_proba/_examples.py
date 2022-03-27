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

example_dict = dict()
example_dict_lossy = dict()
example_dict_metadata = dict()

###
# example 0: univariate, no upper/lower pairs

pred_q = pd.DataFrame({0.2: [1, 2, 3], 0.6: [2, 3, 4]})
pred_q.columns = pd.MultiIndex.from_product([["Quantiles"], [0.2, 0.6]])

# we need to use this due to numerical inaccuracies from the binary based representation
pseudo_0_2 = 2 * np.abs(0.6 - 0.5)

example_dict[("pred_quantiles", "Proba", 0)] = pred_q
example_dict_lossy[("pred_quantiles", "Proba", 0)] = False

pred_int = pd.DataFrame({0.2: [1, 2, 3], 0.6: [2, 3, 4]})
pred_int.columns = pd.MultiIndex.from_tuples(
    [("Coverage", 0.6, "lower"), ("Coverage", pseudo_0_2, "upper")]
)

example_dict[("pred_interval", "Proba", 0)] = pred_int
example_dict_lossy[("pred_interval", "Proba", 0)] = False


example_dict_metadata[("Proba", 0)] = {
    "is_univariate": True,
    "is_empty": False,
    "has_nans": False,
}

###
# example 1: multi, no upper/lower pairs

pred_q = pd.DataFrame({0.2: [1, 2, 3], 0.6: [2, 3, 4], 42: [5, 3, -1], 46: [5, 3, -1]})
pred_q.columns = pd.MultiIndex.from_product([["foo", "bar"], [0.2, 0.6]])

example_dict[("pred_quantiles", "Proba", 1)] = pred_q
example_dict_lossy[("pred_quantiles", "Proba", 1)] = False

pred_int = pd.DataFrame(
    {0.2: [1, 2, 3], 0.6: [2, 3, 4], 42: [5, 3, -1], 46: [5, 3, -1]}
)
pred_int.columns = pd.MultiIndex.from_tuples(
    [
        ("foo", 0.6, "lower"),
        ("foo", pseudo_0_2, "upper"),
        ("bar", 0.6, "lower"),
        ("bar", pseudo_0_2, "upper"),
    ]
)

example_dict[("pred_interval", "Proba", 1)] = pred_int
example_dict_lossy[("pred_interval", "Proba", 1)] = False


example_dict_metadata[("Proba", 1)] = {
    "is_univariate": False,
    "is_empty": False,
    "has_nans": False,
}

###
# example 2: univariate, upper/lower pairs

alphas = np.array([0.2, 0.8, 0.4, 0.6])
coverage = 2 * np.abs(0.5 - alphas)
alphas = list(alphas)
coverage = list(coverage)

pred_q = pd.DataFrame(
    {0.2: [1, 2, 3], 0.8: [2.5, 3.5, 4.5], 0.4: [1.5, 2.5, 3.5], 0.6: [2, 3, 4]}
)
pred_q.columns = pd.MultiIndex.from_product([["Quantiles"], alphas])

# we need to use this due to numerical inaccuracies from the binary based representation
pseudo_0_2 = 2 * np.abs(0.6 - 0.5)
pseudo_0_6 = 2 * np.abs(0.8 - 0.5)

example_dict[("pred_quantiles", "Proba", 2)] = pred_q
example_dict_lossy[("pred_quantiles", "Proba", 2)] = False

pred_int = pred_q.copy()
pred_int.columns = pd.MultiIndex.from_arrays(
    [4 * ["Coverage"], coverage, 2 * ["lower", "upper"]]
)

example_dict[("pred_interval", "Proba", 2)] = pred_int
example_dict_lossy[("pred_interval", "Proba", 2)] = False


example_dict_metadata[("Proba", 2)] = {
    "is_univariate": True,
    "is_empty": False,
    "has_nans": False,
}

###
# example 3: multi, no upper/lower pairs

pred_q = pd.DataFrame(
    {
        0.2: [1, 2, 3],
        0.8: [2.5, 3.5, 4.5],
        0.4: [1.5, 2.5, 3.5],
        0.6: [2, 3, 4],
        42: [5, 3, -1],
        43: [5, 3, -1],
        44: [5, 3, -1],
        45: [5, 3, -1],
    }
)
pred_q.columns = pd.MultiIndex.from_product([["foo", "bar"], [0.2, 0.8, 0.4, 0.6]])

example_dict[("pred_quantiles", "Proba", 3)] = pred_q
example_dict_lossy[("pred_quantiles", "Proba", 3)] = False

pred_int = pred_q.copy()
pred_int.columns = pd.MultiIndex.from_arrays(
    [4 * ["foo", "bar"], 2 * coverage, 4 * ["lower", "upper"]]
)

example_dict[("pred_interval", "Proba", 3)] = pred_int
example_dict_lossy[("pred_interval", "Proba", 3)] = False


example_dict_metadata[("Proba", 3)] = {
    "is_univariate": False,
    "is_empty": False,
    "has_nans": False,
}

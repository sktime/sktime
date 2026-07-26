"""Machine type converters for Series scitype.

Exports conversion and mtype dictionary for Series scitype:

convert_dict: dict indexed by triples of str
  1st element = convert from - str
  2nd element = convert to - str
  3rd element = considered as this scitype - str
elements are conversion functions of machine type (1st) -> 2nd

Function signature of all elements
convert_dict[(from_type, to_type, as_scitype)]

Parameters
----------
obj : from_type - object to convert
store : dictionary - reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

Raises
------
ValueError and TypeError, if requested conversion is not possible
                            (depending on conversion logic)
"""

__author__ = ["fkiraly"]

__all__ = ["convert_dict"]

import numpy as np
import pandas as pd

# this needs to be refactored with the convert module
MTYPE_LIST_PROBA = ["pred_interval", "pred_quantiles"]

##############################################################
# methods to convert one machine type to another machine type
##############################################################

convert_dict = dict()


def convert_identity(obj, store=None):
    return obj


# assign identity function to type conversion to self
for tp in MTYPE_LIST_PROBA:
    convert_dict[(tp, tp, "Proba")] = convert_identity


def convert_pred_interval_to_quantiles(y_pred, inplace=False):
    """Convert interval predictions to quantile predictions.

    Parameters
    ----------
    y_pred : pd.DataFrame
        Column has multi-index: first level is variable name from y in fit,
            second level coverage fractions for which intervals were computed.
                in the same order as in input `coverage`.
            Third level is string "lower" or "upper", for lower/upper interval end.
        Row index is fh. Entries are forecasts of lower/upper interval end,
            for var in col index, at nominal coverage in selencond col index,
            lower/upper depending on third col index, for the row index.
            Upper/lower interval end forecasts are equivalent to
            quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
    inplace : bool, optional, default=False
        whether to copy the input data frame (False), or modify (True)

    Returns
    -------
    y_pred : pd.DataFrame
        Column has multi-index: first level is variable name from y in fit,
            second level being the values of alpha passed to the function.
        Row index is fh. Entries are quantile forecasts, for var in col index,
            at quantile probability in second col index, for the row index.
    """
    if not inplace:
        y_pred = y_pred.copy()

    # all we need to do is to replace the index with var_names/alphas
    # var_names will be the same as interval level 0
    idx = y_pred.columns
    var_names = idx.get_level_values(0)

    # alpha, we compute by the coverage/alphas formula correspondence
    coverages = idx.get_level_values(1)
    alphas = np.array(coverages.copy())
    lower_upper = idx.get_level_values(2)

    lower_selector = lower_upper == "lower"
    upper_selector = lower_upper == "upper"

    alphas[lower_selector] = 0.5 - 0.5 * alphas[lower_selector]
    alphas[upper_selector] = 0.5 + 0.5 * alphas[upper_selector]

    # idx returned by _predict_quantiles
    #   is 2-level MultiIndex with variable names, alpha
    int_idx = pd.MultiIndex.from_arrays([var_names, alphas])
    y_pred.columns = int_idx

    return y_pred


def convert_interval_to_quantiles(obj: pd.DataFrame, store=None) -> pd.DataFrame:
    return convert_pred_interval_to_quantiles(y_pred=obj)


convert_dict[("pred_interval", "pred_quantiles", "Proba")] = (
    convert_interval_to_quantiles
)


def convert_pred_quantiles_to_interval(y_pred, inplace=False):
    """Convert quantile predictions to interval predictions.

    Parameters
    ----------
    y_pred : pd.DataFrame
        Column has multi-index: first level is variable name from y in fit,
            second level being the values of alpha passed to the function.
        Row index is fh. Entries are quantile forecasts, for var in col index,
            at quantile probability in second col index, for the row index.
    inplace : bool, optional, default=False
        whether to copy the input data frame (False), or modify (True)

    Returns
    -------
    y_pred : pd.DataFrame
        Column has multi-index: first level is variable name from y in fit,
            second level coverage fractions for which intervals were computed.
                in the same order as in input `coverage`.
            Third level is string "lower" or "upper", for lower/upper interval end.
        Row index is fh. Entries are forecasts of lower/upper interval end,
            for var in col index, at nominal coverage in selencond col index,
            lower/upper depending on third col index, for the row index.
            Upper/lower interval end forecasts are equivalent to
            quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
    """
    if not inplace:
        y_pred = y_pred.copy()

    # all we need to do is to replace the index with var_names/alphas
    # var_names will be the same as interval level 0
    idx = y_pred.columns
    var_names = idx.get_level_values(0)

    # coverages we compute by the coverage/alphas formula correspondence
    alphas = idx.get_level_values(1)
    alphas = np.array(alphas.copy())
    coverages = 2 * np.abs(0.5 - alphas)
    lower_upper = ["lower" if a <= 0.5 else "upper" for a in alphas]

    # idx returned by _predict_quantiles
    #   is 3-level MultiIndex with variable names, coverages, lower/upper
    int_idx = pd.MultiIndex.from_arrays([var_names, coverages, lower_upper])
    y_pred.columns = int_idx

    return y_pred


def convert_quantiles_to_interval(obj: pd.DataFrame, store=None) -> pd.DataFrame:
    return convert_pred_quantiles_to_interval(y_pred=obj)


convert_dict[("pred_quantiles", "pred_interval", "Proba")] = (
    convert_quantiles_to_interval
)

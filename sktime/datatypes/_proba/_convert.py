"""Machine type converters for Proba scitype.

Conversions are implemented as ``BaseConverter`` classes, one class per
conversion, and collected by ``generate_convert_dict`` in ``datatypes._convert``.
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.datatypes._base import BaseConverter

# this needs to be refactored with the convert module
MTYPE_LIST_PROBA = ["pred_interval", "pred_quantiles"]


class ProbaIdentity(BaseConverter):
    """All Proba scitype conversions of any mtype to itself.

    This is the identity conversion, the object is returned as is.
    """

    _tags = {
        "object_type": "converter",
        "mtype_from": None,
        "mtype_to": None,
        "multiple_conversions": True,
        "python_version": None,
        "python_dependencies": None,
    }

    @classmethod
    def get_conversions(cls):
        """Get all conversions.

        Returns
        -------
        list of tuples (str, str)
            List of all conversions in this class.
        """
        return [(tp, tp) for tp in MTYPE_LIST_PROBA]

    def _convert(self, obj, store=None):
        return obj


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


class PredIntervalToPredQuantiles(BaseConverter):
    """Conversion: pred_interval -> pred_quantiles, of Proba scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pred_interval",
        "mtype_to": "pred_quantiles",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_pred_interval_to_quantiles(y_pred=obj)


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


class PredQuantilesToPredInterval(BaseConverter):
    """Conversion: pred_quantiles -> pred_interval, of Proba scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pred_quantiles",
        "mtype_to": "pred_interval",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_pred_quantiles_to_interval(y_pred=obj)

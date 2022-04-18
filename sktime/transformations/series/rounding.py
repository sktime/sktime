#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Implements a transformer which will round Series to the nearest integer,
 multiple of a number; or nearest value in a set of specified values."""

__author__ = ["AJarman"]
__all__ = ["Discretizer"]

# stdlib
from typing import Optional, Union, List

# data
import pandas as pd
import numpy as np

# internal
from sktime.transformations.base import BaseTransformer


class Discretizer(BaseTransformer):
    """
    A transformer which can be used to round values by methods useful for forecasting.
    This is particularly useful as the last step of a `TransformedTargetForecaster` to convert
     floating point forecasts to discrete values.
    The default method is to round to the nearest integer.
    - By using `round_to_dp`, the values will be rounded to the specified number of decimal places.
    - By using `round_to_multiple`, the values will be rounded to the nearest multiple
     of the specified value.
    - By using `round_to_list`, the values will be rounded to the nearest value in the specified
     list.

    Parameters
    ----------
    round_to_dp: Optional[int], default=None
        How many decimal places to round to.
    round_to_multiple: Optional[Union[int, float]], default=None
        A multiple (numeric) to round to.
    round_to_list: Optional[List[Union[float, int]]], default=None
        A list of values to round to.

    Examples
    --------
    >>> from sktime.transformations.series.rounding import Discretizer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Discretizer() # default, round to nearest integer.
    >>> transformer1 = Discretizer(round_to_dp=2) # round to 2 decimal places.
    >>> transformer2 = Discretizer(round_to_multiple=2) # round to multiples of 2.
    >>> transformer2 = Discretizer(round_to_list=[1,10,100,1000,10000]) # round to specific values.
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": False,
        # is transform result always guaranteed to be equal length (and series)?
        #   not relevant for transformers that return Primitives in transform-output
        "handles-missing-data": False,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
    }

    def __init__(self, round_to_dp: Optional[int] = None,
                 round_to_multiple: Optional[Union[int, float]] = None,
                 round_to_list: Optional[List[Union[float, int]]] = None):

        # Write any hyper-parameters and components to self
        self.round_to_dp = round_to_dp
        self.round_to_multiple = round_to_multiple
        self.round_to_list = round_to_list

        self._default: bool = not any((round_to_dp, round_to_multiple, round_to_list))

        # Base class init
        super(Discretizer, self).__init__()

        # validate arguments
        self._check_arguments()

    def _check_method(self):
        """
        Check that the inputs are valid for the transformation.

        Warns
        -----
        UserWarning
            (1) If more than one of round_to_dp, round_to_multiple, or round_to_list are not None.

        Raises
        ------
        ValueError
            (1)  `round_to_dp` is not None and is not an int or float.
            (2)  `round_to_multiple` is not None and is not an int or float.
            (3)  `round_to_list` is not None and is not an int or float.
            (4)  `round_to_list` is not None and any element is not an int or float.
        """
        if sum((bool(i) for i in (
                self.round_to_dp,
                self.round_to_list,
                self.round_to_multiple
                ))) > 1:
            # Raise warning if more than one argument passed
            raise UserWarning(
                """
                    More than one argument has been passed,
                    only one argument can be applied.
                    These are evaluated/applied in the order:
                    `round_to_dp` , `round_to_list`, `round_to_multiple`.
                    """)

        if self.round_to_dp is not None and not isinstance(
                self.round_to_dp, (int, float)):
            raise ValueError(
                f"""
                round_to_dp: {self.round_to_dp} is not an integer or float.
                """)

        if self.round_to_multiple is not None and not isinstance(
                self.round_to_multiple, (int, float)):
            raise ValueError(
                f"""
                round_to_multiple: {self.round_to_multiple} is not an integer or float.
                """
            )
        if self.round_to_list is not None and not isinstance(
                self.round_to_list, list):
            raise ValueError(
                f"""
                round_to_list: {self.round_to_list}, {type(self.round_to_list)} is not a list.
                """)
        if self.round_to_list is not None and not all(
            (
                isinstance(i, (int, float)) for i in self.round_to_list)):
            wrong_types: list = [
                type(i) for i in self.round_to_list if not isinstance(
                    i, (int, float))]
            raise ValueError(
                f"""
                round_to_list: {self.round_to_list}, not all elements are integer or float.
                {wrong_types} were found in the list.
                """)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """

        if self._default:
            # no args passed, use default
            # traditional rounding and convert to int
            return np.round_(X).astype(int)

        if self.round_to_dp is not None:
            # Return rounded array/series
            return np.round_(X, decimals=self.round_to_dp)

        if self.round_to_multiple is not None:
            # Return rounded values
            return np.round_(X / self.round_to_multiple) * self.round_to_multiple

        if self.round_to_list is not None:
            # First convert list to ndarray
            # Return result from method call
            return self.round_to_nearest_value(X, np.array(self.round_to_list))

    def _inverse_transform(self, X, y=None):
        """Not possible to back-calculate transformation without storing all values
        , therefore just return X"""
        return X

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params


def round_to_nearest_value(values: Union[np.ndarray, pd.Series],
                           allowed_values: Union[np.ndarray]) -> Union[np.ndarray, pd.Series]:
    """
    Finds the closest values in allowed_values from values.
    Returns the original index if values is a Series.

    Parameters
    ----------
    values : Union[np.ndarray,pd.Series]
        Values to be rounded to values of second array.
    allowed_values : Union[np.ndarray]
        Values that are allowed, (desired discrete outputs).

    Returns
    -------
    Union[np.ndarray,pd.Series]
        Closest values in allowed_values from values.
    """

    if isinstance(values, pd.Series):
        original_index = values.index
        is_series = True
    else:
        is_series = False

    # Sort outputs array if isn't sorted.
    if not all(allowed_values[:-1] <= allowed_values[1:]):
        allowed_values = np.sort(allowed_values)

    # find index to the left of the allowed_values
    idxs = np.searchsorted(allowed_values, values, side="left")
    # find indexes where previous (left) index is closer
    prev_idx_is_less = (
        (idxs == len(allowed_values)) | (
            np.fabs(values - allowed_values[np.maximum(idxs-1, 0)]) <
            np.fabs(values - allowed_values[np.minimum(idxs, len(allowed_values)-1)])
            )
        )
    # apply change to indexes
    idxs[prev_idx_is_less] -= 1

    rounded_vals = allowed_values[idxs]

    if is_series:
        return pd.Series(rounded_vals, index=original_index)
    else:
        return rounded_vals

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dummy variables for time series."""

__author__ = ["ericjb"]

import calendar
from typing import Optional

import pandas as pd

from sktime.transformations.base import BaseTransformer


class SeasonalDummies(BaseTransformer):
    r"""Seasonal Dummy Features for time series seasonality.

    A standard approach to capture seasonal effects is to add dummy exogenous variables,
    one for each season. e.g. for monthly seasonality add binary dummy variables
    Jan, Feb, .... For time 't', these variables are set to 1 (resp 0) if 't' occurs
    (resp does not occur) on that season. To avoid collinearity, one season is dropped
    when an intercept is also part of the model.

    In the language of machine learning, the use of seasonal dummies is one hot encoding
    for the seasonal categorical variable.


    Parameters
    ----------
    sp : int, optional, default = None
    freq : str, optional, default = None
    drop : bool, default = True
        Drop the first seasonal dummy? (Should be True if model contains an intercept)

    Examples
    --------
    >>> from sktime.transformations.series.dummies import SeasonalDummies
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SeasonalDummies()
    >>> X = transformer.fit_transform(y=y, X=None)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ltsaprounis", "blazingbhavneek"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Panel
        "scitype:transform-output": "Panel",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd.Series",
        ],  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": [
            "None",
            "pd.Series",
        ],  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": [
            pd.PeriodIndex,
            pd.DatetimeIndex,
        ],  # index type that needs to be enforced
        # in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
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

    def __init__(
        self,
        sp: Optional[int] = None,
        freq: Optional[str] = None,
        drop: Optional[bool] = True,
    ):
        self.sp = sp
        self.freq = freq
        self.drop = drop

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        Does nothing.

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Logic:
          - if X is not None, use its index to create the seasonal dummies
            else if X is None, use y's index to create the seasonal dummies
          - if sp is not None, use it to determine which dummies to create
          - else if sp is None, get the freq from self.freq
          - if sp is None and freq is None
              - get the freq from the index
              - if the index does not have a freq, raise an error


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
        if X is not None:
            index = X.index
        elif y is not None:
            index = y.index
        else:
            raise ValueError("Both X and y cannot be None")

        if self.sp is not None:
            sp = self.sp
        else:
            sp = self.freq
            if sp is None:
                sp = index.freq
                if sp is None:
                    raise ValueError("Frequency cannot be determined from the index")

        if isinstance(index, pd.DatetimeIndex):
            period_index = index.to_period(freq=sp)
        else:
            period_index = index

        # Extract month from the period index
        month_index = period_index.month

        # Create dummy variables for the months
        dummies = pd.get_dummies(month_index, prefix="", prefix_sep="")
        dummies.columns = dummies.columns.map(lambda x: calendar.month_abbr[int(x)])
        dummies = dummies.astype(int)  # Convert boolean values to integers

        dummies.index = index

        if self.drop:
            dummies = dummies.iloc[
                :, 1:
            ]  # drop the first column to avoid multicollinearity

        if X is not None:
            X_transformed = pd.concat([X, dummies], axis=1, copy=True)
        else:
            X_transformed = dummies

        return X_transformed

    # TBD
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        param0 = {}
        param1 = {"sp": 12}
        param2 = {"sp": 12, "drop": False}

        return [param0, param1, param2]

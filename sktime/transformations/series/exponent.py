#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformers raise time series to user provided exponent."""

__author__ = ["Ryan Kuhns"]
__all__ = ["ExponentTransformer", "SqrtTransformer"]

from warnings import warn

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class ExponentTransformer(BaseTransformer):
    """Apply element-wise exponentiation transformation to a time series.

    Transformation performs the following operations element-wise:
        * adds the constant `offset` (shift)
        * raises to the `power` provided (exponentiation)
    Offset="auto" computes offset as the smallest offset that ensure all elements
    are non-negative before exponentiation.

    Parameters
    ----------
    power : int or float, default=0.5
        The power to raise the input timeseries to.

    offset : "auto", int or float, default="auto"
        Offset to be added to the input timeseries prior to raising
        the timeseries to the given `power`. If "auto" the series is checked to
        determine if it contains negative values. If negative values are found
        then the offset will be equal to the absolute value of the most negative
        value. If not negative values are present the offset is set to zero.
        If an integer or float value is supplied it will be used as the offset.

    Attributes
    ----------
    power : int or float
        User supplied power.

    offset : int or float, or iterable.
        User supplied offset value.
        Scalar or 1D iterable with as many values as X columns in transform.

    See Also
    --------
    BoxCoxTransformer :
        Applies Box-Cox power transformation. Can help normalize data and
        compress variance of the series.
    LogTransformer :
        Transformer input data using natural log. Can help normalize data and
        compress variance of the series.
    sktime.transformations.series.exponent.SqrtTransformer :
        Transform input data by taking its square root. Can help compress
        variance of input series.

    Notes
    -----
    For an input series `Z` the exponent transformation is defined as
    :math:`(Z + offset)^{power}`.

    Examples
    --------
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ExponentTransformer()
    >>> y_transform = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "univariate-only": False,
        "capability:inverse_transform": True,
    }

    def __init__(self, power=0.5, offset="auto"):
        self.power = power
        self.offset = offset

        if not isinstance(self.power, (int, float)):
            raise ValueError(
                f"Expected `power` to be int or float, but found {type(self.power)}."
            )

        offset_types = (int, float, pd.Series, np.ndarray)
        if not isinstance(offset, offset_types) and offset != "auto":
            raise ValueError(
                f"Expected `offset` to be int or float, but found {type(self.offset)}."
            )

        super(ExponentTransformer, self).__init__()

        if abs(power) < 1e-6:
            warn(
                "power close to zero passed to ExponentTransformer, "
                "inverse_transform will default to identity "
                "if called, in order to avoid division by zero"
            )
            self.set_tags(**{"skip-inverse-transform": True})

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        offset = self._get_offset(X)
        Xt = X.add(offset).pow(self.power)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be inverse transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            inverse transformed version of X
        """
        offset = self._get_offset(X)
        Xt = X.pow(1.0 / self.power).add(-offset)
        return Xt

    def _get_offset(self, X):
        if self.offset == "auto":
            Xmin = X.min()
            offset = -Xmin * (Xmin < 0)
        else:
            offset = self.offset

        if isinstance(X, pd.DataFrame):
            if isinstance(offset, (int, float)):
                offset = pd.Series(offset, index=X.columns)
            else:
                offset = pd.Series(offset)
                offset.index = X.columns

        return offset

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
        return [{"power": 2.5, "offset": 1}, {"power": 0}]


class SqrtTransformer(ExponentTransformer):
    """Apply element-sise square root transformation to a time series.

    Transformation performs the following operations element-wise:
        * adds the constant `offset` (shift)
        * applies the square root
    Offset="auto" computes offset as the smallest offset that ensure all elements
    are non-negative before taking the square root.

    Parameters
    ----------
    offset : "auto", int or float, default="auto"
        Offset to be added to the input timeseries prior to raising
        the timeseries to the given `power`. If "auto" the series is checked to
        determine if it contains negative values. If negative values are found
        then the offset will be equal to the absolute value of the most negative
        value. If not negative values are present the offset is set to zero.
        If an integer or float value is supplied it will be used as the offset.

    Attributes
    ----------
    offset : int or float
        User supplied offset value.

    See Also
    --------
    BoxCoxTransformer :
        Applies Box-Cox power transformation. Can help normalize data and
        compress variance of the series.
    LogTransformer :
        Transformer input data using natural log. Can help normalize data and
        compress variance of the series.
    sktime.transformations.series.exponent.ExponentTransformer :
        Transform input data by raising it to an exponent. Can help compress
        variance of series if a fractional exponent is supplied.

    Notes
    -----
    For an input series `Z` the square root transformation is defined as
    :math:`(Z + offset)^{0.5}`.

    Examples
    --------
    >>> from sktime.transformations.series.exponent import SqrtTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SqrtTransformer()
    >>> y_transform = transformer.fit_transform(y)
    """

    def __init__(self, offset="auto"):
        super().__init__(power=0.5, offset=offset)

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
        return [{}, {"offset": 4.2}]

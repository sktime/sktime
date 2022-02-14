# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Bootstrapping methods for time series."""


__author__ = ["ltsaprounis"]

# todo: add any necessary sktime external imports here

from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import STL as _STL

from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.boxcox import BoxCoxTransformer


class UnivariateBootsrappingTransformer(BaseTransformer):
    """Creates a population of similar time series.

    Parameters
    ----------
    number_of_new_series : int, optional
        [description], by default 10
    sp : int, optional
        [description], by default 12
    block_length : int, optional
        [description], by default None
    return_actual : bool, optional
        [description], by default True
    series_name : str, optional
        [description], by default None
    boxcox_bounds : Tuple, optional
        [description], by default None
    boxcox_method : str, optional
        [description], by default "guerrero"
    seasonal : int, optional
        [description], by default 7
    trend : int, optional
        [description], by default None
    low_pass : int, optional
        [description], by default None
    seasonal_deg : int, optional
        [description], by default 1
    trend_deg : int, optional
        [description], by default 1
    low_pass_deg : int, optional
        [description], by default 1
    robust : bool, optional
        [description], by default False
    seasonal_jump : int, optional
        [description], by default 1
    trend_jump : int, optional
        [description], by default 1
    low_pass_jump : int, optional
        [description], by default 1
    inner_iter : int, optional
        [description], by default None
    outer_iter : int, optional
        [description], by default None
    """

    _tags = {
        # todo: what is the scitype of X: Series, or Panel
        "scitype:transform-input": "Series",
        # todo: what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Panel",
        # todo: what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
        # X_inner_mtype can be Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit-in-transform": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(
        self,
        number_of_new_series: int = 10,
        sp: int = 12,
        block_length: int = None,
        return_actual: bool = True,
        series_name: str = None,
        boxcox_bounds: Tuple = None,
        boxcox_method: str = "guerrero",
        seasonal: int = 7,
        trend: int = None,
        low_pass: int = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        low_pass_jump: int = 1,
        inner_iter: int = None,
        outer_iter: int = None,
    ):
        self.number_of_new_series = number_of_new_series
        self.sp = sp
        self.block_length = block_length
        self.return_actual = return_actual
        self.series_name = series_name
        self.boxcox_bounds = boxcox_bounds
        self.boxcox_method = boxcox_method
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter

        super(UnivariateBootsrappingTransformer, self).__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

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
        if self.sp <= 2:
            raise ValueError(
                "UnivariateBootstrappingTransformer does not support non-seasonal data"
            )

        if len(X) <= self.sp:
            raise ValueError(
                "UnivariateBootstrappingTransformer requires that sp is greater than"
                "the length of X"
            )
        else:
            # implement as static method
            self.block_length_ = (
                self.block_length
                if self.block_length is not None
                else min(self.sp * 2, len(X) - self.sp)
            )

        # fit boxcox to get lambda and transform X
        self.BoxCoxTransformer_ = BoxCoxTransformer(
            sp=self.sp, bounds=self.boxcox_bounds, method=self.boxcox_method
        )
        self.BoxCoxTransformer_.fit(X)

        return self

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
        X_index = X.index

        X_transformed = self.BoxCoxTransformer_.transform(X)

        # fit STL on X_transformed series and extract trend, seasonal and residuals
        STL_ = _STL(X_transformed, period=self.sp).fit()
        seasonal = pd.Series(STL_.seasonal, index=X_index)
        resid = pd.Series(STL_.resid, index=X_index)
        trend = pd.Series(STL_.trend, index=X_index)

        # time series id prefix
        prefix = self.series_name + "_" if self.series_name is not None else ""

        # initialize the dataframe that will store the bootstrapped series
        if self.return_actual:
            df = pd.DataFrame(
                X.values,
                index=pd.MultiIndex.from_product(
                    iterables=[[f"{prefix}actual"], X_index],
                    names=["series_id", "time_index"],
                ),
            )
        else:
            df = pd.DataFrame()

        # create multiple series
        for i in range(self.number_of_new_series):
            new_series = self.BoxCoxTransformer_.inverse_transform(
                moving_block_bootstrap(ts=resid, block_length=self.block_length_)
                + seasonal
                + trend
            )

            new_series_id = f"{prefix}synthetic_{i}"
            new_df_index = pd.MultiIndex.from_product(
                iterables=[[new_series_id], new_series.index],
                names=["series_id", "time_index"],
            )

            df = df.append(pd.DataFrame(data=new_series.values, index=new_df_index))

        return df

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [{}]

        return params


def moving_block_bootstrap(
    ts: pd.Series, block_length: int, replacement=False
) -> pd.Series:
    """Implement the moving block bootstrap method MBB.

    Parameters
    ----------
    ts : pd.Series
        a stationary time series
    block_length : int
        the length of the bootstrapping block

    Returns
    -------
    pd.Series
        bootstrapped time series

    References
    ----------
    .. [1] Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential
        smoothing methods using STL decomposition and Box-Cox transformation.
        International Journal of Forecasting, 32(2), 303-312
    .. [2] Kunsch HR (1989) The jackknife and the bootstrap for general stationary
        observations. Annals of Statistics 17(3), 1217-1241
    .. [3] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3.
        Accessed on January 24th 2022.
    """
    ts_length = len(ts)
    ts_index = ts.index
    ts_values = ts.values

    if ts_length <= block_length:
        raise ValueError("ts length should be greater than block_length")
    total_num_blocks = int(ts_length / block_length) + 2

    block_origns = np.random.choice(
        ts_length - block_length + 1, size=total_num_blocks, replace=replacement
    )

    mbb_values = [val for i in block_origns for val in ts_values[i : i + block_length]]
    # remove the first few observations and ensure new series has the same length
    # as the original
    remove_first = np.random.choice(block_length - 1)
    mbb_values = mbb_values[remove_first : remove_first + ts_length]
    mbb_series = pd.Series(data=mbb_values, index=ts_index)

    return mbb_series

# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Bootstrapping methods for time series."""

__author__ = ["ltsaprounis"]

from copy import copy
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import STL as _STL

from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.boxcox import BoxCoxTransformer


class BootsrappingTransformer(BaseTransformer):
    """Creates a population of similar time series.

    This method utilises a form of bootstrapping to generate a population of
    similar time series to the input time series [1]_, [2]_.

    First the observed time series is transformed using a Box-Cox transformation to
    stabilise the variance. Then it's decomposed to seasonal, trend and residual
    time series, using the STL implementation from statsmodels [4]_. We then sample
    blocks from the residuals time series using the Moving Block Bootstrapping (MBB)
    method [3]_ to create synthetic residuals series that mimic the autocorrelation
    patterns of the observed series. Finally these bootstrapped residuals are added
    to the season and trend components and we use the inverse Box-Cox transform to
    return a panel of similar time series.

    The resulting panel can be used for Bagging forecasts, prediction intervals and
    data augmentation.

    Parameters
    ----------
    n_series : int, optional
        The number of bootstraped time series that will be generated, by default 10
    sp : int, optional
        Seasonal periodicity of the data in integer form, by default 12.
        Must be an integer >= 2
    block_length : int, optional
        The length of the block in the MBB method, by default None.
        If not provided, the following heuristic is used, the block length will the
        minimum between 2*sp and len(X) - sp.
    sampling_replacement: bool, optional
        Whether the MBB sample is with or without replacement, by default False.
    return_actual : bool, optional
        If True the output will contain the actual time series, by default True.
        The actual time series will be labelled as "<series_name>_actual" (or "actual"
        if series name is None).
    series_name : str, optional
        The series name, by default None
        If provided, the synthetic series names will have the series_name as a
        prefix followed by an undescore e.g. "<series_name>_synthetic_1". If not
        provided the series names will no prefix will be present e.g. "synthetic_1".
    lambda_bounds : Tuple, optional
        Lower and upper bounds used to restrict the feasible range
        when solving for the value of lambda, by default None.
    lambda_method : str, optional
        {"pearsonr", "mle", "all", "guerrero"}, by default "guerrero"
        The optimization approach used to determine the lambda value used
        in the Box-Cox transformation.
    seasonal : int, optional
        Length of the seasonal smoother. Must be an odd integer, and should
        normally be >= 7, by default 7
    trend : int, optional
        Length of the trend smoother, by default None.
        Must be an odd integer. If not provided uses the smallest odd integer greater
        than 1.5 * period / (1 - 1.5 / seasonal), following the suggestion in the
        original implementation.
    low_pass : int, optional
        Length of the low-pass filter, by default None.
        Must be an odd integer >=3. If not provided, uses the smallest odd
        integer > period
    seasonal_deg : int, optional
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend), by default 1.
    trend_deg : int, optional
        Degree of trend LOESS. 0 (constant) or 1 (constant and trend), by default 1.
    low_pass_deg : int, optional
        Degree of low pass LOESS. 0 (constant) or 1 (constant and trend), by default 1
    robust : bool, optional
        Flag indicating whether to use a weighted version that is robust to
        some forms of outliers, by default False
    seasonal_jump : int, optional
        Positive integer determining the linear interpolation step, by default 1.
        If larger than 1, the LOESS is used every seasonal_jump points and linear
        interpolation is between fitted points. Higher values reduce estimation time.
    trend_jump : int, optional
        Positive integer determining the linear interpolation step, by default 1.
        If larger than 1, the LOESS is used every trend_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation time.
    low_pass_jump : int, optional
        Positive integer determining the linear interpolation step, by default 1.
        If larger than 1, the LOESS is used every low_pass_jump points and values
        between the two are linearly interpolated. Higher values reduce estimation
        time.
    inner_iter : int, optional
        Number of iterations to perform in the inner loop, by default None.
        If not provided uses 2 if robust is True, or 5 if not. This param goes into
        STL.fit() from statsmodels.
    outer_iter : int, optional
        Number of iterations to perform in the outer loop, by default None.
        If not provided uses 15 if robust is True, or 0 if not.
        This param goes into STL.fit() from statsmodels.

    References
    ----------
    .. [1] Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential
        smoothing methods using STL decomposition and Box-Cox transformation.
        International Journal of Forecasting, 32(2), 303-312
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3.
        Accessed on February 13th 2022.
    .. [3] Kunsch HR (1989) The jackknife and the bootstrap for general stationary
        observations. Annals of Statistics 17(3), 1217-1241
    .. [4] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html

    Examples
    --------
    >>> from sktime.transformations.series.bootstrapping import (
    ...     BootsrappingTransformer
    ... )
    >>> from sktime.datasets import load_airline
    >>> from sktime.utils.plotting import plot_series
    >>> y = load_airline()
    >>> transformer = BootsrappingTransformer(10)
    >>> y_hat = transformer.fit_transform(y)
    >>> series_list = []
    >>> names = []
    >>> for group, series in y_hat.groupby(level=[0], as_index=False):
    ...     series.index = series.index.droplevel(0)
    ...     series_list.append(series)
    ...     names.append(group)
    >>> plot_series(*series_list, labels=names)
    (...)
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
        n_series: int = 10,
        sp: int = 12,
        block_length: int = None,
        sampling_replacement: bool = False,
        return_actual: bool = True,
        series_name: str = None,
        lambda_bounds: Tuple = None,
        lambda_method: str = "guerrero",
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
        self.n_series = n_series
        self.sp = sp
        self.block_length = block_length
        self.sampling_replacement = sampling_replacement
        self.return_actual = return_actual
        self.series_name = series_name
        self.lambda_bounds = lambda_bounds
        self.lambda_method = lambda_method
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

        super(BootsrappingTransformer, self).__init__()

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
            raise NotImplementedError(
                "BootstrappingTransformer does not support non-seasonal data"
            )

        if len(X) <= self.sp:
            raise ValueError(
                "BootstrappingTransformer requires that sp is greater than"
                " the length of X"
            )
        else:
            # implement as static method
            self.block_length_ = (
                self.block_length
                if self.block_length is not None
                else min(self.sp * 2, len(X) - self.sp)
            )

        # fit boxcox to get lambda and transform X
        self.box_cox_transformer_ = BoxCoxTransformer(
            sp=self.sp, bounds=self.lambda_bounds, method=self.lambda_method
        )
        self.box_cox_transformer_.fit(X)

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

        X_transformed = self.box_cox_transformer_.transform(X)

        # fit STL on X_transformed series and extract trend, seasonal and residuals
        stl = _STL(
            X_transformed,
            period=self.sp,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            low_pass_jump=self.low_pass_jump,
        ).fit(inner_iter=self.inner_iter, outer_iter=self.outer_iter)
        seasonal = pd.Series(stl.seasonal, index=X_index)
        resid = pd.Series(stl.resid, index=X_index)
        trend = pd.Series(stl.trend, index=X_index)

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
        for i in range(self.n_series):
            new_series = self.box_cox_transformer_.inverse_transform(
                self._moving_block_bootstrap(
                    ts=resid,
                    block_length=self.block_length_,
                    replacement=self.sampling_replacement,
                )
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

    @staticmethod
    def _moving_block_bootstrap(
        ts: pd.Series, block_length: int, replacement: bool = False
    ) -> pd.Series:
        """Create a synthetic time series using the moving block bootstrap method MBB.

        Parameters
        ----------
        ts : pd.Series
            a stationary time series
        block_length : int
            The length of the bootstrapping block
        replacement: bool, optional
           Whether the sample is with or without replacement, by default True.

        Returns
        -------
        pd.Series
            synthetic time series
        """
        ts_length = len(ts)
        ts_index = ts.index
        ts_values = ts.values

        if ts_length <= block_length:
            raise ValueError(
                "X length in BootstrappingTransformer should be greater"
                " than block_length"
            )

        if block_length == 1 and not replacement:
            mbb_values = copy(ts_values)
            np.random.shuffle(mbb_values)
        elif block_length == 1:
            mbb_values = np.random.choice(
                ts_values, size=ts_length, replace=replacement
            )
        else:
            total_num_blocks = int(ts_length / block_length) + 2
            block_origns = np.random.choice(
                ts_length - block_length + 1, size=total_num_blocks, replace=replacement
            )
            mbb_values = [
                val for i in block_origns for val in ts_values[i : i + block_length]
            ]
            # remove the first few observations and ensure new series has the
            # same length as the original
            remove_first = np.random.choice(block_length - 1)
            mbb_values = mbb_values[remove_first : remove_first + ts_length]

        mbb_series = pd.Series(data=mbb_values, index=ts_index)

        return mbb_series

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
        params = [
            {},
            {"block_length": 1},
            {"series_name": "test"},
            {"return_actual": False},
            {"sampling_replacement": True},
        ]

        return params

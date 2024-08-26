# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Bootstrapping methods for time series."""

__author__ = ["ltsaprounis"]

from copy import copy
from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.boxcox import BoxCoxTransformer


class STLBootstrapTransformer(BaseTransformer):
    """Creates a population of similar time series.

    This method utilises a form of bootstrapping to generate a population of
    similar time series to the input time series [1]_, [2]_.

    First the observed time series is transformed using a Box-Cox transformation to
    stabilise the variance. Then it's decomposed to seasonal, trend and residual
    time series, using the STL implementation from statsmodels
    (``statsmodels.tsa.api.STL``) [4]_. We then sample blocks from the residuals time
    series using the Moving Block Bootstrapping (MBB) method [3]_ to create synthetic
    residuals series that mimic the autocorrelation patterns of the observed series.
    Finally these bootstrapped residuals are added to the season and trend components
    and we use the inverse Box-Cox transform to return a panel of similar time series.
    The output can be used for bagging forecasts, prediction intervals and data
    augmentation.

    The returned panel will be a multiindex dataframe (``pd.DataFrame``) with the
    series_id and time_index as the index and a single column of the time series value.
    The values for series_id are "actual" for the original series and "synthetic_n"
    (where n is an integer) for the generated series.
    See the **Examples** section for example output.

    Parameters
    ----------
    n_series : int, optional
        The number of bootstrapped time series that will be generated, by default 10.
    sp : int, optional
        Seasonal periodicity of the data in integer form, by default 12.
        Must be an integer >= 2
    block_length : int, optional
        The length of the block in the MBB method, by default None.
        If not provided, the following heuristic is used, the block length will the
        minimum between 2*sp and len(X) - sp.
    sampling_replacement : bool, optional
        Whether the MBB sample is with or without replacement, by default False.
    return_actual : bool, optional
        If True the output will contain the actual time series, by default True.
        The actual time series will be labelled as "<series_name>_actual" (or "actual"
        if series name is None).
    lambda_bounds : Tuple, optional
        BoxCox parameter:
        Lower and upper bounds used to restrict the feasible range
        when solving for the value of lambda, by default None.
    lambda_method : str, optional
        BoxCox parameter:
        {"pearsonr", "mle", "all", "guerrero"}, by default "guerrero".
        The optimization approach used to determine the lambda value used
        in the Box-Cox transformation.
    seasonal : int, optional
        STL parameter:
        Length of the seasonal smoother. Must be an odd integer, and should
        normally be >= 7, by default 7.
    trend : int, optional
        STL parameter:
        Length of the trend smoother, by default None.
        Must be an odd integer. If not provided uses the smallest odd integer greater
        than 1.5 * period / (1 - 1.5 / seasonal), following the suggestion in the
        original implementation.
    low_pass : int, optional
        STL parameter:
        Length of the low-pass filter, by default None.
        Must be an odd integer >=3. If not provided, uses the smallest odd
        integer > period
    seasonal_deg : int, optional
        STL parameter:
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend), by default 1.
    trend_deg : int, optional
        STL parameter:
        Degree of trend LOESS. 0 (constant) or 1 (constant and trend), by default 1.
    low_pass_deg : int, optional
        STL parameter:
        Degree of low pass LOESS. 0 (constant) or 1 (constant and trend), by default 1.
    robust : bool, optional
        STL parameter:
        Flag indicating whether to use a weighted version that is robust to
        some forms of outliers, by default False.
    seasonal_jump : int, optional
        STL parameter:
        Positive integer determining the linear interpolation step, by default 1.
        If larger than 1, the LOESS is used every seasonal_jump points and linear
        interpolation is between fitted points. Higher values reduce estimation time.
    trend_jump : int, optional
        STL parameter:
        Positive integer determining the linear interpolation step, by default 1.
        If larger than 1, the LOESS is used every trend_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation time.
    low_pass_jump : int, optional
        STL parameter:
        Positive integer determining the linear interpolation step, by default 1.
        If larger than 1, the LOESS is used every low_pass_jump points and values
        between the two are linearly interpolated. Higher values reduce estimation
        time.
    inner_iter : int, optional
        STL parameter:
        Number of iterations to perform in the inner loop, by default None.
        If not provided uses 2 if robust is True, or 5 if not. This param goes into
        STL.fit() from statsmodels.
    outer_iter : int, optional
        STL parameter:
        Number of iterations to perform in the outer loop, by default None.
        If not provided uses 15 if robust is True, or 0 if not.
        This param goes into STL.fit() from statsmodels.
    random_state : int, np.random.RandomState or None, by default None
        Controls the randomness of the estimator

    See Also
    --------
    sktime.transformations.bootstrap.MovingBlockBootstrapTransformer :
        Transformer that applies the Moving Block Bootstrapping method to create
        a panel of synthetic time series.

    References
    ----------
    .. [1] Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential
        smoothing methods using STL decomposition and Box-Cox transformation.
        International Journal of Forecasting, 32(2), 303-312
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3,
        Chapter 12.5. Accessed on February 13th 2022.
    .. [3] Kunsch HR (1989) The jackknife and the bootstrap for general stationary
        observations. Annals of Statistics 17(3), 1217-1241
    .. [4] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html

    Examples
    --------
    >>> from sktime.transformations.bootstrap import STLBootstrapTransformer
    >>> from sktime.datasets import load_airline
    >>> from sktime.utils.plotting import plot_series  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = STLBootstrapTransformer(10)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    >>> series_list = []  # doctest: +SKIP
    >>> names = []  # doctest: +SKIP
    >>> for group, series in y_hat.groupby(level=0, as_index=False):
    ...     series.index = series.index.droplevel(0)
    ...     series_list.append(series)
    ...     names.append(group)  # doctest: +SKIP
    >>> plot_series(*series_list, labels=names)  # doctest: +SKIP
    (...)
    >>> print(y_hat.head())  # doctest: +SKIP
                          Number of airline passengers
    series_id time_index
    actual    1949-01                            112.0
              1949-02                            118.0
              1949-03                            132.0
              1949-04                            129.0
              1949-05                            121.0
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "ltsaprounis",
        "python_dependencies": "statsmodels",
        # estimator type
        # --------------
        # todo: what is the scitype of X: Series, or Panel
        "scitype:transform-input": "Series",
        # todo: what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Panel",
        # todo: what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # X_inner_mtype can be Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(
        self,
        n_series: int = 10,
        sp: int = 12,
        block_length: int = None,
        sampling_replacement: bool = False,
        return_actual: bool = True,
        lambda_bounds: tuple = None,
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
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.n_series = n_series
        self.sp = sp
        self.block_length = block_length
        self.sampling_replacement = sampling_replacement
        self.return_actual = return_actual
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
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series
            Data to be transformed
        y : ignored, for interface compatibility

        Returns
        -------
        self: reference to self
        """
        if self.sp <= 1:
            raise NotImplementedError(
                "STLBootstrapTransformer does not support non-seasonal data"
            )

        if not isinstance(self.sp, int):
            raise ValueError(
                "sp parameter of STLBootstrapTransformer must be an integer"
            )

        if len(X) <= self.sp:
            raise ValueError(
                "STLBootstrapTransformer requires that sp is greater than"
                " the length of X"
            )

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
        X : pd.Series
            Data to be transformed
        y : ignored, for interface compatibility

        Returns
        -------
        transformed version of X
        """
        from statsmodels.tsa.api import STL as _STL

        Xcol = X.columns
        X = X[X.columns[0]]

        if len(X) <= self.block_length_:
            raise ValueError(
                "STLBootstrapTransformer requires that block_length is"
                " strictly smaller than the length of X"
            )

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
        col_name = _get_series_name(X)

        # initialize the dataframe that will store the bootstrapped series
        if self.return_actual:
            df_list = [
                pd.DataFrame(
                    X.values,
                    index=pd.MultiIndex.from_product(
                        iterables=[["actual"], X_index],
                        names=["series_id", "time_index"],
                    ),
                    columns=[col_name],
                )
            ]
        else:
            df_list = []

        # set the random state
        rng = check_random_state(self.random_state)
        # create multiple series
        for i in range(self.n_series):
            new_series = self.box_cox_transformer_.inverse_transform(
                _moving_block_bootstrap(
                    ts=resid,
                    block_length=self.block_length_,
                    replacement=self.sampling_replacement,
                    random_state=rng,
                )
                + seasonal
                + trend
            )

            new_series_id = f"synthetic_{i}"
            new_df_index = pd.MultiIndex.from_product(
                iterables=[[new_series_id], new_series.index],
                names=["series_id", "time_index"],
            )

            df_list.append(
                pd.DataFrame(
                    data=new_series.values, index=new_df_index, columns=[col_name]
                )
            )

        Xt = pd.concat(df_list)
        Xt.columns = Xcol

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [
            {"sp": 3},
            {"block_length": 1, "sp": 3},
            {"return_actual": False, "sp": 3},
            {"sampling_replacement": True, "sp": 3},
        ]

        return params


class MovingBlockBootstrapTransformer(BaseTransformer):
    """Moving Block Bootstrapping method for synthetic time series generation.

    The Moving Block Bootstrapping (MBB) method introduced in [1]_ is  can be used to
    create synthetic time series that mimic the autocorelation patterns of an observed
    stationary series. This method is frequently combined with other transformations
    e.g. BoxCox and STL to produce synthetic time series similar to the observed time
    series [2]_, [3]_.

    The returned panel will be a multiindex dataframe (``pd.DataFrame``) with the
    series_id and time_index as the index and a single column of the time series value.
    The values for series_id are "actual" for the original series and "synthetic_n"
    (where n is an integer) for the generated series.
    See the **Examples** section for example output.

    Parameters
    ----------
    n_series : int, optional
        The number of bootstrapped time series that will be generated, by default 10
    block_length : int, optional
        The length of the block in the MBB method, by default None.
        If not provided, the following heuristic is used, the block length will the
        minimum between 2*sp and len(X) - sp.
    sampling_replacement : bool, optional
        Whether the MBB sample is with or without replacement, by default False.
    return_actual : bool, optional
        If True the output will contain the actual time series, by default True.
        The actual time series will be labelled as "actual"
    random_state : int, np.random.RandomState or None, by default None
        Controls the randomness of the estimator

    See Also
    --------
    sktime.transformations.bootstrap.STLBootstrapTransformer :
        Transformer that utilises BoxCox, STL and Moving Block Bootstrapping to create
        a panel of similar time series.

    References
    ----------
    .. [1] Kunsch HR (1989) The jackknife and the bootstrap for general stationary
        observations. Annals of Statistics 17(3), 1217-1241
    .. [2] Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential
        smoothing methods using STL decomposition and Box-Cox transformation.
        International Journal of Forecasting, 32(2), 303-312
    .. [3] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3,
        Chapter 12.5. Accessed on February 13th 2022. Accessed on February 13th 2022.

    Examples
    --------
    >>> from sktime.transformations.bootstrap import MovingBlockBootstrapTransformer
    >>> from sktime.datasets import load_airline
    >>> from sktime.utils.plotting import plot_series  # doctest: +SKIP
    >>> y = load_airline()
    >>> transformer = MovingBlockBootstrapTransformer(10)
    >>> y_hat = transformer.fit_transform(y)
    >>> series_list = []
    >>> names = []
    >>> for group, series in y_hat.groupby(level=[0], as_index=False):
    ...     series.index = series.index.droplevel(0)
    ...     series_list.append(series)
    ...     names.append(group)
    >>> plot_series(*series_list, labels=names)  # doctest: +SKIP
    (...)
    >>> print(y_hat.head()) # doctest: +NORMALIZE_WHITESPACE
                          Number of airline passengers
    series_id time_index
    actual    1949-01                            112.0
              1949-02                            118.0
              1949-03                            132.0
              1949-04                            129.0
              1949-05                            121.0
    """

    _tags = {
        # todo: what is the scitype of X: Series, or Panel
        "scitype:transform-input": "Series",
        # todo: what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Panel",
        # todo: what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # X_inner_mtype can be Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(
        self,
        n_series: int = 10,
        block_length: int = 10,
        sampling_replacement: bool = False,
        return_actual: bool = True,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.n_series = n_series
        self.block_length = block_length
        self.sampling_replacement = sampling_replacement
        self.return_actual = return_actual
        self.random_state = random_state

        super().__init__()

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
        Xcol = X.columns
        X = X[X.columns[0]]

        if len(X) <= self.block_length:
            raise ValueError(
                "MovingBlockBootstrapTransformer requires that block_length is"
                " greater than the length of X"
            )

        X_index = X.index

        # time series name
        col_name = _get_series_name(X)

        # initialize the dataframe that will store the bootstrapped series
        if self.return_actual:
            df_list = [
                pd.DataFrame(
                    X.values,
                    index=pd.MultiIndex.from_product(
                        iterables=[["actual"], X_index],
                        names=["series_id", "time_index"],
                    ),
                    columns=[col_name],
                )
            ]
        else:
            df_list = []

        # set the random state
        rng = check_random_state(self.random_state)
        # create multiple series
        for i in range(self.n_series):
            new_series = _moving_block_bootstrap(
                ts=X,
                block_length=self.block_length,
                replacement=self.sampling_replacement,
                random_state=rng,
            )

            new_series_id = f"synthetic_{i}"
            new_df_index = pd.MultiIndex.from_product(
                iterables=[[new_series_id], new_series.index],
                names=["series_id", "time_index"],
            )
            df_list.append(
                pd.DataFrame(
                    data=new_series.values, index=new_df_index, columns=[col_name]
                )
            )

        Xt = pd.concat(df_list)
        Xt.columns = Xcol

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [
            {"block_length": 5},
            {"block_length": 1},
            {"block_length": 5, "return_actual": False},
            {"block_length": 5, "sampling_replacement": True},
        ]

        return params


def _moving_block_bootstrap(
    ts: pd.Series,
    block_length: int,
    replacement: bool = False,
    random_state: Union[int, np.random.RandomState] = None,
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
    random_state : int, np.random.RandomState or None, by default None
        Controls the randomness of the estimator

    Returns
    -------
    pd.Series
        synthetic time series
    """
    ts_length = len(ts)
    ts_index = ts.index
    ts_values = ts.values
    rng = check_random_state(random_state)

    if ts_length <= block_length:
        raise ValueError(
            "X length in moving block bootstrapping should be greater"
            " than block_length"
        )

    if block_length == 1 and not replacement:
        mbb_values = copy(ts_values)
        rng.shuffle(mbb_values)
    elif block_length == 1:
        mbb_values = rng.choice(ts_values, size=ts_length, replace=replacement)
    else:
        total_num_blocks = int(ts_length / block_length) + 2
        block_origns = rng.choice(
            ts_length - block_length + 1, size=total_num_blocks, replace=replacement
        )
        mbb_values = [
            val for i in block_origns for val in ts_values[i : i + block_length]
        ]
        # remove the first few observations and ensure new series has the
        # same length as the original
        remove_first = rng.choice(block_length - 1)
        mbb_values = mbb_values[remove_first : remove_first + ts_length]

    mbb_series = pd.Series(data=mbb_values, index=ts_index)

    return mbb_series


def _get_series_name(ts: Union[pd.Series, pd.DataFrame]) -> str:
    """Get series name from Series or column name from DataFrame.

    Parameters
    ----------
    ts : Union[pd.Series, pd.DataFrame]
        input series / dataframe

    Returns
    -------
    str
        series name or column name
    """
    if isinstance(ts, pd.Series):
        return ts.name
    elif isinstance(ts, pd.DataFrame):
        return ts.columns.values[0]

#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Transformer to impute missing values in series."""

__author__ = ["aiwalter"]
__all__ = ["Imputer"]

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.base import BaseTransformer


class Imputer(BaseTransformer):
    """Missing value imputation.

    The Imputer transforms input series by replacing missing values according
    to an imputation strategy specified by `method`.

    Parameters
    ----------
    method : str, default="drift"
        Method to fill the missing values.

        * "drift" : drift/trend values by sktime.PolynomialTrendForecaster(degree=1)
            first, X in transform() is filled with ffill then bfill
            then PolynomialTrendForecaster(degree=1) is fitted to filled X, and
            predict values are queried at indices which had missing values
        * "linear" : linear interpolation, uses pd.Series.interpolate()
            WARNING: This method can not extrapolate, so it is fitted always on the
            data given to transform().
        * "nearest" : use nearest value, uses pd.Series.interpolate()
        * "constant" : same constant value (given in arg value) for all NaN
        * "mean" : pd.Series.mean() of *fit* data
        * "median" : pd.Series.median() of *fit* data
        * "backfill" ot "bfill" : adapted from pd.Series.fillna()
        * "pad" or "ffill" : adapted from pd.Series.fillna()
        * "random" : random values between pd.Series.min() and .max() of *fit* data
            if pd.Series dtype is int, sample is uniform discrete
            if pd.Series dtype is float, sample is uniform continuous
        * "forecaster" : use an sktime Forecaster, given in param forecaster.
            First, X in *fit* is filled with ffill then bfill
            then forecaster is fitted to filled X, and *predict* values are queried
            at indices of X data in *transform* which had missing values
        For the following methods, the train data is used to fit them:
        "drift", "mean", "median", "random". For all other methods, the
        transform data is sufficient to compute the impute values.

    missing_values : int/float/str, default=None
        The placeholder for the missing values. All occurrences of
        missing_values will be imputed, in addition to np.nan.
        If None, then only np.nan values are imputed.
    value : int/float, default=None
        Value to use to fill missing values when method="constant".
    forecaster : Any Forecaster based on sktime.BaseForecaster, default=None
        Use a given Forecaster to impute by insample predictions when
        method="forecaster". Before fitting, missing data is imputed with
        method="ffill" or "bfill" as heuristic. in case of multivariate X,
        the forecaster is applied separete to each column like a
        ColumnEnsembleForecaster.
    random_state : int/float/str, optional
        Value to set random.seed() if method="random", default None

    Examples
    --------
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> transformer = Imputer(method="drift")
    >>> transformer.fit(y_train)
    Imputer(...)
    >>> y_test.iloc[3] = np.nan
    >>> y_hat = transformer.transform(y_test)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,
        "handles-missing-data": True,
        "skip-inverse-transform": True,
        "capability:inverse_transform": True,
        "univariate-only": False,
        "capability:missing_values:removes": True,
        # is transform result always guaranteed to contain no missing values?
        "remember_data": False,  # remember all data seen as _X
    }

    def __init__(
        self,
        method="drift",
        random_state=None,
        value=None,
        forecaster=None,
        missing_values=None,
    ):
        self.method = method
        self.missing_values = missing_values
        self.value = value
        self.forecaster = forecaster
        self.random_state = random_state
        super().__init__()

        # these methods require self._X remembered in _fit and _update
        if method in ["drift", "forecaster", "random"]:
            self.set_tags(**{"remember_data": True})

        # these methods can be applied to multi-index frames without vectorization or
        # by using an efficient pandas native method
        if method in [
            "constant",
            "mean",
            "median",
            "backfill",
            "bfill",
            "pad",
            "ffill",
        ]:
            self.set_tags(
                **{
                    "X_inner_mtype": [
                        "pd.DataFrame",
                        "pd-multiindex",
                        "pd_multiindex_hier",
                    ]
                }
            )

        if method in "forecaster":
            self.set_tags(**{"y_inner_mtype": ["pd.DataFrame"]})

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
        self._check_method()
        # all methods of Imputer that are actually doing a fit are
        # implemented here. Some methods don't need to fit, so they are just
        # implemented in _transform

        index = X.index
        if isinstance(index, pd.MultiIndex):
            X_grouped = X.groupby(level=list(range(index.nlevels - 1)))
            if self.method == "mean":
                self._mean = X_grouped.mean()
            elif self.method == "median":
                self._median = X_grouped.median()
        else:
            if self.method in ["drift", "forecaster"]:
                self._y = y.copy() if y is not None else None
                if self.method == "drift":
                    self._forecaster = PolynomialTrendForecaster(degree=1)
                elif self.method == "forecaster":
                    self._forecaster = self.forecaster.clone()
            elif self.method == "mean":
                self._mean = X.mean()
            elif self.method == "median":
                self._median = X.median()

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
        X : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        X = X.copy()

        # replace missing_values with np.nan
        if self.missing_values:
            X = X.replace(to_replace=self.missing_values, value=np.nan)

        if not _has_missing_values(X):
            return X

        index = X.index

        if self.method == "random":
            for col in X.columns:
                isna = X[col].isna()
                X.loc[isna, col] = self._create_random_distribution(X[col])(isna.sum())
            return X
        elif self.method == "constant":
            return X.fillna(value=self.value)
        elif isinstance(index, pd.MultiIndex):
            X_grouped = X.groupby(level=list(range(index.nlevels - 1)))

            if self.method in ["backfill", "bfill"]:
                X = X_grouped.fillna(method="bfill")
                # fill trailing NAs of panel instances with reverse method
                return X.fillna(method="ffill")
            elif self.method in ["pad", "ffill"]:
                X = X_grouped.fillna(method="ffill")
                # fill leading NAs of panel instances with reverse method
                return X.fillna(method="bfill")
            elif self.method == "mean":
                return X_grouped.fillna(value=self._mean)
            elif self.method == "median":
                return X_grouped.fillna(value=self._median)
            else:
                raise AssertionError("Code should not be reached")
        else:
            if self.method in ["backfill", "bfill", "pad", "ffill"]:
                X = X.fillna(method=self.method)
            elif self.method == "drift":
                X = self._impute_with_forecaster(X, y)
            elif self.method == "forecaster":
                X = self._impute_with_forecaster(X, y)
            elif self.method == "mean":
                return X.fillna(value=self._mean)
            elif self.method == "median":
                return X.fillna(value=self._median)
            elif self.method in ["nearest", "linear"]:
                X = X.interpolate(method=self.method)
            else:
                raise ValueError(f"`method`: {self.method} not available.")

            # fill first/last elements of series,
            # as some methods (e.g. "linear") can't impute those
            X = X.fillna(method="ffill").fillna(method="backfill")

            return X

    def _check_method(self):
        method = self.method
        if method not in [
            "mean",
            "drift",
            "linear",
            "nearest",
            "constant",
            "median",
            "backfill",
            "bfill",
            "pad",
            "ffill",
            "random",
            "forecaster",
        ]:
            raise ValueError(f"Given method {method} is not an allowed method.")
        if (
            self.value is not None
            and method != "constant"
            or method == "constant"
            and self.value is None
        ):
            raise ValueError(
                """Imputing with a value can only be
                used if method="constant" and if parameter "value" is not None"""
            )
        elif (
            self.forecaster is not None
            and method != "forecaster"
            or method == "forecaster"
            and self.forecaster is None
        ):
            raise ValueError(
                """Imputing with a forecaster can only be used if
                method=\"forecaster\" and if arg forecaster is not None"""
            )
        else:
            pass

    def _create_random_distribution(self, z: pd.Series):
        """Create uniform distribution function within boundaries of given series.

        The distribution is discrete, if the series contains only int-like values.

        Parameters
        ----------
        z : pd.Series
            A series to create a random distribution from

        Returns
        -------
        Callable[[Optional[int]], float]
            Random (discrete) uniform distribution between min and max of series
        """
        rng = check_random_state(self.random_state)
        if (z.dropna() % 1 == 0).all():
            return lambda size, low=z.min(), high=z.max(): rng.randint(
                low=low, high=high, size=size
            )
        else:
            return lambda size, low=z.min(), high=z.max(): rng.uniform(
                low=low, high=high, size=size
            )

    def _impute_with_forecaster(self, X, y):
        """Use a given forecaster for imputation by in-sample predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Series to impute.
        y : pd.DataFrame
            Exog data for forecaster.

        Returns
        -------
        Xt : pd.DataFrame
            Series with imputed values.
        """
        for col in X.columns:
            if _has_missing_values(X[col]):
                # define fh based on index of missing values
                na_index = X[col].index[X[col].isna()]
                fh = ForecastingHorizon(values=na_index, is_relative=False)

                # fill NaN before fitting with ffill and backfill (heuristic)

                self._forecaster.fit(
                    y=self._X[col].fillna(method="ffill").fillna(method="backfill"),
                    X=self._y[col].fillna(method="ffill").fillna(method="backfill")
                    if self._y is not None
                    else None,
                    fh=fh,
                )

                # replace missing values with predicted values
                X[col][na_index] = self._forecaster.predict(fh=fh, X=y)
        return X

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.trend import TrendForecaster

        linear_forecaster = make_reduction(LinearRegression(), strategy="multioutput")

        return [
            {"method": "drift"},
            {"method": "linear"},
            {"method": "nearest"},
            {"method": "constant", "value": 1},
            {"method": "median"},
            {"method": "backfill"},
            {"method": "bfill"},
            {"method": "pad"},
            {"method": "random"},
            {"method": "forecaster", "forecaster": TrendForecaster()},
            {"method": "forecaster", "forecaster": linear_forecaster},
        ]


def _has_missing_values(X):
    return X.isnull().to_numpy().any()

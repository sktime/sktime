# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements X-13ARIMA-SEATS seasonal adjustment."""

__author__ = ["Mxdhaa"]
__all__ = ["X13ArimaSeats"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class X13ArimaSeats(BaseTransformer):
    """X-13ARIMA-SEATS seasonal adjustment transformer.

    Interfaces the U.S. Census Bureau's X-13ARIMA-SEATS seasonal adjustment
    program via statsmodels.

    Parameters
    ----------
    maxorder : tuple, default=(2, 1, 2)
        Maximum order of the regular and seasonal ARMA polynomials.
    maxdiff : tuple, default=(2, 1)
        Maximum orders for regular and seasonal differencing.
    diff : tuple or None, default=None
        Orders of differencing to be used.
    log : bool or None, default=None
        Whether to take log of the series.
    outlier : bool, default=True
        Whether to test for and correct detected outliers.
    trading : bool, default=False
        Whether to test for trading day effects.
    forecast_years : int or None, default=None
        Number of years to forecast.
    x12path : str or None, default=None
        Path to the X-13/X-12 binary. If None, uses the X13PATH environment variable.
    prefer_x13 : bool, default=True
        Whether to prefer X-13 over X-12.
    return_components : bool, default=False
        If True, returns a DataFrame containing the components:
        'seasadj' (seasonally adjusted), 'trend', 'seasonal', and 'irregular'.
        If False, returns only the seasonally adjusted series.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Mxdhaa"],
        "python_dependencies": "statsmodels",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "transform-returns-same-time-index": True,
        "capability:multivariate": False,
        "fit_is_empty": False,
        "python_dependencies": "statsmodels",
        "capability:inverse_transform": True,
        "capability:inverse_transform:exact": False,
        "skip-inverse-transform": False,
        "capability:categorical_in_X": False,
        # CI and test flags
        # -----------------
        "tests:skip_all": True,
    }

    def __init__(
        self,
        maxorder=(2, 1, 2),
        maxdiff=(2, 1),
        diff=None,
        log=None,
        outlier=True,
        trading=False,
        forecast_years=None,
        x12path=None,
        prefer_x13=True,
        return_components=False,
    ):
        self.maxorder = maxorder
        self.maxdiff = maxdiff
        self.diff = diff
        self.log = log
        self.outlier = outlier
        self.trading = trading
        self.forecast_years = forecast_years
        self.x12path = x12path
        self.prefer_x13 = prefer_x13
        self.return_components = return_components
        self._X = None
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y."""
        from statsmodels.tsa.x13 import x13_arima_analysis

        self._X = X

        try:
            self.results_ = x13_arima_analysis(
                X,
                maxorder=self.maxorder,
                maxdiff=self.maxdiff,
                diff=self.diff,
                log=self.log,
                outlier=self.outlier,
                trading=self.trading,
                forecast_years=self.forecast_years,
                x12path=self.x12path,
                prefer_x13=self.prefer_x13,
            )
        except Exception as e:
            if (
                "x12a" in str(e)
                or "x13as" in str(e)
                or "executable" in str(e).lower()
                or "find" in str(e).lower()
            ):
                raise FileNotFoundError(
                    "X-13ARIMA-SEATS executable not found. Please download the X-13 binary "
                    "from the U.S. Census Bureau website and set the X13PATH environment "
                    "variable, or pass the absolute path to the x12path parameter of the transformer."
                ) from e
            raise e

        # Store component series
        self.seasadj_ = self.results_.seasadj
        self.trend_ = self.results_.trend
        self.seasonal_ = self.results_.seasonal
        self.irregular_ = self.results_.irregular

        return self

    def _transform(self, X, y=None):
        """Transform X."""
        # If transforming the fitted data, return cached results
        if X.index.equals(self._X.index):
            return self._make_return_object(X)

        # Otherwise fit and transform on new data
        from statsmodels.tsa.x13 import x13_arima_analysis

        try:
            results = x13_arima_analysis(
                X,
                maxorder=self.maxorder,
                maxdiff=self.maxdiff,
                diff=self.diff,
                log=self.log,
                outlier=self.outlier,
                trading=self.trading,
                forecast_years=self.forecast_years,
                x12path=self.x12path,
                prefer_x13=self.prefer_x13,
            )
        except Exception as e:
            if (
                "x12a" in str(e)
                or "x13as" in str(e)
                or "executable" in str(e).lower()
                or "find" in str(e).lower()
            ):
                raise FileNotFoundError(
                    "X-13ARIMA-SEATS executable not found. Please download the X-13 binary "
                    "from the U.S. Census Bureau website and set the X13PATH environment "
                    "variable, or pass the absolute path to the x12path parameter of the transformer."
                ) from e
            raise e

        return self._make_return_object(X, results)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X."""
        if self.return_components:
            # Reconstruct original series.
            # If log transform was used, components are multiplicative: Original = seasadj * seasonal
            # Otherwise, components are additive: Original = seasadj + seasonal.
            if self.log is True:
                ret = X["seasadj"] * X["seasonal"]
            else:
                ret = X["seasadj"] + X["seasonal"]
        else:
            # If return_components=False, we only have seasonally adjusted series.
            # To reverse it, we add/multiply the fitted seasonal component.
            if X.index.equals(self._X.index):
                seasonal = self.seasonal_
            else:
                raise ValueError(
                    "Inverse transform of seasonally adjusted series is only supported "
                    "on the index seen in fit."
                )
            if self.log is True:
                ret = X * seasonal
            else:
                ret = X + seasonal

        if hasattr(self, "_X") and self._X is not None:
            ret.name = self._X.name
        return ret

    def _make_return_object(self, X, results=None):
        if results is None:
            seasadj = self.seasadj_
            trend = self.trend_
            seasonal = self.seasonal_
            irregular = self.irregular_
        else:
            seasadj = results.seasadj
            trend = results.trend
            seasonal = results.seasonal
            irregular = results.irregular

        if self.return_components:
            return pd.DataFrame(
                {
                    "seasadj": seasadj,
                    "trend": trend,
                    "seasonal": seasonal,
                    "irregular": irregular,
                },
                index=X.index,
            )
        else:
            ret = pd.Series(seasadj, name=X.name, index=X.index)
            return ret

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"outlier": False},
            {"return_components": True},
        ]

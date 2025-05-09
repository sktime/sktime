# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements STLForecaster based on statsmodels."""

__author__ = ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"]
__all__ = ["STLForecaster"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies


class STLForecaster(BaseForecaster):
    """Implements STLForecaster based on statsmodels.tsa.seasonal.STL implementation.

    The STLForecaster applies the following algorithm, also see [1]_.

    In ``fit``:

    1. Use ``statsmodels`` ``STL`` [2]_ to decompose the given series ``y`` into
       the three components: ``trend``, ``season`` and ``residuals``.
    2. Fit clones of ``forecaster_trend`` to ``trend``, ``forecaster_seasonal`` to
    ``season``,
       and ``forecaster_resid`` to ``residuals``, using ``y``, ``X``, ``fh`` from
       ``fit``.
       The forecasters are fitted as clones, stored in the attributes
       ``forecaster_trend_``, ``forecaster_seasonal_``, ``forecaster_resid_``.

    In ``predict``, forecasts as follows:

    1. Obtain forecasts ``y_pred_trend`` from ``forecaster_trend_``,
       ``y_pred_seasonal`` from ``forecaster_seasonal_``, and
       ``y_pred_residual`` from ``forecaster_resid_``, using ``X``, ``fh``, from
       ``predict``.
    2. Recompose ``y_pred`` as ``y_pred = y_pred_trend + y_pred_seasonal +
    y_pred_residual``
    3. Return ``y_pred``

        ``update`` refits entirely, i.e., behaves as ``fit`` on all data seen so far.

    Parameters
    ----------
    sp : int, optional, default=2. Passed to ``statsmodels`` ``STL``.
        Length of the seasonal period passed to ``statsmodels`` ``STL``.
        (forecaster_seasonal, forecaster_resid) that are None. The
        default forecaster_trend does not get sp as trend is independent
        to seasonality.
    seasonal : int, optional., default=7. Passed to ``statsmodels`` ``STL``.
        Length of the seasonal smoother. Must be an odd integer >=3, and should
        normally be >= 7 (default).
    trend : {int, None}, optional, default=None. Passed to ``statsmodels`` ``STL``.
        Length of the trend smoother. Must be an odd integer. If not provided
        uses the smallest odd integer greater than
        1.5 * period / (1 - 1.5 / seasonal), following the suggestion in
        the original implementation.
    low_pass : {int, None}, optional, default=None. Passed to ``statsmodels`` ``STL``.
        Length of the low-pass filter. Must be an odd integer >=3. If not
        provided, uses the smallest odd integer > period.
    seasonal_deg : int, optional, default=1. Passed to ``statsmodels`` ``STL``.
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
    trend_deg : int, optional, default=1. Passed to ``statsmodels`` ``STL``.
        Degree of trend LOESS. 0 (constant) or 1 (constant and trend).
    low_pass_deg : int, optional, default=1. Passed to ``statsmodels`` ``STL``.
        Degree of low pass LOESS. 0 (constant) or 1 (constant and trend).
    robust : bool, optional, default=False. Passed to ``statsmodels`` ``STL``.
        Flag indicating whether to use a weighted version that is robust to
        some forms of outliers.
    seasonal_jump : int, optional, default=1. Passed to ``statsmodels`` ``STL``.
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every seasonal_jump points and linear
        interpolation is between fitted points. Higher values reduce
        estimation time.
    trend_jump : int, optional, default=1. Passed to ``statsmodels`` ``STL``.
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every trend_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.
    low_pass_jump : int, optional, default=1. Passed to ``statsmodels`` ``STL``.
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every low_pass_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.
    inner_iter: int or None, optional, default=None. Passed to ``statsmodels`` ``STL``.
        Number of iterations to perform in the inner loop. If not provided uses 2 if
        robust is True, or 5 if not. This param goes into STL.fit() from statsmodels.
    outer_iter: int or None, optional, default=None. Passed to ``statsmodels`` ``STL``.
        Number of iterations to perform in the outer loop. If not provided uses 15 if
        robust is True, or 0 if not. This param goes into STL.fit() from statsmodels.
    forecaster_trend : sktime forecaster, optional
        Forecaster to be fitted on trend_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="drift") is used.
    forecaster_seasonal : sktime forecaster, optional
        Forecaster to be fitted on seasonal_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="last") is used.
    forecaster_resid : sktime forecaster, optional
        Forecaster to be fitted on resid_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="mean") is used.

    Attributes
    ----------
    trend_ : pd.Series
        Trend component.
    seasonal_ : pd.Series
        Seasonal component.
    resid_ : pd.Series
        Residuals component.
    forecaster_trend_ : sktime forecaster
        Fitted trend forecaster.
    forecaster_seasonal_ : sktime forecaster
        Fitted seasonal forecaster.
    forecaster_resid_ : sktime forecaster
        Fitted residual forecaster.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import STLForecaster
    >>> y = load_airline()
    >>> forecaster = STLForecaster(sp=12)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    STLForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP

    See Also
    --------
    Deseasonalizer
    Detrender

    References
    ----------
    .. [1] R. B. Cleveland, W. S. Cleveland, J.E. McRae, and I. Terpenning (1990)
       STL: A Seasonal-Trend Decomposition Procedure Based on LOESS.
       Journal of Official Statistics, 6, 3-73.
    .. [2] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
    """

    _tags = {
        "authors": ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly", "ericjb"],
        "maintainers": ["tensorflow-as-tf"],
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        sp=2,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        robust=False,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
        inner_iter=None,
        outer_iter=None,
        forecaster_trend=None,
        forecaster_seasonal=None,
        forecaster_resid=None,
    ):
        self.sp = sp
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
        self.forecaster_trend = forecaster_trend
        self.forecaster_seasonal = forecaster_seasonal
        self.forecaster_resid = forecaster_resid
        super().__init__()

        for forecaster in (
            self.forecaster_trend,
            self.forecaster_seasonal,
            self.forecaster_resid,
        ):
            if forecaster is not None and not forecaster.get_tag(
                "ignores-exogeneous-X"
            ):
                ignore_exogenous = False
                break
        else:  # none of the forecasters (if provided) use exogenous feature variables
            ignore_exogenous = True  # corresponding to NaiveForecaster in missing case

        self.set_tags(**{"ignores-exogeneous-X": ignore_exogenous})

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """
        from statsmodels.tsa.seasonal import STL as _STL

        from sktime.forecasting.naive import NaiveForecaster

        self._stl = _STL(
            y.values,
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

        self.seasonal_ = pd.Series(self._stl.seasonal, index=y.index)
        self.resid_ = pd.Series(self._stl.resid, index=y.index)
        self.trend_ = pd.Series(self._stl.trend, index=y.index)

        self.forecaster_seasonal_ = (
            NaiveForecaster(sp=self.sp, strategy="last")
            if self.forecaster_seasonal is None
            else self.forecaster_seasonal.clone()
        )
        # trend forecaster does not need sp
        self.forecaster_trend_ = (
            NaiveForecaster(strategy="drift")
            if self.forecaster_trend is None
            else self.forecaster_trend.clone()
        )
        self.forecaster_resid_ = (
            NaiveForecaster(sp=self.sp, strategy="mean")
            if self.forecaster_resid is None
            else self.forecaster_resid.clone()
        )

        # fitting forecasters to different components
        self.forecaster_seasonal_.fit(y=self.seasonal_, X=X, fh=fh)
        self.forecaster_trend_.fit(y=self.trend_, X=X, fh=fh)
        self.forecaster_resid_.fit(y=self.resid_, X=X, fh=fh)

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
                Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_pred_seasonal = self.forecaster_seasonal_.predict(fh=fh, X=X)
        y_pred_trend = self.forecaster_trend_.predict(fh=fh, X=X)
        y_pred_resid = self.forecaster_resid_.predict(fh=fh, X=X)
        y_pred = y_pred_seasonal + y_pred_trend + y_pred_resid
        y_pred.name = self._y.name
        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.array
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.seasonal import STL as _STL

        self._stl = _STL(
            y.values,
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

        self.seasonal_ = pd.Series(self._stl.seasonal, index=y.index)
        self.resid_ = pd.Series(self._stl.resid, index=y.index)
        self.trend_ = pd.Series(self._stl.trend, index=y.index)

        self.forecaster_seasonal_.update(
            y=self.seasonal_, X=X, update_params=update_params
        )
        self.forecaster_trend_.update(y=self.trend_, X=X, update_params=update_params)
        self.forecaster_resid_.update(y=self.resid_, X=X, update_params=update_params)
        return self

    def plot_components(self, title=None):
        """Plot the observed, trend, seasonal, and residual components.

        Requires state to be "fitted", i.e., ``self.is_fitted=True``.
        """
        _check_soft_dependencies(
            ["matplotlib", "seaborn"], obj="STLForecaster.plot_components"
        )
        import matplotlib.pyplot as plt

        from sktime.utils.plotting import plot_series

        self.check_is_fitted()

        fig, ax = plt.subplots(4, 1, sharex=True)

        plot_series(self._y, ax=ax[0], markers=[""])
        plot_series(self.trend_, ax=ax[1], markers=[""])
        plot_series(self.seasonal_, ax=ax[2], markers=[""])
        plot_series(self.resid_, ax=ax[3])
        # Get the lines from the 4th plot and remove them (or at least make them
        # invisible, while keeping the markers)
        for line in ax[3].lines:
            line.set_linestyle("None")
        ax[3].axhline(0, color="black", linestyle="-")
        ax[0].text(
            1.02, 0.5, "Obs", transform=ax[0].transAxes, va="center", rotation=-90
        )
        ax[1].text(
            1.02, 0.5, "Trend", transform=ax[1].transAxes, va="center", rotation=-90
        )
        ax[2].text(
            1.02, 0.5, "Season", transform=ax[2].transAxes, va="center", rotation=-90
        )
        ax[3].text(
            1.02, 0.5, "Resid", transform=ax[3].transAxes, va="center", rotation=-90
        )

        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        return fig, ax

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
        from sktime.forecasting.naive import NaiveForecaster

        params_list = [
            {},
            {
                "sp": 3,
                "seasonal": 7,
                "trend": 5,
                "seasonal_deg": 2,
                "trend_deg": 2,
                "robust": True,
                "seasonal_jump": 2,
                "trend_jump": 2,
                "low_pass_jump": 2,
                "forecaster_trend": NaiveForecaster(strategy="drift"),
                "forecaster_seasonal": NaiveForecaster(sp=3),
                "forecaster_resid": NaiveForecaster(strategy="mean"),
            },
        ]

        return params_list

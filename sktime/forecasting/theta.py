"""Theta forecasters."""

__all__ = ["ThetaForecaster", "ThetaModularForecaster"]
__author__ = ["big-o", "mloning", "kejsitake", "fkiraly", "GuzalBulatova"]

import numpy as np
import pandas as pd
from scipy.stats import norm

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.compose._ensemble import _aggregate
from sktime.forecasting.compose._pipeline import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.theta import ThetaLinesTransformer
from sktime.utils.dependencies import _check_estimator_deps
from sktime.utils.slope_and_trend import _fit_trend
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.warnings import warn


class ThetaForecaster(ExponentialSmoothing):
    """Theta method for forecasting.

    The theta method as defined in [1]_ is equivalent to simple exponential
    smoothing (SES) with drift (as demonstrated in [2]_).

    The series is tested for seasonality using the test outlined in A&N. If
    deemed seasonal, the series is seasonally adjusted using a classical
    multiplicative decomposition before applying the theta method. The
    resulting forecasts are then reseasonalised.

    In cases where SES results in a constant forecast, the theta forecaster
    will revert to predicting the SES constant plus a linear trend derived
    from the training data.

    Prediction intervals are computed using the underlying state space model.

    Parameters
    ----------
    initial_level : float, optional
        The alpha value of the simple exponential smoothing, if the value is
        set then this will be used, otherwise it will be estimated from the data.
    deseasonalize : bool, optional (default=True)
        If True, data is seasonally adjusted.
    sp : int, optional (default=1)
        The number of observations that constitute a seasonal period for a
        multiplicative deseasonaliser, which is used if seasonality is detected in the
        training data. Ignored if a deseasonaliser transformer is provided.
        Default is 1 (no seasonality).

    Attributes
    ----------
    initial_level_ : float
        The estimated alpha value of the SES fit.
    drift_ : float
        The estimated drift of the fitted model.
    se_ : float
        The standard error of the predictions. Used to calculate prediction
        intervals.

    References
    ----------
    .. [1] Assimakopoulos, V. and Nikolopoulos, K. The theta model: a
       decomposition approach to forecasting. International Journal of
       Forecasting 16, 521-530, 2000.
       https://www.sciencedirect.com/science/article/pii/S0169207000000662

    .. [2] `Hyndman, Rob J., and Billah, Baki. Unmasking the Theta method.
       International J. Forecasting, 19, 287-290, 2003.
       https://www.sciencedirect.com/science/article/pii/S0169207001001431

    See Also
    --------
    StatsForecastAutoTheta

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> y = load_airline()
    >>> forecaster = ThetaForecaster(sp=12)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    ThetaForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    _fitted_param_names = ("initial_level", "smoothing_level")
    _tags = {
        # packaging info
        # --------------
        "authors": ["big-o", "mloning", "kejsitake", "fkiraly", "GuzalBulatova"],
        "scitype:y": "univariate",
        # "python_dependencies": "statsmodels" - inherited from _StatsModelsAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, initial_level=None, deseasonalize=True, sp=1):
        self.sp = sp
        self.deseasonalize = deseasonalize
        self.deseasonalizer_ = None
        self.trend_ = None
        self.initial_level_ = None
        self.drift_ = None
        self.se_ = None
        super().__init__(initial_level=initial_level, sp=sp)

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        sp = check_sp(self.sp)
        if sp > 1 and not self.deseasonalize:
            warn(
                "`sp` in ThetaForecaster is ignored when `deseasonalise`=False",
                obj=self,
            )

        if self.deseasonalize:
            self.deseasonalizer_ = Deseasonalizer(sp=self.sp, model="multiplicative")
            y = self.deseasonalizer_.fit_transform(y)

        self.initialization_method = "known" if self.initial_level else "estimated"
        # fit exponential smoothing forecaster
        # find theta lines: Theta lines are just SES + drift
        super()._fit(y, X=None, fh=fh)
        self.initial_level_ = self._fitted_forecaster.params["smoothing_level"]

        # compute and store historical residual standard error
        self.sigma_ = np.sqrt(self._fitted_forecaster.sse / (len(y) - 1))

        # compute trend
        self.trend_ = self._compute_trend(y)

        return self

    def _predict(self, fh, X):
        """Make forecasts.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        y_pred = super()._predict(fh, X)

        # Add drift.
        drift = self._compute_drift()
        y_pred += drift

        if self.deseasonalize:
            y_pred = self.deseasonalizer_.inverse_transform(y_pred)

        return y_pred

    @staticmethod
    def _compute_trend(y):
        # Trend calculated through least squares regression.
        coefs = _fit_trend(y.values.reshape(1, -1), order=1)
        return coefs[0, 0] / 2

    def _compute_drift(self):
        fh = self.fh.to_relative(self.cutoff)
        if np.isclose(self.initial_level_, 0.0):
            # SES was constant, so revert to simple trend
            drift = self.trend_ * fh
        else:
            # Calculate drift from SES parameters
            n_timepoints = len(self._y)
            drift = self.trend_ * (
                fh
                + (1 - (1 - self.initial_level_) ** n_timepoints) / self.initial_level_
            )

        return drift

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        pred_int = BaseForecaster._predict_interval(self, fh=fh, X=X, coverage=coverage)
        return pred_int

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and predict_interval

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # prepare return data frame
        var_names = self._get_varnames()
        var_name = var_names[0]
        index = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(columns=index)

        sem = self.sigma_ * np.sqrt(
            self.fh.to_relative(self.cutoff) * self.initial_level_**2 + 1
        )

        y_pred = self._predict(fh, X)

        # we assume normal additive noise with sem variance
        for a in alpha:
            pred_quantiles[(var_name, a)] = y_pred + norm.ppf(a) * sem
        # todo: should this not increase with the horizon?
        # i.e., sth like norm.ppf(a) * sem * fh.to_absolute(cutoff) ?
        # I've just refactored this so will leave it for now

        return pred_quantiles

    def _update(self, y, X=None, update_params=True):
        super()._update(y, X, update_params=False)  # use custom update_params routine
        if update_params:
            if self.deseasonalize:
                y = self.deseasonalizer_.transform(self._y)  # use updated y
            self.initial_level_ = self._fitted_forecaster.params["smoothing_level"]
            self.trend_ = self._compute_trend(y)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str , default = "default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params :dict or list of dict , default = {}
            parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in `params
        """
        params0 = {}
        params1 = {"sp": 2, "deseasonalize": True}
        params2 = {"deseasonalize": False}
        params3 = {"initial_level": 0.5}

        return [params0, params1, params2, params3]


def _zscore(level: float, two_tailed: bool = True) -> float:
    """Calculate a z-score from a confidence level.

    Parameters
    ----------
    level : float
        A confidence level, in the open interval (0, 1).
    two_tailed : bool (default=True)
        If True, return the two-tailed z score.

    Returns
    -------
    z : float
        The z score.
    """
    alpha = 1 - level
    if two_tailed:
        alpha /= 2
    return -norm.ppf(alpha)


class ThetaModularForecaster(BaseForecaster):
    """Modular theta method for forecasting.

    Modularized implementation of Theta method as defined in [1]_.
    Also see auto-theta method as described in [2]_ *not contained in this estimator).

    Overview: Input :term:`univariate series <Univariate time series>` of length
    "n" and decompose with :class:`ThetaLinesTransformer
    <sktime.transformations.series.theta>` by modifying the local curvature of
    the time series using Theta-coefficient values - ``theta_values`` parameter.
    Thansformation gives a pd.DataFrame of shape ``len(input series) * len(theta)``.

    The resulting transformed series (Theta-lines) are extrapolated separately.
    The forecasts are then aggregated into one prediction - aunivariate series,
    of ``len(fh)``.

    Parameters
    ----------
    forecasters: list of tuples (str, estimator, int or pd.index), default=None
        Forecasters to apply to each Theta-line based on the third element
        (the index). Indices must correspond to the theta_values, see Examples.
        If None, will apply PolynomialTrendForecaster (linear regression) to the
        Theta-lines where theta_value equals 0, and ExponentialSmoothing - where
        theta_value is different from 0.
    theta_values: sequence of float, default=(0,2)
        Theta-coefficients to use in transformation. If ``forecasters`` parameter
        is passed, must be the same length as ``forecasters``.
    aggfunc: str, default="mean"
        Must be one of ["mean", "median", "min", "max", "gmean"].
        Calls :func:`_aggregate` of
        :class:`EnsembleForecaster<sktime.forecasting.compose._ensemble>` to
        apply to results of multivariate theta-lines predictions (pd.DataFrame)
        in order to get resulting univariate prediction (pd.Series).
        The aggregation takes place across different theta-lines (row-wise), for
        given time stamps and hierarchy indices, if present.
    weights: list of floats, default=None
        Weights to apply in aggregation. Weights are passed as a parameter to
        the aggregation function, must correspond to each theta-line. None will
        result in non-weighted aggregation.

    References
    ----------
    .. [1] V.Assimakopoulos et al., "The theta model: a decomposition approach
       to forecasting", International Journal of Forecasting, vol. 16, pp. 521-530,
       2000.
    .. [2] E.Spiliotis et al., "Generalizing the Theta method for
       automatic forecasting ", European Journal of Operational
       Research, vol. 284, pp. 550-558, 2020.

    See Also
    --------
    ThetaForecaster, ThetaLinesTransformer

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.theta import ThetaModularForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> y = load_airline()
    >>> forecaster = ThetaModularForecaster(
    ...     forecasters=[
    ...         ("trend", PolynomialTrendForecaster(), 0),
    ...         ("arima", NaiveForecaster(), 3),
    ...     ],
    ...     theta_values=(0, 3),
    ... )
    >>> forecaster.fit(y)
    ThetaModularForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "authors": ["GuzalBulatova", "fkiraly"],
        "univariate-only": False,
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_version": ">3.7",
    }

    def __init__(
        self,
        forecasters=None,
        theta_values=(0, 2),
        aggfunc="mean",
        weights=None,
    ):
        super().__init__()
        self.forecasters = forecasters
        self.aggfunc = aggfunc
        self.weights = weights
        self.theta_values = theta_values

        forecasters_ = self._check_forecasters(forecasters)
        self._colens = ColumnEnsembleForecaster(forecasters=forecasters_)

        self.pipe_ = TransformedTargetForecaster(
            steps=[
                ("transformer", ThetaLinesTransformer(theta=self.theta_values)),
                ("forecaster", self._colens),
            ]
        )

    def _check_forecasters(self, forecasters):
        if forecasters is None:
            _forecasters = []
            for i, theta in enumerate(self.theta_values):
                if theta == 0:
                    name = f"trend{str(i)}"
                    forecaster = (name, PolynomialTrendForecaster(), i)
                elif _check_estimator_deps(ExponentialSmoothing, severity="none"):
                    name = f"ses{str(i)}"
                    forecaster = name, ExponentialSmoothing(), i
                else:
                    raise RuntimeError(
                        "Constructing ThetaModularForecaster without forecasters "
                        "results in using ExponentialSmoothing for non-zero theta "
                        "components. Ensure that statsmodels package is available "
                        "when constructing ThetaModularForecaster with this default."
                    )
                _forecasters.append(forecaster)
        elif len(forecasters) != len(self.theta_values):
            raise ValueError(
                f"The N of forecasters should be the same as the N "
                f"of theta_values, but found {len(forecasters)} forecasters and"
                f"{len(self.theta_values)} theta values."
            )
        else:
            _forecasters = forecasters
        return _forecasters

    def _fit(self, y, X, fh):
        self.pipe_.fit(y=y, X=X, fh=fh)
        return self

    def _predict(self, fh, X=None, return_pred_int=False):
        # Call predict on the forecaster directly, not on the pipeline
        # because of output conversion
        Y_pred = self.pipe_.steps_[-1][-1].predict(fh, X)
        y_pred = _aggregate(Y_pred, aggfunc=self.aggfunc, weights=self.weights)
        y_pred.name = self._y.name
        return y_pred

    def _update(self, y, X=None, update_params=True):
        self.pipe_._update(y, X=None, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test
            instance, i.e., ``MyClass(**params)`` or ``MyClass(**params[i])``
            creates a valid test instance. ``create_test_instance`` uses the first
            (or only) dictionary in ``params``.
        """
        # imports
        from sktime.forecasting.naive import NaiveForecaster

        params0 = {
            "forecasters": [
                ("naive", NaiveForecaster(), 0),
                ("naive1", NaiveForecaster(), 1),
            ]
        }
        params1 = {"theta_values": (0, 3)}
        params2 = {"weights": [1.0, 0.8]}

        # params1 and params2 invoke ExponentialSmoothing which requires statsmodels
        if _check_estimator_deps(ExponentialSmoothing, severity="none"):
            params = [params0, params1, params2]
        else:
            params = params0

        return params

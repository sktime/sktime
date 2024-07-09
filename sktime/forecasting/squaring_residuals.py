# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements the probabilistic Squaring Residuals forecaster."""

__all__ = ["SquaringResiduals"]
__author__ = ["kcc-lion"]

import pandas as pd

from sktime.datatypes._convert import convert_to
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.utils.warnings import warn


class SquaringResiduals(BaseForecaster):
    r"""Compute the prediction variance based on a separate forecaster.

    Wraps a ``forecaster`` with another ``residual_forecaster`` object that
    allows for quantile and interval estimation by fitting the
    ``residual_forecaster`` to the rolling residuals.

    Fitting proceeds as follows:
    Let :math:`t_1, \dots, t_N` be the train set.
    Let ``steps_ahead`` be a positive integer indicating the steps ahead
    we want to forecast the residuals. Let ``initial_window`` be
    the minimal number of observations to which the forecaster is fitted.

    1. For :math:`i = initial\_window, \dots, N - steps\_ahead`
        a. Train/Update forecaster A on :math:`y(t_1), \dots, y(t_i)`
        b. Make point prediction for :math:`t_{i+steps\_ahead}` to get
           :math:`\hat{y}(t_{i+steps\_ahead})`
        c. Compute the residual for :math:`t_{i+steps\_ahead}` as
           :math:`r(t_{i+steps\_ahead}) := y(t_{i+steps\_ahead})
           - \hat{y}(t_{i+steps\_ahead})`
        d. Compute :math:`e(t_{i+steps\_ahead}) := h(r(t_{i+steps\_ahead}))`
           where :math:`h(x)` is given by :math:`strategy`
    2. Train ``residual_forecaster`` on
       :math:`e(t_{initial\_window+steps\_ahead}), \dots, e(t_{N})`

    Prediction for :math:`t_{N+steps\_ahead}` is done as follows:

    1. Use ``forecaster`` to predict location param :math:`\hat{y}(t_{N+steps\_ahead})`
    2. Use ``residual_forecaster`` to predict scale param :math:`e(t_{N+steps\_ahead})`
    3. Calculate prediction intervals based on e.g. normal assumption
       :math:`N(\hat{y}(t_{N+steps\_ahead}),  e(t_{N+steps\_ahead}))`

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster descendant, optional
        Estimator to which probabilistic forecasts are being added
        Default = NaiveForecaster()
    residual_forecaster : sktime forecaster, BaseForecaster descendant, optional
        Estimator which is fitted to the residuals of forecaster
        Default = NaiveForecaster()
    initial_window : int, optional, default=2
        Size of initial_window to which forecaster is fitted
    steps_ahead : int, optional, default=1
        Steps ahead for which we predict the residuals
    strategy : str, optional, default='square'
        Function applied to the residuals
    distr : str, optional, default='norm'
        Distributional assumption (["norm", "laplace", "t", "cauchy"])
    distr_kwargs : dict, optional
        Additional arguments required by the distribution

    Examples
    --------
    >>> from sktime.datasets import load_macroeconomic
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> from sktime.forecasting.squaring_residuals import SquaringResiduals
    >>> fc = NaiveForecaster()
    >>> var_fc = ThetaForecaster()  # doctest: +SKIP
    >>> y = load_macroeconomic().realgdp  # doctest: +SKIP
    >>> sqr = SquaringResiduals(forecaster=fc, residual_forecaster=var_fc)
    ... # doctest: +SKIP
    >>> fh = ForecastingHorizon(values=[1, 2, 3])  # doctest: +SKIP
    >>> sqr = sqr.fit(y, fh=fh)  # doctest: +SKIP
    >>> pred_interval = sqr.predict_interval(coverage=0.95)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["kcc-lion", "fkiraly"],
        "maintainers": ["kcc-lion"],
        # estimator type
        # --------------
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": True,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": True,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:insample": False,
        "capability:pred_int": True,  # does forecaster implement proba forecasts?
        "capability:pred_int:insample": False,
    }

    def __init__(
        self,
        forecaster=None,
        residual_forecaster=None,
        initial_window=5,
        strategy="square",
        distr="norm",
        distr_kwargs=None,
    ):
        self.forecaster = forecaster
        self.residual_forecaster = residual_forecaster
        self.strategy = strategy
        self.initial_window = initial_window
        self.distr = distr
        self.distr_kwargs = distr_kwargs
        super().__init__()

        assert self.distr in ["norm", "laplace", "t", "cauchy"]
        assert self.strategy in ["square", "abs"]
        assert self.initial_window >= 1, (
            "Initial window should be larger or equal" " to one"
        )

        if self.forecaster is None:
            self.forecaster = NaiveForecaster()
        if self.residual_forecaster is None:
            self.residual_forecaster = NaiveForecaster()

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        fh_rel = fh.to_relative(self.cutoff)
        self._res_forecasters = {}
        self._residual_forecaster_ = self.residual_forecaster.clone()
        self._forecaster_ = self.forecaster.clone()

        y = convert_to(y, "pd.Series")
        cv = ExpandingWindowSplitter(initial_window=self.initial_window, fh=fh_rel)
        self._forecaster_.fit(y=y.iloc[: self.initial_window], X=X)
        y_pred = self._forecaster_.update_predict(y=y, cv=cv, X=X, update_params=True)

        for step_ahead in fh_rel:
            if isinstance(y.index, pd.DatetimeIndex):
                fh_current = ForecastingHorizon(step_ahead, freq=y.index.freq)
            else:
                fh_current = ForecastingHorizon(step_ahead)
            # create current prediction series
            if len(fh_rel) == 1:
                y_pred_current = y_pred
            else:
                y_pred_current = []
                y_pred_current_index = []
                for col in y_pred.columns:
                    fh_current_abs = fh_current.to_absolute_index(col)
                    y_pred_current.append(y_pred.at[fh_current_abs[0], col])
                    y_pred_current_index.append(fh_current_abs[0])
                y_pred_current = pd.Series(
                    data=y_pred_current, index=y_pred_current_index
                )

            # get residuals
            y_step = y[y_pred_current.index]
            residuals = y_step - y_pred_current
            if self.strategy == "square":
                residuals = residuals**2
            else:
                residuals = residuals.abs()
            # residuals.index = y_step_index
            if isinstance(residuals.index, pd.DatetimeIndex):
                residuals = residuals.asfreq(y.index.freq)

            # fit to residuals
            res_step_forecaster_ = self.residual_forecaster.clone()
            res_step_forecaster_.fit(y=residuals)
            self._res_forecasters[step_ahead] = res_step_forecaster_
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        fh_abs = fh.to_absolute(self.cutoff)
        y_pred = self._forecaster_.predict(X=X, fh=fh_abs)
        y_pred.name = self._y.name
        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        self._forecaster_.update(X=X, y=y, update_params=update_params)
        for forecaster in self._res_forecasters.values():
            forecaster.update(X=X, y=y, update_params=update_params)
        return self

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        eval(f"exec('from scipy.stats import {self.distr}')")
        fh_abs = fh.to_absolute(self.cutoff)
        y_pred = self._forecaster_.predict(fh=fh_abs, X=X)
        pred_var = self._predict_var(fh=fh, X=X)
        if self.distr_kwargs is not None:
            z_scores = eval(self.distr).ppf(alpha, **self.distr_kwargs)
        else:
            z_scores = eval(self.distr).ppf(alpha)

        errors = [pred_var * z for z in z_scores]

        var_names = self._get_varnames()
        var_name = var_names[0]

        index = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(columns=index)
        for a, error in zip(alpha, errors):
            pred_quantiles[(var_name, a)] = y_pred + error

        pred_quantiles.index = fh_abs.to_pandas()

        return pred_quantiles

    def _predict_var(self, fh, X=None, cov=False):
        """Compute/return prediction variance for a forecast.

        Must be run *after* the forecaster has been fitted.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        cov : bool, optional (default=False)
            If True, return the covariance matrix.
            If False, return the marginal variance.

        Returns
        -------
        pred_var :
            if cov=False, pd.DataFrame with index fh.
                a vector of same length as fh with predictive marginal variances;
            if cov=True, pd.DataFrame with index fh and columns fh.
                a square matrix of size len(fh) with predictive covariance matrix.
        """
        if cov:
            warn(
                f"cov={cov} is not supported in SquaringResiduals. "
                "Defaulting to cov=False instead.",
                obj=self,
            )
        fh_abs = fh.to_absolute(self.cutoff)
        fh_rel = fh.to_relative(self.cutoff)
        fh_rel_index = fh_rel.to_pandas()
        pred_var = pd.Series(index=fh_rel_index, dtype="float64")
        for el in fh_rel:
            pred_var.at[el] = self._res_forecasters[el].predict(fh=el)
        if self.strategy == "square":
            pred_var = pred_var**0.5
        pred_var.index = fh_abs.to_pandas()
        return pred_var

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.forecasting.croston import Croston
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import TrendForecaster

        params = [
            {
                "forecaster": Croston(),
                "residual_forecaster": Croston(),
                "initial_window": 2,
                "distr": "t",
                "distr_kwargs": {"df": 21},
            },
            {
                "forecaster": TrendForecaster(),
                "residual_forecaster": NaiveForecaster(),
                "initial_window": 5,
                "distr": "norm",
            },
        ]
        return params

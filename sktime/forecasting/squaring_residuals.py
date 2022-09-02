# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Extension template for forecasters.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y, cutoff, _fh,
    _cutoff, _converter_store_y, forecasters_, _tags, _tags_dynamic, _is_vectorized
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details: https://www.sktime.org/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, y, X=None, fh=None)
    forecasting     - _predict(self, fh=None, X=None)

Optional implements:
    updating                    - _update(self, y, X=None, update_params=True):
    predicting quantiles        - _predict_quantiles(self, fh, X=None, alpha=None)
    OR predicting intervals     - _predict_interval(self, fh, X=None, coverage=None)
    predicting variance         - _predict_var(self, fh, X=None, cov=False)
    distribution forecast       - _predict_proba(self, fh, X=None)
    fitted parameter inspection - get_fitted_params()

Testing - implement if sktime forecaster (not needed locally):
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
__author__ = ["kcc-lion"]


import pandas as pd

from sktime.datatypes._convert import convert_to
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster


class SquaringResiduals(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    est : sktime.estimator, BaseEstimator descendant
        descriptive explanation of est
    est2: another estimator
        descriptive explanation of est2
    and so on
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    #  when changing scitype:y to multivariate or both:
    #   y_inner_mtype should be changed to pd.DataFrame
    # other tags are "safe defaults" which can usually be left as-is
    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": True,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement proba forecasts?
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }
    #  in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__ (avoid this if possible)

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        forecaster=None,
        residual_forecaster=None,
        initial_window=2,
        steps_ahead=1,
        strategy="square",
        distr="norm",
        distr_kwargs=None,
    ):
        # estimators should precede parameters
        #  if estimators have default values, set None and initalize below
        self.forecaster = forecaster
        self.residual_forecaster = residual_forecaster
        self.strategy = strategy
        self.steps_ahead = steps_ahead
        self.initial_window = initial_window
        self.distr = distr
        self.distr_kwargs = distr_kwargs
        super(SquaringResiduals, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        assert self.distr in ["norm", "laplace", "t", "cauchy"]
        assert self.strategy in ["square", "abs"]
        assert self.steps_ahead >= 1, "Steps ahead should be larger or equal to one"
        assert self.initial_window >= 1, (
            "Initial window should be larger or equal" " to one"
        )

        if self.forecaster is None:
            self.forecaster = NaiveForecaster()
        if self.residual_forecaster is None:
            self.residual_forecaster = NaiveForecaster()

    def _fit(self, y, X=None, fh=None):
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
        self._residual_forecaster_ = self.residual_forecaster.clone()
        self._forecaster_ = self.forecaster.clone()

        y = convert_to(y, "pd.Series")
        self.cv = ExpandingWindowSplitter(
            initial_window=self.initial_window, fh=self.steps_ahead
        )
        self._forecaster_.fit(y=y.iloc[: self.initial_window], X=X)
        y_pred = self._forecaster_.update_predict(
            y=y, cv=self.cv, X=X, update_params=True
        )
        residuals = y.iloc[self.initial_window :] - y_pred
        if self.strategy == "square":
            residuals = residuals**2
        else:
            residuals = residuals.abs()

        self._residual_forecaster_.fit(y=residuals)
        return self

    def _predict(self, fh, X=None):
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
        return y_pred

    def _update_predict_single(self, y, fh, X=None, update_params=True):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.
        """
        self._forecaster_.update(self._y, X, update_params=update_params)
        y_pred = self._forecaster_.predict(fh=y.index, X=X)
        residuals = y - y_pred
        if self.strategy == "square":
            residuals = residuals**2
        else:
            residuals = residuals.abs()
        self._residual_forecaster_.update(y=residuals, update_params=update_params)
        return self.predict(fh, X)

    def _predict_quantiles(self, fh, X=None, alpha=None):
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
        pred_var = self._predict_var(fh=fh_abs, X=X)
        if self.distr_kwargs is not None:
            z_scores = eval(self.distr).ppf(alpha, **self.distr_kwargs)
        else:
            z_scores = eval(self.distr).ppf(alpha)

        errors = [pred_var * z for z in z_scores]

        index = pd.MultiIndex.from_product([["Quantiles"], alpha])
        pred_quantiles = pd.DataFrame(columns=index)
        for a, error in zip(alpha, errors):
            pred_quantiles[("Quantiles", a)] = y_pred + error

        pred_quantiles.index = fh_abs

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
        pred_var = self._residual_forecaster_.predict(X=X, fh=fh)
        pred_var = convert_to(pred_var, to_type="pd.Series")
        if self.strategy == "square":
            pred_var = pred_var**0.5
        return pred_var

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
        y_pred = self._forecaster_.update_predict(
            y=y, cv=self.cv, X=X, update_params=update_params
        )
        residuals = y.iloc[self.initial_window :] - y_pred
        if self.strategy == "square":
            residuals = residuals**2
        else:
            residuals = residuals.abs()
        self._residual_forecaster_.update_predict(
            y=residuals, update_params=update_params
        )
        return self

        # implement here
        # IMPORTANT: avoid side effects to X, fh

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.croston import Croston
        from sktime.forecasting.naive import NaiveForecaster

        params = [
            {
                "forecaster": NaiveForecaster(),
                "residual_forecaster": NaiveForecaster(),
                "initial_window": 2,
                "distr": "norm",
            },
            {
                "forecaster": NaiveForecaster(),
                "residual_forecaster": NaiveForecaster(),
                "initial_window": 2,
                "distr": "t",
                "distr_kwargs": {"df": 21},
            },
            {
                "forecaster": Croston(),
                "residual_forecaster": Croston(),
                "initial_window": 2,
                "distr": "t",
                "distr_kwargs": {"df": 21},
            },
        ]
        return params


if __name__ == "__main__":
    from sktime.datasets import load_macroeconomic
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.utils.estimator_checks import check_estimator

    forecaster = NaiveForecaster()
    residual_forecaster = ThetaForecaster()
    y = load_macroeconomic().realgdp
    sqr = SquaringResiduals(
        forecaster=forecaster, residual_forecaster=residual_forecaster
    )
    fh_fit = ForecastingHorizon(values=[1])
    sqr.fit(y, fh=fh_fit)
    fh_pred = ForecastingHorizon(values=[1, 2, 3])
    print(sqr.predict(ForecastingHorizon(values=[2, 5])))
    print(sqr.predict_quantiles(fh_pred, X=None, alpha=[0.025, 0.975]))
    print(sqr.predict_interval(fh_pred, coverage=0.95))
    print(check_estimator(SquaringResiduals, return_exceptions=True))

# -*- coding: utf-8 -*-
"""
Extension template for forecasters

How to use this:
- this is meant as a "fill in" template for easy extension
- do NOT import this file directly - it will break
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by testing forecasting/tests/test_all_forecasters
        and forecasting/tests/test_sktime_forecasters
- once complete: use as a local library, or contribute to sktime via PR

Mandatory implements:
    fitting         - _fit(self, y, X=None, fh=None)
    forecasting     - _predict(self, fh=None, X=None, return_pred_int=False,
                               alpha=DEFAULT_ALPHA)

Optional implements:
    updating        - _update(self, y, X=None, update_params=True):
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from sktime.forecasting.base import BaseEstimator
from sktime.forecasting.base import DEFAULT_ALPHA

# todo: add any necessary imports here


class MyForecaster(BaseEstimator):
    """Base forecaster

    The base forecaster specifies the methods and method
    signatures that all forecasters have to implement.

    Specific implementations of these methods is deferred to concrete
    forecasters.

    Hyper-parameters
    ----------------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on

    Components
    ----------
    est : sktime.estimator, BaseEstimator descendant
        descriptive explanation of est
    est2: another estimator
        descriptive explanation of est2
    and so on
    """

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):

        # todo: write any hyper-parameters and components to self
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

        # todo: initialize None parameters, where necessary
        if paramc is None:
            self.paramc = "42"

        # todo: uncomment if forecast horizon is needed only in predict
        # self._tags["fh_in_fit"] = "optional"

        # todo: change "MyForecaster" to the name of the class
        super(MyForecaster, self).__init__()

    # todo: implement this, mandatory
    def _fit(self, y, X=None, fh=None):
        """fit forecaster to training data
            core logic

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

        # implement here
        # IMPORTANT: avoid side effects to y, X, fh

    # todo: implement this, mandatory
    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon
            core logic

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """

        # implement here
        # IMPORTANT: avoid side effects to X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data
            core logic

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals

        State change
        ------------
        updates self._X and self._y with new data
        updates self.cutoff to most recent time in y
        if update_params=True, updates model (attributes ending in "_")
        """

        # implement here
        # IMPORTANT: avoid side effects to X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the method
    def _update_predict_single(
        self,
        y,
        fh,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Internal method for updating and making forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.
        """
        self.update(y, X, update_params=update_params)
        return self.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)

        # implement here
        # IMPORTANT: avoid side effects to y, X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the method
    def _compute_pred_int(self, alphas):
        """Calculate the prediction errors for each point.

        Parameters
        ----------

        alpha : float or list, optional (default=0.95)
            A significance level or list of significance levels.

        Returns
        -------

        errors : list of pd.Series
            Each series in the list will contain the errors for each point in
            the forecast for the corresponding alpha.
        """
        # implement here

    # todo: consider implementing this, optional
    # if not implementing, delete the method
    def _predict_moving_cutoff(
        self,
        y,
        cv,
        X=None,
        update_params=True,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Make single-step or multi-step moving cutoff predictions

        Parameters
        ----------
        y : pd.Series
        cv : temporal cross-validation generator
        X : pd.DataFrame
        update_params : bool
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series
        """

        # implement here
        # IMPORTANT: avoid side effects to y, X, cv

    # todo: consider implementing this, optional
    # if not implementing, delete the method
    def get_fitted_params(self):
        """Get fitted parameters

        Returns
        -------
        fitted_params : dict
        """
        # implement here

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extension template for forecasters, SIMPLE version.

Contains only bare minimum of implementation requirements for a functional forecaster.
Also assumes *no composition*, i.e., no forecaster or other estimator components.
Assumes pd.DataFrame used internally, and no hierarchical functionality.
For advanced cases (probabilistic, composition, hierarchical, etc),
    see extension templates in forecasting.py or forecasting_simple.py

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
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, y, X=None, fh=None)
    forecasting     - _predict(self, fh=None, X=None)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [sbuse]
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base.adapters import _ProphetAdapter
from sktime.forecasting.fbprophet import Prophet

class PiecewiseLinearTrendForecaster(_ProphetAdapter):
    """This is a piecewise linear forecaster. It is bascially a wrapper around the 
    prophet model with Prophet(growth='linear') and extracts the trend modeling from it. 

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    """

    # todo: fill in the scitype:y tag for univariate/multivariate
    _tags = {
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "univariate",
        # fill in "univariate" or "both"
        #   "univariate": inner _fit, _predict, receives only single-column DataFrame
        #   "both": inner _predict gets pd.DataFrame series with any number of columns
        #
        # do not change these:
        # (look at advanced templates if you think these should change)
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.05,
        verbose=0,
    ):
        self.freq = None
        self.add_seasonality = None
        self.add_country_holidays = None
        self.growth = "linear"
        self.growth_floor = 0.0
        self.growth_cap = None
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = "auto"
        self.weekly_seasonality = "auto"
        self.daily_seasonality = "auto"
        self.holidays = None
        self.seasonality_mode = "additive"
        self.seasonality_prior_scale = 10.0
        self.changepoint_prior_scale = changepoint_prior_scale
        self.holidays_prior_scale = 10.0
        self.mcmc_samples = 0
        self.alpha = DEFAULT_ALPHA
        self.uncertainty_samples = 1000
        self.stan_backend = None
        self.verbose = verbose

        super().__init__()

        # import inside method to avoid hard dependency
        from prophet.forecaster import Prophet as _Prophet

        self._ModelClass = _Prophet

        # "changepoint_prior_scale": should be within [0.001,0.5]
        # "changepoint_range": has to be within [0,1], default = 0.8

    def _instantiate_model(self):
        self._forecaster = self._ModelClass(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=float(self.seasonality_prior_scale),
            holidays_prior_scale=float(self.holidays_prior_scale),
            changepoint_prior_scale=float(self.changepoint_prior_scale),
            mcmc_samples=self.mcmc_samples,
            interval_width=1 - self.alpha,
            uncertainty_samples=self.uncertainty_samples,
            stan_backend=self.stan_backend,
        )
        return self
    
    # _fit is defined in the superclass and is fine as is. 
    # here i overwrite the _predict from the superclass to just return the trend. 
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
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        fh = self._get_prophet_fh()
        future = pd.DataFrame({"ds": fh}, index=fh)
        
        out = self._forecaster.setup_dataframe(future.copy())
        out["trend"] = self._forecaster.predict_trend(out)
        
        y_pred = out.loc[:, "trend"]
        y_pred.index = future.index
        y_pred.name = self._y.columns[0]

        if self.y_index_was_int_ or self.y_index_was_period_:
            y_pred.index = self.fh.to_absolute_index(cutoff=self.cutoff)

        return y_pred
    
    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params

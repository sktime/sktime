"""THieF Forecaster implementation."""

"""Extension template for forecasters.

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

Optional implements:
    updating                    - _update(self, y, X=None, update_params=True):
    predicting quantiles        - _predict_quantiles(self, fh, X=None, alpha=None)
    OR predicting intervals     - _predict_interval(self, fh, X=None, coverage=None)
    predicting variance         - _predict_var(self, fh, X=None, cov=False)
    distribution forecast       - _predict_proba(self, fh, X=None)
    fitted parameter inspection - _get_fitted_params()

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""

# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.mapa import MAPAForecaster

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class THieFForecaster(BaseForecaster):
    """
    Temporal Hierarchical Forecasting (THieF) implementation for sktime.

    THieF fits multiple instances of a base forecaster at different temporal aggregation
    levels, then reconciles their forecasts to ensure consistency across time scales.

    Parameters
    ----------
    base_forecaster : sktime-compatible forecaster
        The forecasting model to be used at each aggregation level.
    aggregation_levels : list of int, default=None
        The levels at which the time series will be aggregated
        (e.g., [1, 2, 4, 12] for monthly data).
        If None, the levels are automatically determined based on the frequency of `y`.
    reconciliation_method : str, default="ols"
        The method used to reconcile forecasts across levels. Options include:
        - "ols": Ordinary Least Squares
        - "bu": Bottom-Up (uses the lowest-level forecast)
        - "mse": Variance scaling based on Mean Squared Error
        - "shr": Shrinkage estimator for covariance matrix
        - "sam": Sample covariance matrix
    """

    _tags = {
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": False,
        "authors": "satvshr",
    }

    def __init__(
        self, base_forecaster, aggregation_levels=None, reconciliation_method="ols"
    ):
        self.base_forecaster = base_forecaster
        self.aggregation_levels = aggregation_levels
        self.reconciliation_method = reconciliation_method
        self.agg_method = "mean"
        self.forecasters = {}
        super().__init__()

    def _reconcile_forecasts(self, forecasts):
        """Reconcile forecasts using the specified reconciliation method."""
        if self.reconciliation_method == "bu":
            return forecasts[min(forecasts.keys())]
        elif self.reconciliation_method == "ols":
            return np.mean(list(forecasts.values()), axis=0)
        elif self.reconciliation_method == "mse":
            weights = [
                1 / np.var(f) if np.var(f) > 0 else 1 for f in forecasts.values()
            ]
            weights = np.array(weights) / np.sum(weights)
            return np.average(list(forecasts.values()), axis=0, weights=weights)
        elif self.reconciliation_method == "struc":
            weights = [1 / (level + 1) for level in forecasts.keys()]
            weights = np.array(weights) / np.sum(weights)
            return np.average(list(forecasts.values()), axis=0, weights=weights)
        elif self.reconciliation_method == "shr":
            residuals = np.array(list(forecasts.values()))
            cov_matrix = np.cov(residuals, rowvar=False)
            shrinkage = np.mean(np.diag(cov_matrix)) * np.eye(cov_matrix.shape[0])
            shrunk_cov = 0.5 * cov_matrix + 0.5 * shrinkage
            weights = 1 / np.diag(shrunk_cov)
            weights /= np.sum(weights)
            return np.average(list(forecasts.values()), axis=0, weights=weights)
        elif self.reconciliation_method == "sam":
            residuals = np.array(list(forecasts.values()))
            cov_matrix = np.cov(residuals, rowvar=False)
            inv_cov = np.linalg.pinv(cov_matrix)
            weights = np.sum(inv_cov, axis=1)
            weights /= np.sum(weights)
            return np.average(list(forecasts.values()), axis=0, weights=weights)
        else:
            raise NotImplementedError(
                f"Reconciliation method '{self.reconciliation_method}' not implemented."
            )

    def _determine_aggregation_levels(self, y):
        """Determine the aggregation level based on the frequency of the time series."""
        freq = y.index.freqstr or pd.infer_freq(y.index)
        if freq is None:
            raise ValueError("Could not determine frequency of time series.")

        freq_map = {"D": 7, "W": 52, "ME": 12, "H": 24, "Q": 4, "Y": 1}
        m = freq_map.get(freq, None)

        if m is None:
            raise ValueError(f"Unsupported frequency '{freq}'.")

        aggregation_levels = [i for i in range(1, m + 1) if m % i == 0]
        return aggregation_levels

    # todo: implement this, mandatory
    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
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
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self._y_cols = (
            y.columns
            if isinstance(y, pd.DataFrame)
            else pd.Index([y.name])
            if y.name
            else pd.Index(["c0"])
        )
        if isinstance(y, pd.Series):
            y = pd.DataFrame(y)
        self.aggregation_levels_ = self._determine_aggregation_levels(y)
        for level in self.aggregation_levels_:
            y_agg = MAPAForecaster._aggregate(self, y, level)
            y_agg.columns = self._y_cols
            print("*************")
            print(y_agg)
            print("*************")
            forecaster = self.base_forecaster.clone()
            forecaster.fit(y_agg, X, fh)
            self.forecasters[level] = forecaster

        return self

    # todo: implement this, mandatory
    def _predict(self, fh=None, X=None):
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
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        self._forecast_store = {}
        for level, forecaster in self.forecasters.items():
            y_pred_agg = forecaster.predict(fh, X)
            self._forecast_store[level] = y_pred_agg

        result = self._reconcile_forecasts(self._forecast_store)

        forecast_index = fh.to_absolute(self.cutoff).to_pandas()
        self._y_pred = pd.DataFrame(
            result.reshape(-1, len(self._y_cols)),
            index=forecast_index,
            columns=self._y_cols,
        )

        return self._y_pred

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
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        if isinstance(y, pd.Series):
            raise AttributeError("THieF requires a DataFrame with an index column.")

        for level in self.aggregation_levels_:
            y_agg = MAPAForecaster._aggregate(self, y, level)
            if level in self.forecasters:
                self.forecasters[level].update(y_agg, X, update_params=update_params)
            else:
                raise ValueError(
                    f"No trained forecaster found for aggregation level {level}"
                )

        return self

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
        from sktime.forecasting.naive import NaiveForecaster

        params = [
            {"base_forecaster": NaiveForecaster(strategy="mean")},
            {
                "base_forecaster": NaiveForecaster(strategy="last"),
                "reconciliation_method": "bu",
            },
        ]

        return params

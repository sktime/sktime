# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""PyFableARIMA: A forecaster that wraps CRAN fable::ARIMA."""

__author__ = ["ericjb"]

import io
import sys

import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.dependencies import _check_soft_dependencies


class PyFableARIMA(BaseForecaster):
    r"""A wrapper to the ARIMA model in the CRAN package fable.

    (See https://cran.r-project.org/web/packages/fable/index.html)
    (The documentation here is lifted from the documentation there. Consult the fable
    version for additional detail.)
    Searches through the model space specified in the specials to identify
    the best ARIMA model, with the lowest AIC, AICc or BIC value. It is
    implemented using R's stats::arima() and allows ARIMA models to be used
    in the fable framework.

    Parameters
    ----------
    formula	: string, optional (default = None)
        Model specification (e.g. "y ~ z")
        N.B. To specify a model fully (avoid automatic selection), the
        intercept and `pdq()/PDQ()` values must be specified. For example,
        formula = `sales ~ 1 + pdq(1, 1, 1) + PDQ(1, 0, 0)`.
    ic : string, optional (default = "aicc")
        The information criterion used in selecting the model.
        One of "aic", "aicc", "bic"
    selection_metric : opitonal; a function
        selection_metric = function(x) x[[ic]],
        A function used to compute a metric from an Arima object which is minimised
        to select the best model.
    stepwise : logical (default = True)
        Should the stepwise search algorithm be used? Stepwise is a greedy-like
        algorithm that can significantly reduce the number of models tested,
        which can make the search much faster. If used, there is a risk of
        missing the global minimum.
    greedy : logical (default = True)
        Should the stepwise search move to the next best option immediately?
    approximation : logical (default = None)
        Should CSS (conditional sum of squares) be used during model selection?
        The default (NULL) will use the approximation if there are more than
        150 observations or if the seasonal period is greater than 12.
    order_constraint : string, optional
        (default = `p + q + P + Q <= 6 & (constant + d + D <= 2)`
        A logical predicate on the orders of p, d, q, P, D, Q and constant to consider
        in the search. See "Specials" for the meaning of these terms.
    unitroot_spec : optional
        A specification of unit root tests to use in the selection of d and D.
        See unitroot_options() for more details
    trace : logical (default = False)
        If True, the selection_metric of estimated models in the selection procedure
        will be outputted to the console.
    is_regular : logical (default = True)
        Is the series regular? (i.e. are the time-steps equal throughout)


    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.PyFableARIMA import PyFableARIMA
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> airline = load_airline()  # a pandas Series with a PeriodIndex (freq='M')
    >>> airline.name = "Passengers"  # Ensure the name matches your ARIMA formula
    >>> train, test = temporal_train_test_split(airline, test_size=12)
    >>> best = PyFableARIMA(formula='Passengers').fit(train)
    >>> print(best.report())
    >>> fitted = best.predict(train.index)
    >>> print(f"fitted = \n{fitted}")
    >>> pred = best.predict(test.index)
    >>> print(f"pred = \n{pred}")
    >>> pred_int = best.predict_interval(fh=test.index, coverage=[0.95, 0.50])
    >>> print(f"pred_int = \n{pred_int}")
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "authors": ["ericjb"],
        "maintainers": ["ericjb"],
        "python_version": None,
        "python_dependencies": ["rpy2"],
    }

    def __init__(
        self,
        formula=None,
        ic="aicc",
        selection_metric=None,
        stepwise=True,
        greedy=True,
        approximation=None,
        order_constraint=None,
        unitroot_spec=None,
        trace=False,
        is_regular=True,
    ):
        self.formula = formula
        self.ic = ic
        self.selection_metric = selection_metric
        self.stepwise = stepwise
        self.greedy = greedy
        self.approximation = approximation
        self.order_constraint = order_constraint
        self.unitroot_spec = unitroot_spec
        self.trace = trace
        self.is_regular = is_regular
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

    def _custom_prepare_tsibble(self, Z, is_regular=True):
        """fable::ARIMA expects an R tsibble object.

        Parameters
        ----------
        Z : a pd.Series or pd.DataFrame
            Will be either y or X
        is_regular : logical
            Is the series regular. (The frequency is obtained from Z.)
        """
        _check_soft_dependencies("rpy2", severity="error")

        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        # Activate the automatic conversion of pandas DataFrames to R data frames
        # pandas2ri.activate()

        # Import necessary R packages
        _ = importr("tidyverse")
        _ = importr("fpp3")
        tsibble = importr("tsibble")
        _ = importr("dplyr")
        _ = importr("fabletools")
        _ = importr("fable")

        if isinstance(Z.index, pd.DatetimeIndex):
            freq = Z.index.freq
            if freq is None:
                freq = Z.index.inferred_freq
            freq = str(freq)[0]
        elif isinstance(Z.index, pd.PeriodIndex):
            freq = Z.index.freqstr[0]
        else:
            raise ValueError("Index must be of type DatetimeIndex or PeriodIndex")

        date_col_name = "Date"
        if isinstance(Z, pd.Series):
            Z = pd.DataFrame({Z.name: Z.values, "Date": Z.index.to_series()})
        else:
            if date_col_name in Z.columns:
                counter = 1
                while f"{date_col_name}_{counter}" in Z.columns:
                    counter += 1
                date_col_name = f"{date_col_name}_{counter}"

            Z[date_col_name] = Z.index.to_series()

        if isinstance(Z.index, pd.PeriodIndex):
            Z[date_col_name] = Z.index.to_timestamp()

        # Convert the pandas DataFrame to an R data frame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(Z)

        # Define the R script to prepare the tsibble
        if is_regular:
            freq_func_map = {
                "A": "lubridate::year",
                "Y": "lubridate::year",
                "Q": "tsibble::yearquarter",
                "M": "tsibble::yearmonth",
                "W": "tsibble::yearweek",
                "D": "as.Date",
            }
            if freq in freq_func_map:
                freq_func = freq_func_map[freq]
            else:
                raise ValueError("Unsupported frequency")

            r_script = f"""
            r_data <- dplyr::as_tibble(r_data) |>
                dplyr::mutate(idx = {freq_func}({date_col_name})) |>
                tsibble::as_tsibble(index = idx, regular = TRUE)
            r_data
            """
        else:
            r_script = f"""
            r_data <- dplyr::as_tibble(r_data) |>
                dplyr::mutate({date_col_name} = as.Date({date_col_name})) |>
                tsibble::as_tsibble(index = {date_col_name}, regular = FALSE)
            r_data
            """

        # Execute the R script
        robjects.globalenv["r_data"] = r_data
        tsibble = robjects.r(r_script)

        return tsibble

    def _custom_fit_arima(self, train, expr, x="mdl"):
        """Fit an ARIMA model using R fable::ARIMA.

        Parameters
        ----------
        train      a tsibble; the data to use to train the model
        expr       a character string; either a column name or a formula
                e.g. 'y' or 'y ~ z'
        x          a character string; a name for the model
        """
        import rpy2.robjects as robjects

        if self.stepwise is False:
            model_string = (
                f"fabletools::model({x} = "
                f"fable::ARIMA(formula = {expr}, stepwise = FALSE))"
            )
        else:
            model_string = f"fabletools::model({x} = fable::ARIMA(formula = {expr}))"

        # Define the R script to fit the ARIMA model
        r_script = f"""
        fit.aut.arima <- train |> {model_string}
        fit.aut.arima
        """

        # Execute the R script
        robjects.globalenv["train"] = train
        fit_aut_arima = robjects.r(r_script)

        return fit_aut_arima

    def report(self):
        """Call the R function report on the ARIMA fit and return the output."""
        import rpy2.robjects as robjects

        # Capture the R output
        report_output = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = report_output

        try:
            robjects.r["report"](self._fit_auto_arima_)
        finally:
            sys.stdout = sys_stdout

        # Get the captured output
        report_str = report_output.getvalue()

        return report_str

    def PyFableARIMA_report(self):
        """Call the report method which calls the R report() on the ARIMA fit.

        Print the report
        """
        report_str = self.report()
        print("ARIMA Report:")
        print(report_str)

    def get_fitted_values(self):
        """Extract the fitted values from the ARIMA fit object."""
        import rpy2.robjects as robjects

        fitted_values = robjects.r["fitted"](self._fit_auto_arima_)
        fitted_values_series = robjects.pandas2ri.rpy2py(fitted_values)
        fitted_values_series.index = self._fit_index_
        series = fitted_values_series[".fitted"].squeeze()
        series.name = self._y.name

        return series

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
        fh : ignored
        X :  sktime time series object, optional (default=None)
            Exogeneous time series used in fitting. Must have same index as y.

        Returns
        -------
        self : reference to self
        """
        if X is None:
            Z = y
        else:
            if not y.index.equals(X.index):
                raise ValueError("y and X must have the same index")
            Z = pd.concat([y, X], axis=1)

        r_tsibble = self._custom_prepare_tsibble(Z, is_regular=self.is_regular)

        self._fit_auto_arima_ = self._custom_fit_arima(r_tsibble, self.formula)
        self._fit_index_ = y.index

    # -------------------------------------------------------------------------
    def _predict_special(self, fh, X=None, coverage=None):
        """Unified internal prediction logic for point and interval forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
        X : pd.DataFrame or None
        Optional exogenous data matching fh.
        coverage : list of float, optional
        Coverage levels for prediction intervals (e.g. [0.9] for 90% interval).
        If None, only point forecasts are returned.

        Returns
        -------
        y_pred : pd.Series
        Mean point forecasts.
        pred_int : pd.DataFrame or None
        MultiIndex DataFrame with prediction intervals if `coverage` is provided.
        """
        import pandas as pd
        import rpy2.robjects as robjects

        if not isinstance(fh, ForecastingHorizon):
            fh = ForecastingHorizon(fh)

        cutoff = self._fit_index_[-1]
        l_fh_index = fh.to_absolute_index(cutoff=cutoff)

        # Prepare exogenous features or dummy
        if X is not None:
            l_fh_X_index = ForecastingHorizon(X.index).to_absolute_index()
            if not l_fh_index.equals(l_fh_X_index):
                raise ValueError(
                    "internal error: l_fh_index and l_fh_X_index do not agree"
                )
            a_tsibble = self._custom_prepare_tsibble(X, is_regular=self.is_regular)
        else:
            dummy = pd.Series(0, index=l_fh_index)
            a_tsibble = self._custom_prepare_tsibble(dummy, is_regular=self.is_regular)

        robjects.globalenv["fit_aut_arima"] = self._fit_auto_arima_
        robjects.globalenv["a_tsibble"] = a_tsibble

        # Always compute prediction intervals - simpler R logic below
        if not coverage:
            coverage = [0.9]

        level_vector = robjects.FloatVector([c * 100 for c in coverage])
        robjects.globalenv["level_vec"] = level_vector

        r_script = """
        library(fable)
        library(dplyr)
        library(purrr)
        library(tibble)

        fc <- forecast(fit_aut_arima, new_data = a_tsibble)

        dist_col <- names(fc)[sapply(fc, function(x) inherits(x, "distribution"))][1]

        # Extract hilo intervals for each coverage level
        intervals_all <- map(level_vec, function(lv) {
            dists <- map(fc[[dist_col]], function(x) hilo(x, lv))
            df <- tibble(interval = dists) %>%
                mutate(
                    lower = map_dbl(interval, function(x) x$lower),
                    upper = map_dbl(interval, function(x) x$upper)
                ) %>%
                select(lower, upper)
            df
        })

        names(intervals_all) <- paste0("c", as.character(level_vec))

        intervals_named <- imap(intervals_all, function(df, name) {
            rename_with(df, ~ paste0(., "_", name))
        })

        intervals_df <- bind_cols(intervals_named)

        fc_tbl <- bind_cols(as_tibble(fc), intervals_df)
        fc_tbl
        """

        forecasts = robjects.r(r_script)
        forecasts_df = robjects.pandas2ri.rpy2py(forecasts)

        # Extract forecast mean
        forecast_column = next(
            (col for col in forecasts_df.columns if ".mean" in col), None
        )
        if forecast_column is None:
            forecast_column = forecasts_df.select_dtypes(include="number").columns[0]

        forecast_values = forecasts_df[forecast_column].values
        y_pred = pd.Series(forecast_values, index=l_fh_index, name="forecast")

        # Extract prediction intervals
        intervals = {}
        y_varname = self._y.name if hasattr(self._y, "name") and self._y.name else "y"

        for col in forecasts_df.columns:
            col_str = str(col)
            if col_str.startswith("lower_") or col_str.startswith("upper_"):
                parts = col_str.split("_")
                bound = parts[0]  # "lower" or "upper"
                level = (
                    float(parts[1][1:]) / 100
                )  # strip "c" prefix and convert back to [0, 1]
                intervals[(y_varname, level, bound)] = forecasts_df[col].values

        pred_int = pd.DataFrame(intervals, index=l_fh_index)
        pred_int.columns = pd.MultiIndex.from_tuples(
            pred_int.columns, names=["variable", "coverage", "bound"]
        )

        return y_pred, pred_int

    # -------------------------------------------------------------------------

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : required. A ForecastingHorizon or something that can be converted to one.
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Mean predictions
        """
        import pandas as pd

        if not isinstance(fh, ForecastingHorizon):
            fh = ForecastingHorizon(fh)

        cutoff = self._fit_index_[-1]
        abs_fh = fh.to_absolute_index(cutoff=cutoff)

        # Get fitted values for in-sample support
        fitted_values = self.get_fitted_values()

        # Partition fh into in-sample and out-of-sample
        is_in_sample = abs_fh.isin(fitted_values.index)
        in_sample_index = abs_fh[is_in_sample]
        out_sample_index = abs_fh[~is_in_sample]

        y_pred_parts = []

        # In-sample values: look them up in fitted_values
        if len(in_sample_index) > 0:
            y_pred_in = fitted_values[in_sample_index]
            y_pred_parts.append(y_pred_in)

        # Out-of-sample values: use _predict_special
        if len(out_sample_index) > 0:
            fh_out = ForecastingHorizon(out_sample_index, is_relative=False)
            y_pred_out, _ = self._predict_special(fh_out, X=X)
            y_pred_parts.append(y_pred_out)

        # Combine predictions in correct order
        y_pred = pd.concat(y_pred_parts).loc[abs_fh]

        return y_pred

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
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        _, pred_int = self._predict_special(fh, X=X, coverage=coverage)
        return pred_int

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
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
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
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params

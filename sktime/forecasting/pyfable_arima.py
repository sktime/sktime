# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""PyFableARIMA: A forecaster that wraps CRAN fable::ARIMA."""

__author__ = ["ericjb"]

import io
import sys
from typing import Optional  # see _letter_from_index definition

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_fh


class PyFableARIMA(BaseForecaster):
    r"""ARIMA model from the CRAN package fable.

    Wraps the ``ARIMA`` model from the ``fable`` package in R, see [1] for details.

    Searches through the model space specified in the specials to identify
    the best ARIMA model, with the lowest AIC, AICc or BIC value. It is
    implemented using R's ``stats::arima()`` and allows ARIMA models to be used
    in the ``fable`` framework.

    Parameters
    ----------
    formula : string, optional (default = None)
        Model specification (e.g. "y ~ z")
        N.B. To specify a model fully (avoid automatic selection), the
        intercept and `pdq()/PDQ()` values must be specified. For example,
        formula = `sales ~ 1 + pdq(1, 1, 1) + PDQ(1, 0, 0)`.
    ic : string, optional (default = "aicc")
        The information criterion used in selecting the model.
        One of "aic", "aicc", "bic"
    selection_metric : optional; a function
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
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>> from sktime.forecasting.PyFableARIMA import PyFableARIMA  # doctest: +SKIP
    >>> from sktime.forecasting.model_selection import (  # doctest: +SKIP
    ...     temporal_train_test_split,
    ... )
    >>> airline = load_airline()  # Series with PeriodIndex freq='M'  # doctest: +SKIP
    >>> airline.name = "Passengers"  # name must match ARIMA formula  # doctest: +SKIP
    >>> train, test = temporal_train_test_split(airline, test_size=12)  # doctest: +SKIP
    >>> best = PyFableARIMA(formula='Passengers').fit(train)  # doctest: +SKIP
    >>> print(best.report())  # doctest: +SKIP
    >>> fitted = best.predict(train.index)  # doctest: +SKIP
    >>> print(f"fitted = \n{fitted}")  # doctest: +SKIP
    >>> pred = best.predict(test.index)  # doctest: +SKIP
    >>> print(f"pred = \n{pred}")  # doctest: +SKIP
    >>> pred_int = best.predict_interval(  # doctest: +SKIP
    ...     fh=test.index, coverage=[0.95, 0.50]
    ... )
    >>> print(f"pred_int = \n{pred_int}")  # doctest: +SKIP

    References
    ----------
    .. [1] Fable: https://cran.r-project.org/web/packages/fable/index.html
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:update": False,
        "authors": ["ericjb"],
        "maintainers": ["ericjb"],
        "python_version": ">=3.10",
        "env_marker": 'sys_platform == "linux"',
        "python_dependencies": ["rpy2==3.6.1"],
        # CI and test flags
        # -----------------
        "tests:skip_by_name": [
            # If fh has gaps then X will have gaps and ARIMA cannot handle that
            "test_predict_time_index_with_X",
            "test_update_predict_single",
            "test_update_predict_predicted_index",
        ],
        "tests:vm": True,  # run on separate VM to rpy2 in extras dep set
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
        verbose=False,
    ):
        self.formula = formula
        # resolved formula actually passed to R backend (may derive if None)
        self._resolved_formula = None
        self.ic = ic
        self.selection_metric = selection_metric
        self.stepwise = stepwise
        self.greedy = greedy
        self.approximation = approximation
        self.order_constraint = order_constraint
        self.unitroot_spec = unitroot_spec
        self.trace = trace
        self.is_regular = is_regular
        self.verbose = verbose
        self.int_index_to_annual = False
        super().__init__()

    # ------------------------------------------------------------------
    def _is_regular_index(self, index):
        """Heuristic regularity check on a DatetimeIndex/PeriodIndex.

        Returns True if successive deltas identical (after sorting) and length > 2.
            For PeriodIndex we assume regular (has freq) and return True.
        """
        if len(index) < 3:
            return True
        if isinstance(index, pd.PeriodIndex):
            return True
        if isinstance(index, pd.DatetimeIndex):
            deltas = index.to_series().diff().dropna().unique()
            return len(deltas) == 1

        # handle integer case
        is_int = pd.api.types.is_integer_dtype(index) and not index.hasnans
        if not is_int:
            return False
        iV = index.to_numpy()
        step = iV[1] - iV[0]
        is_arithmetic_progression = np.array_equal(
            iV, iV[0] + step * np.arange(len(iV))
        )
        return is_arithmetic_progression

    @staticmethod
    def _letter_from_index(idx) -> Optional[str]:  # rhs was str|None
        """
        Map DatetimeIndex/PeriodIndex frequency to one of 'A','Q','M','W','D'.

        Returns None if it can't be determined.
        """
        # PeriodIndex: use freqstr directly (e.g., 'A-DEC','Q-DEC','M','W-SUN','D')
        if isinstance(idx, pd.PeriodIndex):
            base = idx.freqstr.upper()
            # DatetimeIndex: use .freq if set, else infer, then to_offset(...).rule_code
        elif isinstance(idx, pd.DatetimeIndex):
            off = idx.freq or (
                to_offset(idx.inferred_freq) if idx.inferred_freq else None
            )
            if off is None:
                return None
            base = (getattr(off, "rule_code", None) or str(off)).upper()
        else:
            return None

        if base.startswith(("A", "AS", "Y", "YS")):
            return "A"  # or return "Y" if you prefer that label
        if base.startswith(("Q", "QS")):
            return "Q"
        if base.startswith("M"):
            return "M"
        if base.startswith(("W", "WE")):
            return "W"
        if base.startswith(("D", "B")):  # treat business day as daily if you want
            return "D"
        return None

    @staticmethod
    def _get_alt_date_range(n):
        return pd.date_range(start="1900-01-01", periods=n, freq="YE")

    def _get_alt_fh_values(self, fh):
        if not fh.is_relative:
            if (not isinstance(fh._values, pd.DatetimeIndex)) and (
                not pd.api.types.is_period_dtype(fh._values)
            ):
                if self._step > 0:
                    fh = ForecastingHorizon(
                        values=(fh._values - self._end) // self._step,
                        is_relative=True,
                        freq=1,
                    )
        return fh

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

        # one-time R session setup (suppress package startup chatter)
        if not hasattr(self, "_r_session_initialized"):
            robjects.r(
                "suppressPackageStartupMessages({options(verbose=FALSE);library(tidyverse);library(fpp3);library(tsibble);library(dplyr);library(fabletools);library(fable)})"
            )
            self._r_session_initialized = True

        # Work on a copy to avoid mutating caller objects (predict side-effect tests)
        if isinstance(Z, pd.DataFrame):
            Z = Z.copy()

        if isinstance(Z.index, pd.DatetimeIndex) or isinstance(Z.index, pd.PeriodIndex):
            freq = self._letter_from_index(Z.index)
        elif pd.api.types.is_integer_dtype(getattr(Z.index, "dtype", None)):
            Z.index = self._get_alt_date_range(len(Z.index))
            freq = "Y"
            self.int_index_to_annual = True
        else:
            raise ValueError(
                "Index must be of type DatetimeIndex or PeriodIndex (or integer)"
            )

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

        # Preserve full timestamp for sub-daily data; only coerce to date string
        # for (sub-)annual to daily frequencies we explicitly map.
        if isinstance(Z.index, pd.PeriodIndex):
            Z[date_col_name] = Z.index.to_timestamp()

        # determine whether we should strip time component (for Y/Q/M/W/D)
        strip_time = freq in {"A", "Y", "Q", "M", "W", "D"}
        if strip_time:
            Z[date_col_name] = Z[date_col_name].dt.strftime("%Y-%m-%d")

        # Convert the pandas DataFrame to an R data frame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(Z)

        # Define the R script to prepare the tsibble
        if is_regular:
            # aggregate to regular (>= daily) period indices
            freq_func_map = {
                "A": "lubridate::year",
                "Y": "lubridate::year",
                "Q": "tsibble::yearquarter",
                "M": "tsibble::yearmonth",
                "W": "tsibble::yearweek",
                "D": "as.Date",
            }
            if freq in freq_func_map and freq is not None:
                # Coarse frequency we explicitly support with aggregation helper
                freq_func = freq_func_map[freq]
                r_script = f"""
                r_data <- dplyr::as_tibble(r_data) |>
                    dplyr::mutate({date_col_name} = as.Date({date_col_name}),
                                  idx = {freq_func}({date_col_name})) |>
                    tsibble::as_tsibble(index = idx, regular = TRUE)
                r_data
                """
            else:
                # Fallback for sub-daily/unsupported frequencies (H,T,S,etc.)
                # Treat as irregular tsibble to avoid validate_interval issues.
                mutate_dt = f"{date_col_name} = as.POSIXct({date_col_name})"
                r_script = f"""
                r_data <- dplyr::as_tibble(r_data) |>
                    dplyr::mutate({mutate_dt}) |>
                    tsibble::as_tsibble(index = {date_col_name}, regular = FALSE)
                r_data
                """
        else:
            # irregular case - just pass through the (possibly POSIXct) timestamp
            if strip_time:
                date_coerce = f"{date_col_name} = as.Date({date_col_name})"
            else:
                date_coerce = f"{date_col_name} = as.POSIXct({date_col_name})"
            r_script = f"""
            r_data <- dplyr::as_tibble(r_data) |>
                dplyr::mutate({date_coerce}) |>
                tsibble::as_tsibble(index = {date_col_name}, regular = FALSE)
            r_data
            """

        # Execute the R script
        robjects.globalenv["r_data"] = r_data
        tsibble = robjects.r(r_script)

        return tsibble, Z[date_col_name]

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
        fitted_values_df = robjects.pandas2ri.rpy2py(fitted_values)
        fitted_values_df.index = self._y.index
        series = fitted_values_df[".fitted"].squeeze()
        series.name = self._y.name

        return series

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame (single column)
        fh : ignored (handled by base class)
        X : pd.DataFrame, optional
            Exogenous variables with same index as y.

        Returns
        -------
        self
        """
        # Ensure univariate y and determine external & internal target names
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "PyFableARIMA currently supports only univariate y; "
                    f"received {y.shape[1]} columns"
                )
            y_series = y.iloc[:, 0]
            external_name = y.columns[0]
        else:
            y_series = y
            external_name = y.name

        # Internal model name must be a valid R symbol; fall back if None
        model_target_name = external_name if external_name is not None else "y"
        self._orig_target_name = external_name
        self._model_target_name = model_target_name

        # Construct design matrix Z
        if X is not None:
            if not y_series.index.equals(X.index):
                raise ValueError("y and X must have the same index")
            y_df = y_series.to_frame(name=model_target_name)
            Z = pd.concat([y_df, X], axis=1)
            if self.formula is None:
                rhs = " + ".join(str(col) for col in X.columns)
                expr = f"{model_target_name} ~ {rhs}" if rhs else model_target_name
            else:
                expr = self.formula
        else:
            Z = y_series.to_frame(name=model_target_name)
            expr = model_target_name if self.formula is None else self.formula

        # Regularity pre-check if user claims regular; raise early if not
        if self.is_regular and not self._is_regular_index(y_series.index):
            raise ValueError(
                "PyFableARIMA: series marked is_regular=True but index is irregular. "
                "Set is_regular=False or resample to a regular frequency."
            )

        # Fit underlying R model (only proceed if regular claim consistent)
        r_tsibble, Z_date_col = self._custom_prepare_tsibble(
            Z, is_regular=self.is_regular
        )
        self._fit_auto_arima_ = self._custom_fit_arima(r_tsibble, expr)
        self._fit_index_ = y_series.index
        if self.int_index_to_annual:
            self._fit_index_alt_ = self._get_alt_date_range(len(y_series.index))
            self._start = y_series.index[0]
            self._end = y_series.index[-1]
            if len(y_series) > 1:
                self._step = y_series.index[1] - y_series.index[0]
            else:
                self._step = 0
        self._y = y_series
        self._resolved_formula = expr
        return self

    # ----------------------------------------------------------------------------
    def _convert_to_R_index(self, fh):
        """Convert the fh to an index that R handles."""
        # ensure ForecastingHorizon object
        if not isinstance(fh, ForecastingHorizon):
            raise ValueError("internal error: expect ForecastingHorizon")
        if fh.is_relative:
            raise ValueError("internal error: expected absolute ForecastingHorizon")

        if self.int_index_to_annual:
            iV = fh.to_relative(cutoff=self._y.index[-1]).to_numpy()
            fh_rel = ForecastingHorizon(iV, is_relative=True, freq="Y")
            fh = fh_rel.to_absolute(cutoff=self._fit_index_alt_[-1])

        R_index = fh.to_pandas()
        return R_index

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

        if X is not None:
            if not X.index.equals(fh.to_absolute_index()):
                raise ValueError("internal error: X.index and fh.index do not agree")

        l_fh_index = self._convert_to_R_index(fh)

        if X is not None:
            # Prepare exogenous features (copy to avoid caller mutation) or create dummy
            X = X.copy()
            X.index = l_fh_index
            a_tsibble, Z_date_col = self._custom_prepare_tsibble(
                X, is_regular=self.is_regular
            )
        else:
            dummy = pd.Series(0, index=l_fh_index)
            a_tsibble, Z_date_col = self._custom_prepare_tsibble(
                dummy, is_regular=self.is_regular
            )

        robjects.globalenv["fit_aut_arima"] = self._fit_auto_arima_
        robjects.globalenv["a_tsibble"] = a_tsibble

        # Default coverage if none provided
        if not coverage:
            coverage = [0.9]

        level_vector = robjects.FloatVector([c * 100 for c in coverage])
        robjects.globalenv["level_vec"] = level_vector

        # Build forecast call with optional warning suppression
        if self.verbose:
            forecast_call = "fc <- forecast(fit_aut_arima, new_data = a_tsibble)"
        else:
            forecast_call = (
                "withCallingHandlers({ fc <- forecast("
                "fit_aut_arima, new_data = a_tsibble) }, "
                "warning=function(w){ msg <- conditionMessage(w); "
                "if(grepl('contains no packages', msg) || "
                "grepl('irregular time series', msg)) "
                "invokeRestart('muffleWarning') })"
            )

        r_script = "\n".join(
            [
                "# packages already loaded once",
                forecast_call,
                "",
                "dist_col <- names(fc)[",
                'sapply(fc,function(x)inherits(x,"distribution"))][1]',
                "",
                "intervals_all <- purrr::map(level_vec, function(lv) {",
                "  dists <- purrr::map(fc[[dist_col]], function(x) hilo(x, lv))",
                "  df <- tibble::tibble(interval = dists) %>%",
                "    dplyr::mutate(",
                "      lower = purrr::map_dbl(interval, function(x) x$lower),",
                "      upper = purrr::map_dbl(interval, function(x) x$upper)",
                "    ) %>%",
                "    dplyr::select(lower, upper)",
                "  df",
                "})",
                "",
                'names(intervals_all) <- paste0("c", as.character(level_vec))',
                "",
                "intervals_named <- purrr::imap(intervals_all, function(df, name) {",
                '  dplyr::rename_with(df, ~ paste0(., "_", name))',
                "})",
                "",
                "intervals_df <- dplyr::bind_cols(intervals_named)",
                "fc_tbl <- dplyr::bind_cols(tibble::as_tibble(fc), intervals_df)",
                "fc_tbl",
            ]
        )

        forecasts = robjects.r(r_script)
        forecasts_df = robjects.pandas2ri.rpy2py(forecasts)

        # Identify forecast mean column
        forecast_column = next((c for c in forecasts_df.columns if ".mean" in c), None)
        if forecast_column is None:
            forecast_column = forecasts_df.select_dtypes(include="number").columns[0]

        forecast_values = forecasts_df[forecast_column].values
        external_name = self._orig_target_name

        orig_index = fh.to_absolute_index()
        # y_pred must keep original training series name (can be None)
        y_pred = pd.Series(forecast_values, index=orig_index, name=external_name)
        # intervals require numeric 0 when original name None per estimator expectations
        interval_variable_name = external_name if external_name is not None else 0

        # Build prediction intervals DataFrame
        intervals = {}
        y_varname = interval_variable_name
        for col in forecasts_df.columns:
            col_str = str(col)
            if col_str.startswith("lower_") or col_str.startswith("upper_"):
                parts = col_str.split("_")
                bound = parts[0]
                level = float(parts[1][1:]) / 100  # remove leading 'c'
                intervals[(y_varname, level, bound)] = forecasts_df[col].values

        pred_int = pd.DataFrame(intervals, index=orig_index)
        pred_int.columns = pd.MultiIndex.from_tuples(
            pred_int.columns, names=["variable", "coverage", "bound"]
        )
        return y_pred, pred_int

    # -------------------------------------------------------------------------

    @staticmethod
    def _get_fh_consecutive(fh, cutoff, freq=None):
        """Confirm consecutive forecast horizons.

        ARIMA models require consecutive forecasting horizons.
        Returns consecutive absolute fh

        Return an absolute consecutive FH (1..n from cutoff).
        If `freq` is provided and FH is datetime/period-like, the returned FH's
        pandas index will have that `freq` (also for length-1).
        """
        fh = check_fh(fh)
        if fh.is_relative:
            raise ValueError("expected an absolute ForecastingHorizon")

        # --- Ensure FH has a frequency if it's datetime/period-like ---
        abs_idx = fh.to_absolute_index()

        if isinstance(abs_idx, (pd.DatetimeIndex, pd.PeriodIndex)):
            # pick an effective frequency: caller > index.freq > infer
            eff_freq = (
                freq
                or getattr(abs_idx, "freq", None)
                or pd.infer_freq(abs_idx)  # will be None for len==1 or irregular
            )
            if eff_freq is None:
                raise ValueError(
                    "You must pass a `freq` (e.g. 'D','MS','A-DEC'): the absolute "
                    "FH index has no freq and it cannot be inferred."
                )
            # Re-wrap FH with the effective freq so fh._freq is set for to_relative
            fh = ForecastingHorizon(abs_idx, is_relative=False, freq=eff_freq)
        else:
            # this may handle int64 indexes
            cutoff = int(cutoff)  # confirm some additional assumptions for this case?

        # relative steps (1-based)
        steps = np.asarray(fh.to_relative(cutoff=cutoff).to_numpy(), dtype=int)
        steps_sorted = np.sort(steps)
        is_consecutive = (
            len(steps_sorted) > 0
            and steps_sorted[0] == 1
            and np.all(np.diff(steps_sorted) == 1)
            and len(np.unique(steps)) == len(steps)  # no duplicates
        )

        abs_idx = fh.to_absolute_index()

        # If already consecutive, optionally enforce/attach freq on the absolute index
        if is_consecutive:
            if freq is None:
                return fh  # nothing to change
            # Rebuild absolute index with freq (works even for length-1)
            if isinstance(abs_idx, pd.DatetimeIndex):
                idx2 = pd.DatetimeIndex(abs_idx, freq=freq)  # validates compatibility
                return ForecastingHorizon(idx2, is_relative=False)
            elif isinstance(abs_idx, pd.PeriodIndex):
                idx2 = pd.PeriodIndex(abs_idx, freq=freq)
                return ForecastingHorizon(idx2, is_relative=False)
            else:
                # integer-like: no freq concept
                return fh

        # Not consecutive -> build 1..n anchored at cutoff
        n = int(steps_sorted[-1])

        # Determine frequency if needed for datetime/period-like indices
        if isinstance(abs_idx, (pd.DatetimeIndex, pd.PeriodIndex)) and freq is None:
            freq = getattr(abs_idx, "freq", None) or pd.infer_freq(abs_idx)
            if freq is None:
                raise ValueError(
                    "Cannot infer freq from abs FH; supply `freq` explicitly."
                )

        # Build absolute consecutive index with the desired freq
        if isinstance(abs_idx, pd.DatetimeIndex):
            cut = pd.Timestamp(cutoff)
            off = to_offset(freq)  # raises if invalid
            idx2 = pd.DatetimeIndex([cut + i * off for i in range(1, n + 1)], freq=freq)
        elif isinstance(abs_idx, pd.PeriodIndex):
            base = pd.Period(cutoff, freq=freq)
            idx2 = pd.PeriodIndex([base + i for i in range(1, n + 1)], freq=freq)
        else:
            # integer-like horizon
            cut = int(cutoff)
            idx2 = pd.Index([cut + i for i in range(1, n + 1)], dtype=abs_idx.dtype)

        return ForecastingHorizon(idx2, is_relative=False)

    # -----------------------------------------------------------------------------
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

        freq = None

        if not isinstance(fh, ForecastingHorizon):
            fh = ForecastingHorizon(fh)

        cutoff = self._y.index[-1]
        abs_fh = fh.to_absolute_index(cutoff=cutoff)

        # Get fitted values for in-sample support and also for freq
        fitted_values = self.get_fitted_values()
        if isinstance(fitted_values.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            freq = getattr(fitted_values.index, "freq", None) or pd.infer_freq(
                fitted_values.index
            )

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
            abs_fh_consecutive = PyFableARIMA._get_fh_consecutive(
                fh_out, cutoff, freq=freq
            )
            y_pred_out, _ = self._predict_special(abs_fh_consecutive, X=X)
            y_pred_out = y_pred_out.loc[out_sample_index].copy()
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
        if not isinstance(fh, ForecastingHorizon):
            fh = ForecastingHorizon(fh)

        cutoff = self._y.index[-1]
        abs_fh = fh.to_absolute(cutoff=cutoff)
        freq = getattr(self._y.index, "freq", None)

        abs_fh_consecutive = PyFableARIMA._get_fh_consecutive(
            abs_fh, cutoff=cutoff, freq=freq
        )

        _, pred_int = self._predict_special(abs_fh_consecutive, X=X, coverage=coverage)

        pred_int_sub = pred_int.loc[abs_fh.to_absolute_index()].copy()

        return pred_int_sub

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
        # Return the parameters needed to construct the estimator
        params0 = {}
        params1 = {
            "ic": "bic",
            "selection_metric": None,
            "stepwise": False,
            "greedy": False,
        }
        return [params0, params1]

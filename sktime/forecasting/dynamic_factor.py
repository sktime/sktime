# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements DynamicFactor Model as interface to statsmodels."""
import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters import _StatsModelsAdapter

__all__ = ["DynamicFactor"]
__author__ = ["Ris-Bali", "lbventura"]


class DynamicFactor(_StatsModelsAdapter):
    """Dynamic Factor Forecaster.

    Direct interface for `statsmodels.tsa.statespace.dynamic_factor`

    Parameters
    ----------
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of vector autoregression followed by factors.
    error_cov_type : {'scalar','diagonal','unstructured'} ,default = 'diagonal'
        The structure of the covariance matrix of the observation error term, where
        "unstructured" puts no restrictions on the matrix, "diagonal" requires it
        to be any diagonal matrix (uncorrelated errors), and "scalar" requires it
        to be a scalar times the identity matrix.
    error_order : int , default = 0
        The order of the vector autoregression followed by the observation error
        component. Default is None, corresponding to white noise errors.
    error_var : bool , default = False , optional
        Whether or not to model the errors jointly via a vector autoregression,
         rather than as individual autoregression.
    enforce_stationarity : bool default = True
        Whether or not to model the AR parameters to enforce stationarity in the
         autoregressive component of the model.
    start_params :array_like ,default = None
        Initial guess of the solution for the loglikelihood maximization.
    transformed : bool, default = True
        Whether or not start_params is already transformed.
    includes_fixed : bool , default = False
        If parameters were previously fixed with the fix_params method, this argument
         describes whether or not start_params also includes the fixed parameters,
          in addition to the free parameters.
    cov_type : {'opg','oim','approx','robust','robust_approx','none'},default = 'opg'
        'opg' for the outer product of gradient estimator
        'oim' for the observed information matrix estimator, calculated
        using the method of Harvey (1989)
        'approx' for the observed information matrix estimator, calculated using
         a numerical approximation of the Hessian matrix.
        'robust' for an approximate (quasi-maximum likelihood) covariance matrix
         that may be valid even in the presence of some misspecifications.
          Intermediate calculations use the 'oim' method.
        'robust_approx' is the same as 'robust' except that the intermediate
        calculations use the 'approx' method.
        'none' for no covariance matrix calculation
    cov_kwds :dict or None , default = None
        'approx_complex_step' : bool, optional - If True, numerical approximations are
         computed using complex-step methods. If False, numerical approximations are
         computed using finite difference methods. Default is True.
        'approx_centered' : bool, optional - If True, numerical approximations computed
        using finite difference methods use a centered approximation. Default is False.
    method : str , 'lbfgs'
        'newton' for Newton-Raphson
        'nm' for Nelder-Mead
        'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        'lbfgs' for limited-memory BFGS with optional box constraints
        'powell' for modified Powell's method
        'cg' for conjugate gradient
        'ncg' for Newton-conjugate gradient
        'basinhopping' for global basin-hopping solver
    maxiter : int , optional ,default = 50
        The maximum number of iterations to perform.
    full_output : bool , default = 1
        Set to True to have all available output in the
        Results object's mle_retvals attribute.
        The output is dependent on the solver.
    disp : bool ,   default = 5
        Set to True to print convergence messages.
    callback : callable callback(xk) , default = None
        Called after each iteration, as callback(xk),
        where xk is the current parameter vector.
    return_params : bool ,default = False
        Whether or not to return only the array of maximizing parameters.
    optim_score : {'harvey','approx'} , default = None
        The method by which the score vector is calculated. 'harvey' uses the method
        from Harvey (1989), 'approx' uses either finite difference or
        complex step differentiation depending upon the value of optim_complex_step,
        and None uses the built-in gradient approximation of the optimizer.
    optim_complex_step : bool , default = True
        Whether or not to use complex step differentiation
        when approximating the score; if False, finite difference approximation is used.
    optim_hessian : {'opg','oim','approx'} , default = None
        'opg' uses outer product of gradients, 'oim' uses the information
        matrix formula from Harvey (1989), and 'approx' uses numerical approximation.
    low_memory : bool , default = False
        If set to True, techniques are applied to substantially reduce memory usage.
        If used, some features of the results object will not be available
        (including smoothed results and in-sample prediction),
        although out-of-sample forecasting is possible.

    See Also
    --------
    statsmodels.tsa.statespace.dynamic_factor.DynamicFactor
    statsmodels.tsa.statespace.dynamic_factor.DynamicFactorResults

    References
    ----------
    .. [1] Lütkepohl, Helmut. 2007. New Introduction to Multiple Time Series Analysis.
    Berlin: Springer.

    Examples
    --------
    >>> from sktime.utils._testing.series import _make_series
    >>> from sktime.forecasting.dynamic_factor import DynamicFactor
    >>> y = _make_series(n_columns=4)
    >>> forecaster = DynamicFactor()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    DynamicFactor(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    _tags = {
        "scitype:y": "multivariate",
        "ignores-exogeneous-X": False,
        "handles-missing-data": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        k_factors=1,
        factor_order=1,
        error_cov_type="diagonal",
        error_order=0,
        error_var=False,
        enforce_stationarity=True,
        start_params=None,
        transformed=True,
        includes_fixed=False,
        cov_type=None,
        cov_kwds=None,
        method="lbfgs",
        maxiter=50,
        full_output=False,
        disp=False,
        callback=None,
        return_params=False,
        optim_score=None,
        optim_complex_step=None,
        optim_hessian=None,
        flags=None,
        low_memory=False,
    ):
        # Model Params
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_cov_type = error_cov_type
        self.error_order = error_order
        self.error_var = error_var
        self.enforce_stationarity = enforce_stationarity

        # Fit Params
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.return_params = return_params
        self.optim_score = optim_score
        self.optim_complex_step = optim_complex_step
        self.optim_hessian = optim_hessian
        self.flags = flags
        self.low_memory = low_memory

        super().__init__()

    def _predict(self, fh, X):
        """Make forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : pd.Series
            Returns series of predicted values.
        """
        # statsmodels requires zero-based indexing starting at the
        # beginning of the training series when passing integers
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]

        y_pred = self._fitted_forecaster.predict(start=start, end=end, exog=X)

        # statsmodels forecasts all periods from start to end of forecasting
        # horizon, but only return given time points in forecasting horizon

        if "int" in (self._y.index[0]).__class__.__name__:  # Rather fishy solution
            y_pred.index = np.arange(
                start + self._y.index[0], end + self._y.index[0] + 1
            )
        return y_pred.loc[fh.to_absolute_index(self.cutoff)]

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
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
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
        if type(coverage) is not list:
            coverage_list = [coverage]
        else:
            coverage_list = coverage

        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]
        steps = end - len(self._y) + 1
        ix = fh.to_indexer(self.cutoff)

        model = self._fitted_forecaster

        df_list = []
        # generate the forecasts for each alpha/coverage
        for coverage in coverage_list:
            alpha = 1 - coverage

            y_pred = model.get_forecast(steps=steps, exog=X).conf_int(alpha=alpha)

            y_pred = y_pred.iloc[ix]

            y_pred.rename(
                columns={orig_col: orig_col + f" {coverage}" for orig_col in y_pred},
                inplace=True,
            )
            df_list.append(y_pred)

        # concatenate the predictions for different values of alpha
        predictions_df = pd.concat(df_list, axis=1)

        # all the code below is just boilerplate for getting the correct columns
        mod_var_list = list(map(lambda x: str(x).replace(" ", ""), model.data.ynames))

        rename_list = [
            var_name + " " + str(coverage) + " " + bound
            for coverage in coverage_list
            for bound in ["lower", "upper"]
            for var_name in mod_var_list
        ]

        predictions_df.columns = rename_list

        final_ord = [
            var_name + " " + str(coverage) + " " + bound
            for var_name in mod_var_list
            for coverage in coverage_list
            for bound in ["lower", "upper"]
        ]

        predictions_df = predictions_df[final_ord]
        cols = [col_name.split(" ") for col_name in predictions_df.columns]

        predictions_df_2 = pd.DataFrame(
            predictions_df.values, columns=pd.MultiIndex.from_tuples(cols)
        )

        final_columns = [
            [col_name, float(coverage), bound]
            for col_name in model.data.ynames
            for coverage in predictions_df_2.columns.get_level_values(1).unique()
            for bound in predictions_df_2.columns.get_level_values(2).unique()
        ]

        predictions_df_3 = pd.DataFrame(
            predictions_df_2.values, columns=pd.MultiIndex.from_tuples(final_columns)
        )
        predictions_df_3.index = fh.to_absolute_index(self.cutoff)

        return predictions_df_3

    def _fit_forecaster(self, y, X=None):
        """Fit to training data.

        Parameters
        ----------
        y:pd.Series
          Target time series to which forecaster is fit.
        X:pd.DataFrame , optional (default=None)
          Exogenous variables
        """
        from statsmodels.tsa.statespace.dynamic_factor import (
            DynamicFactor as _DynamicFactor,
        )

        self._forecaster = _DynamicFactor(
            endog=y,
            exog=X,
            k_factors=self.k_factors,
            factor_order=self.factor_order,
            error_order=self.error_order,
            error_var=self.error_var,
            error_cov_type=self.error_cov_type,
            enforce_stationarity=self.enforce_stationarity,
        )

        self._fitted_forecaster = self._forecaster.fit(
            start_params=self.start_params,
            transformed=self.transformed,
            includes_fixed=self.includes_fixed,
            cov_type=self.cov_type,
            cov_kwds=self.cov_kwds,
            method=self.method,
            maxiter=self.maxiter,
            full_output=self.full_output,
            disp=self.disp,
            callback=self.callback,
            return_params=self.return_params,
            optim_score=self.optim_score,
            optim_complex_step=self.optim_complex_step,
            optim_hessian=self.optim_hessian,
            flags=self.flags,
            low_memory=self.low_memory,
        )

    def summary(self):
        """Get a summary of the fitted forecaster."""
        return self._fitted_forecaster.summary()

    def simulate(
        self,
        nsimulations,
        measurement_shocks=None,
        state_shocks=None,
        initial_state=None,
        anchor=None,
        repetitions=None,
        X=None,
        extend_model=None,
        extend_kwargs=None,
        transformed=True,
        includes_fixed=False,
        **kwargs,
    ):
        r"""Simulate a new time series following the state space model.

        Taken from original statsmodels implementation.

        Parameters
        ----------
        nsimulations : int
            The number of observations to simulate. If the model is time-invariant
            this can be any number. If the model is time-varying, then this number
            must be less than or equal to the number.
        measurement_shocks : array_like , optional
            If specified, these are the shocks to the measurement equation,.
            If unspecified, these are automatically generated using a
            pseudo-random number generator. If specified, must be shaped
            nsimulations x k_endog, where k_endog is the same as in
            the state space model.
        state_shocks : array_like , optional
            If specified, these are the shocks to the state equation, .
            If unspecified, these are automatically generated using a
            pseudo-random number generator. If specified, must be shaped
            nsimulations x k_posdef where k_posdef is the same as in the
            state space model.
        initial_state : array_like , optional
            If specified, this is the initial state vector to use in
            simulation,which should be shaped (k_states x 1), where
            k_states is the same as in the state space model.
            If unspecified, but the model has been initialized,
            then that initialization is used.
            This must be specified if anchor is anything
            other than "start" or 0.
        anchor : int,str,or datetime , optional
            Starting point from which to begin the simulations; type depends
            on the index of the given endog model.
            Two special cases are the strings 'start' and 'end',
            which refer to starting at the beginning and end of the sample,
            respectively. If a date/time index was provided to the model,
            then this argument can be a date string to parse or a datetime type.
            Otherwise, an integer index should be given. Default is 'start'.
        repetitions : int , optional
            Number of simulated paths to generate. Default is 1 simulated path

        Returns
        -------
        simulated_obs : ndarray
            An array of simulated observations. If repetitions=None, then it will
            be shaped (nsimulations x k_endog) or (nsimulations,) if k_endog=1.
            Otherwise it will be shaped (nsimulations x k_endog x repetitions).
            If the model was given Pandas input then the output will be a Pandas object.
            If k_endog > 1 and repetitions is not None, then the output will be a Pandas
            DataFrame that has a MultiIndex for the columns, with the first level
            containing the names of the endog variables and the second level
            containing the repetition number.
        """
        return self._fitted_forecaster.simulate(
            nsimulations=nsimulations,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            initial_state=initial_state,
            anchor=anchor,
            repetitions=repetitions,
            exog=X,
            extend_model=extend_model,
            extend_kwargs=extend_kwargs,
            transformed=transformed,
            includes_fixed=includes_fixed,
            **kwargs,
        )

    def plot_diagnostics(
        self,
        variable=0,
        lags=10,
        fig=None,
        figsize=None,
        truncate_endog_names=24,
        auto_ylims=False,
        bartlett_confint=False,
        acf_kwargs=None,
    ):
        """Diagnostic plots for standardized residuals of one endogenous variable.

        Parameters
        ----------
        variable : int , optional
            Index of the endogenous variable for which the diagnostic
            plots should be created. Default is 0.
        lags : int , optional
            Number of lags to include in the correlogram. Default is 10.
        fig : Figure , optional
            If given, subplots are created in this figure instead of in
            a new figure. Note that the 2x2 grid will be created in the
            provided figure using fig.add_subplot().
        figsize : tuple , optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).
        auto_ylims : bool , optional
            If True, adjusts automatically the y-axis limits to ACF values.
        bartlett_confint : bool , default = True
            Confidence intervals for ACF values are generally placed at 2
            standard errors around r_k. The formula used for standard error
            depends upon the situation. If the autocorrelations are being
            used to test for randomness of residuals as part of the ARIMA routine,
            the standard errors are determined assuming the residuals are white noise.
            The approximate formula for any lag is that standard error
            of each r_k = 1/sqrt(N).For the ACF of raw data, the standard error at
            a lag k is found as if the right model was an MA(k-1).
            This allows the possible interpretation that if all autocorrelations
            past a certain lag are within the limits,
            the model might be an MA of order defined by
            the last significant autocorrelation.
            In this case, a moving average model is assumed for the data
            and the standard errors for the confidence intervals should be generated
             using Bartlett's formula.
        acf_kwargs : dict , optional
            Optional dictionary of keyword arguments that are directly
            passed on to the correlogram Matplotlib plot produced by plot_acf().

        Returns
        -------
        Figure
            Figure instance with diagnostic plots.
        """
        self._fitted_forecaster.plot_diagonistics(
            variable=variable,
            lags=lags,
            fig=fig,
            figsize=figsize,
            truncate_endog_names=truncate_endog_names,
            auto_ylims=auto_ylims,
            bartlett_confint=bartlett_confint,
            acf_kwargs=acf_kwargs,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str , default = "default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params :dict or list of dict , default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"k_factors": 1, "factor_order": 1}
        params2 = {"maxiter": 25, "low_memory": True}

        return [params1, params2]

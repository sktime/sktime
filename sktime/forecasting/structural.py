# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Wraps the UnobservedComponents (state space) model from statsmodels."""

__all__ = ["UnobservedComponents"]
__author__ = ["juanitorduz"]

import pandas as pd
from statsmodels.tsa.statespace.structural import (
    UnobservedComponents as _UnobservedComponents,
)

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class UnobservedComponents(_StatsModelsAdapter):
    r"""Wrapper class of the UnobservedComponents model from statsmodels.

    Input parameters and doc-stringsare taken from the original implementation.

    Parameters
    ----------
    level : {bool, str}, optional
        Whether or not to include a level component. Default is False. Can also
        be a string specification of the level / trend component.
    trend : bool, optional
        Whether or not to include a trend component. Default is False. If True,
        `level` must also be True.
    seasonal : {int, None}, optional
        The period of the seasonal component, if any. Default is None.
    freq_seasonal : {list[dict], None}, optional.
        Whether (and how) to model seasonal component(s) with trig. functions.
        If specified, there is one dictionary for each frequency-domain
        seasonal component.  Each dictionary must have the key, value pair for
        'period' -- integer and may have a key, value pair for
        'harmonics' -- integer. If 'harmonics' is not specified in any of the
        dictionaries, it defaults to the floor of period/2.
    cycle : bool, optional
        Whether or not to include a cycle component. Default is False.
    autoregressive : {int, None}, optional
        The order of the autoregressive component. Default is None.
    irregular : bool, optional
        Whether or not to include an irregular component. Default is False.
    stochastic_level : bool, optional
        Whether or not any level component is stochastic. Default is False.
    stochastic_trend : bool, optional
        Whether or not any trend component is stochastic. Default is False.
    stochastic_seasonal : bool, optional
        Whether or not any seasonal component is stochastic. Default is True.
    stochastic_freq_seasonal : list[bool], optional
        Whether or not each seasonal component(s) is (are) stochastic.  Default
        is True for each component.  The list should be of the same length as
        freq_seasonal.
    stochastic_cycle : bool, optional
        Whether or not any cycle component is stochastic. Default is False.
    damped_cycle : bool, optional
        Whether or not the cycle component is damped. Default is False.
    cycle_period_bounds : tuple, optional
        A tuple with lower and upper allowed bounds for the period of the
        cycle. If not provided, the following default bounds are used:
        (1) if no date / time information is provided, the frequency is
        constrained to be between zero and :math:`\pi`, so the period is
        constrained to be in [0.5, infinity].
        (2) If the date / time information is provided, the default bounds
        allow the cyclical component to be between 1.5 and 12 years; depending
        on the frequency of the endogenous variable, this will imply different
        specific bounds.
    mle_regression : bool, optional
        Whether or not to estimate regression coefficients by maximum
        likelihood as one of hyperparameters. Default is True.
        If False, the regression coefficients are estimated by recursive OLS,
        included in the state vector.
    use_exact_diffuse : bool, optional
        Whether or not to use exact diffuse initialization for non-stationary
        states. Default is False (in which case approximate diffuse
        initialization is used).
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
    transformed : bool, optional
        Whether or not `start_params` is already transformed. Default is
        True.
    includes_fixed : bool, optional
        If parameters were previously fixed with the `fix_params` method,
        this argument describes whether or not `start_params` also includes
        the fixed parameters, in addition to the free parameters. Default
        is False.
    cov_type : str, optional
        The `cov_type` keyword governs the method for calculating the
        covariance matrix of parameter estimates. Can be one of:
        - 'opg' for the outer product of gradient estimator
        - 'oim' for the observed information matrix estimator, calculated
            using the method of Harvey (1989)
        - 'approx' for the observed information matrix estimator,
            calculated using a numerical approximation of the Hessian matrix.
        - 'robust' for an approximate (quasi-maximum likelihood) covariance
            matrix that may be valid even in the presence of some
            misspecifications. Intermediate calculations use the 'oim'
            method.
        - 'robust_approx' is the same as 'robust' except that the
            intermediate calculations use the 'approx' method.
        - 'none' for no covariance matrix calculation.

        Default is 'opg' unless memory conservation is used to avoid
        computing the loglikelihood values for each observation, in which
        case the default is 'approx'.
    cov_kwds : dict or None, optional
        A dictionary of arguments affecting covariance matrix computation.
        **opg, oim, approx, robust, robust_approx**
        - 'approx_complex_step' : bool, optional - If True, numerical
            approximations are computed using complex-step methods. If False,
            numerical approximations are computed using finite difference
            methods. Default is True.
        - 'approx_centered' : bool, optional - If True, numerical
            approximations computed using finite difference methods use a
            centered approximation. Default is False.
    method : str, optional
        The `method` determines which solver from `scipy.optimize`
        is used, and it can be chosen from among the following strings:
        - 'newton' for Newton-Raphson
        - 'nm' for Nelder-Mead
        - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - 'lbfgs' for limited-memory BFGS with optional box constraints
        - 'powell' for modified Powell's method
        - 'cg' for conjugate gradient
        - 'ncg' for Newton-conjugate gradient
        - 'basinhopping' for global basin-hopping solver
        The explicit arguments in `fit` are passed to the solver,
        with the exception of the basin-hopping solver. Each
        solver has several optional arguments that are not the same across
        solvers. See the notes section below (or scipy.optimize) for the
        available arguments and for the list of explicit arguments that the
        basin-hopping solver supports.
    maxiter : int, optional
        The maximum number of iterations to perform.
    full_output : bool, optional
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    disp : bool, optional
        Set to True to print convergence messages. Default is 0.
    callback : callable callback(xk), optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    return_params : bool, optional
        Whether or not to return only the array of maximizing parameters.
        Default is False.
    optim_score : {'harvey', 'approx'} or None, optional
        The method by which the score vector is calculated. 'harvey' uses
        the method from Harvey (1989), 'approx' uses either finite
        difference or complex step differentiation depending upon the
        value of `optim_complex_step`, and None uses the built-in gradient
        approximation of the optimizer. Default is None. This keyword is
        only relevant if the optimization method uses the score.
    optim_complex_step : bool, optional
        Whether or not to use complex step differentiation when
        approximating the score; if False, finite difference approximation
        is used. Default is True. This keyword is only relevant if
        `optim_score` is set to 'harvey' or 'approx'.
    optim_hessian : {'opg','oim','approx'}, optional
        The method by which the Hessian is numerically approximated. 'opg'
        uses outer product of gradients, 'oim' uses the information
        matrix formula from Harvey (1989), and 'approx' uses numerical
        approximation. This keyword is only relevant if the
        optimization method uses the Hessian matrix.
    low_memory : bool, optional
        If set to True, techniques are applied to substantially reduce
        memory usage. If used, some features of the results object will
        not be available (including smoothed results and in-sample
        prediction), although out-of-sample forecasting is possible.
        Default is False.
    random_state : int, RandomState instance or None, optional ,
        default=None – If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.

    See Also
    --------
    statsmodels.tsa.statespace.structural.UnobservedComponents
    statsmodels.tsa.statespace.structural.UnobservedComponentsResults

    References
    ----------
    .. [1] Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric
       and statistical modeling with python.” Proceedings of the 9th Python
       in Science Conference. 2010.

    .. [2] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.structural import UnobservedComponents
    >>> y = load_airline()
    >>> forecaster = UnobservedComponents(level='local linear trend')
    >>> forecaster.fit(y)
    UnobservedComponents(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        "capability:pred_int": True,
        "handles-missing-data": False,
        "ignores-exogeneous-X": False,
    }

    def __init__(
        self,
        level=False,
        trend=False,
        seasonal=None,
        freq_seasonal=None,
        cycle=False,
        autoregressive=None,
        irregular=False,
        stochastic_level=False,
        stochastic_trend=False,
        stochastic_seasonal=True,
        stochastic_freq_seasonal=None,
        stochastic_cycle=False,
        damped_cycle=False,
        cycle_period_bounds=None,
        mle_regression=True,
        use_exact_diffuse=False,
        start_params=None,
        transformed=True,
        includes_fixed=False,
        cov_type=None,
        cov_kwds=None,
        method="lbfgs",
        maxiter=50,
        full_output=1,
        disp=0,
        callback=None,
        return_params=False,
        optim_score=None,
        optim_complex_step=None,
        optim_hessian=None,
        flags=None,
        low_memory=False,
        random_state=None,
    ):
        # Model params
        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal
        self.cycle = cycle
        self.autoregressive = autoregressive
        self.irregular = irregular
        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        self.stochastic_freq_seasonal = stochastic_freq_seasonal
        self.stochastic_cycle = stochastic_cycle
        self.damped_cycle = damped_cycle
        self.cycle_period_bounds = cycle_period_bounds
        self.mle_regression = mle_regression
        self.use_exact_diffuse = use_exact_diffuse

        # Fit params
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

        super(UnobservedComponents, self).__init__(random_state=random_state)

    def _fit_forecaster(self, y, X=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        """
        self._forecaster = _UnobservedComponents(
            endog=y,
            exog=X,
            level=self.level,
            trend=self.trend,
            seasonal=self.seasonal,
            freq_seasonal=self.freq_seasonal,
            cycle=self.cycle,
            autoregressive=self.autoregressive,
            irregular=self.irregular,
            stochastic_level=self.stochastic_level,
            stochastic_trend=self.stochastic_trend,
            stochastic_seasonal=self.stochastic_seasonal,
            stochastic_freq_seasonal=self.stochastic_freq_seasonal,
            stochastic_cycle=self.stochastic_cycle,
            damped_cycle=self.damped_cycle,
            cycle_period_bounds=self.cycle_period_bounds,
            mle_regression=self.mle_regression,
            use_exact_diffuse=self.use_exact_diffuse,
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

    def _predict_interval(self, fh, X=None, coverage=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.PredictionResults.summary_frame
        """
        valid_indices = fh.to_absolute(self.cutoff).to_pandas()

        start, end = valid_indices[[0, -1]]
        prediction_results = self._fitted_forecaster.get_prediction(
            start=start, end=end, exog=X
        )
        pred_int = pd.DataFrame()
        for c in coverage:
            alpha = 1 - c
            pred_statsmodels = prediction_results.summary_frame(alpha=alpha)
            pred_int[(c, "lower")] = pred_statsmodels["mean_ci_lower"].loc[
                valid_indices
            ]
            pred_int[(c, "upper")] = pred_statsmodels["mean_ci_upper"].loc[
                valid_indices
            ]
        index = pd.MultiIndex.from_product([["Coverage"], coverage, ["lower", "upper"]])
        pred_int.columns = index
        return pred_int

    def summary(self):
        """Get a summary of the fitted forecaster.

        This is the same as the implementation in statsmodels:
        https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_structural_harvey_jaeger.html
        """
        return self._fitted_forecaster.summary()

    def simulate(
        self,
        nsimulations,
        X=None,
        measurement_shocks=None,
        state_shocks=None,
        initial_state=None,
        anchor=None,
        repetitions=None,
        **kwargs
    ):
        r"""Simulate a new time series following the state space model.

        Taken from the original statsmodels implementation.

        Parameters
        ----------
        nsimulations : int
            The number of observations to simulate. If the model is
            time-invariant this can be any number. If the model is
            time-varying, then this number must be less than or equal to the
            number of observations.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        measurement_shocks : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_shocks : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state : array_like, optional
            If specified, this is the initial state vector to use in
            simulation, which should be shaped (`k_states` x 1), where
            `k_states` is the same as in the state space model. If unspecified,
            but the model has been initialized, then that initialization is
            used. This must be specified if `anchor` is anything other than
            "start" or 0 (or else you can use the `simulate` method on a
            results object rather than on the model object).
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample, and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEResults

        Returns
        -------
        simulated_obs : ndarray
            An array of simulated observations. If `repetitions=None`, then it
            will be shaped (nsimulations x k_endog) or (nsimulations,) if
            `k_endog=1`. Otherwise it will be shaped
            (nsimulations x k_endog x repetitions). If the model was given
            Pandas input then the output will be a Pandas object. If
            `k_endog > 1` and `repetitions` is not None, then the output will
            be a Pandas DataFrame that has a MultiIndex for the columns, with
            the first level containing the names of the `endog` variables and
            the second level containing the repetition number.
        """
        return self._fitted_forecaster.simulate(
            nsimulations=nsimulations,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            initial_state=initial_state,
            anchor=anchor,
            repetitions=repetitions,
            exog=X,
            **kwargs
        )

    def plot_diagnostics(
        self,
        variable=0,
        lags=10,
        fig=None,
        figsize=None,
        truncate_endog_names=24,
    ):
        """Diagnostic plots for standardized residuals.

        Taken from the original statsmodels implementation.

        Parameters
        ----------
        variable : int, optional
            Index of the endogenous variable for which the diagnostic plots
            should be created. Default is 0.
        lags : int, optional
            Number of lags to include in the correlogram. Default is 10.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the 2x2 grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEResults

        Returns
        -------
        Figure
            Figure instance with diagnostic plots.
        """
        self._fitted_forecaster.plot_diagnostics(
            variable=variable,
            lags=lags,
            fig=fig,
            figsize=figsize,
            truncate_endog_names=truncate_endog_names,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"level": "local level"}

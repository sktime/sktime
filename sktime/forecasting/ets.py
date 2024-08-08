# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements automatic and manually exponential time series smoothing models."""

__author__ = ["hyang1996"]
__all__ = ["AutoETS"]

import warnings
from itertools import product

import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class AutoETS(_StatsModelsAdapter):
    """ETS models with both manual and automatic fitting capabilities.

    Manual (fixed parameter) use (``auto=False``, default) is a direct interface
    to ``statsmodels`` ``ETSModel`` [2]_,
    while automated tuning (``auto=True``) is an adaptation of the R version of ets
    [3]_,
    on top of ``statsmodels`` ``ETSModel``.

    The first parameters are direct interfaces to the ``statsmodels`` parameters
    (from ``error`` to ``return_params``) [2]_.

    The remaining parameters are adaptations of the parameters of R ets
    (``auto`` to ``additive_only``) [3]_,
    and are used for automatic model selection.

    Parameters
    ----------
    error : str, default="add"
        The error model. Takes one of "add" or "mul". Ignored if auto=True.
    trend : str or None, default=None
        The trend component model. Takes one of "add", "mul", or None. Ignored if
        auto=True.
    damped_trend : bool, default=False
        Whether or not an included trend component is damped. Ignored if auto=True.
    seasonal : str or None, default=None
        The seasonality model. Takes one of "add", "mul", or None. Ignored if auto=True.
    sp : int, default=1
        The number of periods in a complete seasonal cycle for seasonal
        (Holt-Winters) models. For example, 4 for quarterly data with an
        annual cycle or 7 for daily data with a weekly cycle. Required if
        ``seasonal`` is not None.
    initialization_method : str, default='estimated'
        Method for initialization of the state space model. One of:

        * 'estimated' (default)
        * 'heuristic'
        * 'known'

        If 'known' initialization is used, then ``initial_level`` must be
        passed, as well as ``initial_trend`` and ``initial_seasonal`` if
        applicable.
        'heuristic' uses a heuristic based on the data to estimate initial
        level, trend, and seasonal state. 'estimated' uses the same heuristic
        as initial guesses, but then estimates the initial states as part of
        the fitting process.  Default is 'estimated'.
    initial_level : float or None, default=None
        The initial level component. Only used if initialization is 'known'.
    initial_trend : float or None, default=None
        The initial trend component. Only used if initialization is 'known'.
    initial_seasonal : array_like or None, default=None
        The initial seasonal component. An array of length ``seasonal_periods``.
        Only used if initialization is 'known'.
    bounds : dict or None, default=None
        A dictionary with parameter names as keys and the respective bounds
        intervals as values (lists/tuples/arrays).
        The available parameter names are, depending on the model and
        initialization method:

        * "smoothing_level"
        * "smoothing_trend"
        * "smoothing_seasonal"
        * "damping_trend"
        * "initial_level"
        * "initial_trend"
        * "initial_seasonal.0", ..., "initial_seasonal.<m-1>"

        The default option is ``None``, in which case the traditional
        (nonlinear) bounds as described in [1]_ are used.
    start_params : array_like or None, default=None
        Initial values for parameters that will be optimized. If this is
        ``None``, default values will be used.
        The length of this depends on the chosen model. This should contain
        the parameters in the following order, skipping parameters that do
        not exist in the chosen model.

        * ``smoothing_level`` (alpha)
        * ``smoothing_trend`` (beta)
        * ``smoothing_seasonal`` (gamma)
        * ``damping_trend`` (phi)

        If ``initialization_method`` was set to ``'estimated'`` (the
        default), additionally, the parameters

        * ``initial_level`` (:math:`l_{-1}`)
        * ``initial_trend`` (:math:`l_{-1}`)
        * ``initial_seasonal.0`` (:math:`s_{-1}`)
        * ``initial_seasonal.<m-1>`` (:math:`s_{-m}`)

        also have to be specified.
    maxiter : int, default=1000
        The maximum number of iterations to perform.
    full_output : bool, default=True
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    disp : bool, default=False
        Set to True to print convergence messages.
    callback : callable callback(xk) or None, default=None
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    return_params : bool, default=False
        Whether or not to return only the array of maximizing parameters.
    auto : bool, default=False
        Set True to enable automatic model selection. If auto=True, then error,
        trend, seasonal and damped_trend are ignored.
    information_criterion : str, default="aic"
        Information criterion to be used in model selection. One of:

        * "aic"
        * "bic"
        * "aicc"

    allow_multiplicative_trend : bool, default=False
        If True, models with multiplicative trend are allowed when
        searching for a model. Otherwise, the model space excludes them.
    restrict : bool, default=True
        If True, the models with infinite variance will not be allowed.
    additive_only : bool, default=False
        If True, will only consider additive models.
    ignore_inf_ic: bool, default=True
        If True models with negative infinity Information Criterion
        (aic, bic, aicc) will be ignored.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel for automatic model fitting.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, optional ,
        default=None - If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
       principles and practice*, 3rd edition, OTexts: Melbourne,
       Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
    .. [2] Statsmodels ETS:

        https://www.statsmodels.org/stable/_modules/statsmodels/tsa/exponential_smoothing/ets.html#ETSModel
    .. [3] R Version of ETS:
        https://www.rdocumentation.org/packages/forecast/versions/8.12/topics/ets

    See Also
    --------
    StatsForecastAutoETS

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.ets import AutoETS
    >>> y = load_airline()
    >>> forecaster = AutoETS(auto=True, n_jobs=-1, sp=12)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    AutoETS(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    _fitted_param_names = ("aic", "aicc", "bic", "hqic")
    _tags = {
        # packaging info
        # --------------
        "authors": ["hyang1996"],
        "maintainers": ["hyang1996"],
        "python_dependencies": ["statsmodels", "joblib"],
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
    }

    def __init__(
        self,
        error="add",
        trend=None,
        damped_trend=False,
        seasonal=None,
        sp=1,
        initialization_method="estimated",
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
        bounds=None,
        dates=None,
        freq=None,
        missing="none",
        start_params=None,
        maxiter=1000,
        full_output=True,
        disp=False,
        callback=None,
        return_params=False,
        auto=False,
        information_criterion="aic",
        allow_multiplicative_trend=False,
        restrict=True,
        additive_only=False,
        ignore_inf_ic=True,
        n_jobs=None,
        random_state=None,
    ):
        # Model params
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.sp = sp
        self.initialization_method = initialization_method
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.bounds = bounds
        self.dates = dates
        self.freq = freq
        self.missing = missing

        # Fit params
        self.start_params = start_params
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.return_params = return_params
        self.information_criterion = information_criterion
        self.auto = auto
        self.allow_multiplicative_trend = allow_multiplicative_trend
        self.restrict = restrict
        self.additive_only = additive_only
        self.ignore_inf_ic = ignore_inf_ic
        self.n_jobs = n_jobs

        super().__init__(random_state=random_state)

        if self.auto:
            # If auto=True, check if trend, damped_trend, seasonal, or error are not set
            # to default values
            if any([trend, damped_trend, seasonal]) or error != "add":
                warnings.warn(
                    "The user-specified parameters provided alongside auto=True in "
                    "AutoETS may not be respected. The AutoETS function "
                    "automatically selects the best model based on the "
                    "information criterion, ignoring the error, trend, "
                    "seasonal, and damped_trend parameters when auto=True"
                    " is set. Please ensure that your intended behavior"
                    " aligns with the automatic model selection.",
                    stacklevel=2,
                )

    def _fit_forecaster(self, y, X=None):
        from joblib import Parallel, delayed
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel as _ETSModel

        # Select model automatically
        if self.auto:
            # Initialise parameter ranges
            if self.additive_only:
                error_range = ["add"]
            else:
                if (y > 0).all():
                    error_range = ["add", "mul"]
                else:
                    warnings.warn(
                        "Warning: time series is not strictly positive, "
                        "multiplicative components are omitted",
                        stacklevel=2,
                    )
                    error_range = ["add"]

            if self.allow_multiplicative_trend and (y > 0).all():
                trend_range = ["add", "mul", None]
            else:
                trend_range = ["add", None]

            if self.sp <= 1 or self.sp is None:
                seasonal_range = [None]
            elif (y > 0).all():
                seasonal_range = ["add", "mul", None]
            else:
                seasonal_range = ["add", None]

            damped_range = [True, False]

            # Check information criterion input
            if self.information_criterion not in ["aic", "bic", "aicc"]:
                raise ValueError(
                    "information criterion must either be aic, bic or aicc"
                )

            # Fit model, adapted from:
            # https://github.com/robjhyndman/forecast/blob/master/R/ets.R

            # Initialise iterator
            def _iter(error_range, trend_range, seasonal_range, damped_range):
                for error, trend, seasonal, damped in product(
                    error_range, trend_range, seasonal_range, damped_range
                ):
                    if trend is None and damped:
                        continue

                    if self.restrict:
                        if error == "add" and (trend == "mul" or seasonal == "mul"):
                            continue
                        if error == "mul" and trend == "mul" and seasonal == "add":
                            continue
                        if self.additive_only and (
                            error == "mul" or trend == "mul" or seasonal == "mul"
                        ):
                            continue

                    yield error, trend, seasonal, damped

            # Fit function
            def _fit(error, trend, seasonal, damped):
                _forecaster = _ETSModel(
                    y,
                    error=error,
                    trend=trend,
                    damped_trend=damped,
                    seasonal=seasonal,
                    seasonal_periods=self.sp,
                    initialization_method=self.initialization_method,
                    initial_level=self.initial_level,
                    initial_trend=self.initial_trend,
                    initial_seasonal=self.initial_seasonal,
                    bounds=self.bounds,
                    dates=self.dates,
                    freq=self.freq,
                    missing=self.missing,
                )
                _fitted_forecaster = _forecaster.fit(
                    start_params=self.start_params,
                    maxiter=self.maxiter,
                    full_output=self.full_output,
                    disp=self.disp,
                    callback=self.callback,
                    return_params=self.return_params,
                )
                return _forecaster, _fitted_forecaster

            # Fit models
            _fitted_results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit)(error, trend, seasonal, damped)
                for error, trend, seasonal, damped in _iter(
                    error_range, trend_range, seasonal_range, damped_range
                )
            )

            # Store IC values for each model in a list
            # Ignore infinite likelihood models if ignore_inf_ic is True
            _ic_list = []
            for result in _fitted_results:
                ic = getattr(result[1], self.information_criterion)
                if self.ignore_inf_ic and np.isinf(ic):
                    _ic_list.append(np.nan)
                else:
                    _ic_list.append(ic)

            # Select best model based on information criterion
            if np.all(np.isnan(_ic_list)) or len(_ic_list) == 0:
                # if all models have infinite IC raise an error
                raise ValueError(
                    "None of the fitted models have finite %s"
                    % self.information_criterion
                )
            else:
                # Get index of best model
                _index = np.nanargmin(_ic_list)

            # Update best model
            self._forecaster = _fitted_results[_index][0]
            self._fitted_forecaster = _fitted_results[_index][1]

        else:
            self._forecaster = _ETSModel(
                y,
                error=self.error,
                trend=self.trend,
                damped_trend=self.damped_trend,
                seasonal=self.seasonal,
                seasonal_periods=self.sp,
                initialization_method=self.initialization_method,
                initial_level=self.initial_level,
                initial_trend=self.initial_trend,
                initial_seasonal=self.initial_seasonal,
                bounds=self.bounds,
                dates=self.dates,
                freq=self.freq,
                missing=self.missing,
            )

            self._fitted_forecaster = self._forecaster.fit(
                start_params=self.start_params,
                maxiter=self.maxiter,
                full_output=self.full_output,
                disp=self.disp,
                callback=self.callback,
                return_params=self.return_params,
            )

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
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]

        # statsmodels forecasts all periods from start to end of forecasting
        # horizon, but only return given time points in forecasting horizon
        valid_indices = fh.to_absolute_index(self.cutoff)

        y_pred = self._fitted_forecaster.predict(start=start, end=end)
        y_pred.name = self._y.name
        return y_pred.loc[valid_indices]

    @staticmethod
    def _extract_conf_int(prediction_results, alpha) -> pd.DataFrame:
        """Construct confidence interval at specified ``alpha`` for each timestep.

        Parameters
        ----------
        prediction_results : PredictionResults
            results class, as returned by ``self._fitted_forecaster.get_prediction``
        alpha : float
            one minus nominal coverage

        Returns
        -------
        pd.DataFrame
            confidence intervals at each timestep

            The dataframe must have at least two columns ``lower`` and ``upper``, and
            the row indices must be integers relative to ``self.cutoff``. Order of
            columns do not matter, and row indices must be a superset of relative
            integer horizon of ``fh``.
        """
        conf_int = prediction_results.pred_int(alpha=alpha)
        conf_int.columns = ["lower", "upper"]

        return conf_int

    def summary(self):
        """Get a summary of the fitted forecaster.

        This is the same as the implementation in statsmodels:
        https://www.statsmodels.org/dev/examples/notebooks/generated/ets.html
        """
        return self._fitted_forecaster.summary()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict
        """
        params = [
            # default setting, non-auto
            {},
            # "auto-ets"
            # TODO: uncomment following line while fixing #4591
            # {"sp": 2, "auto": True},
            # ets (non-auto) with some non-default parameters
            {"information_criterion": "bic", "trend": "add", "damped_trend": True},
        ]

        return params

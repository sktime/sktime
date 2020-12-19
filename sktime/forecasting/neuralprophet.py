#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter"]
__all__ = ["NeuralProphet"]


from sktime.forecasting.base._adapters import _ProphetAdapter
from sktime.utils.check_imports import _check_soft_dependencies


_check_soft_dependencies("neuralprophet")


class NeuralProphet(_ProphetAdapter):
    """NeuralProphet forecaster by wrapping neuralprophet.
    Parameters
    ----------
    freq: String of DatetimeIndex frequency, defaults to freq of y. See here
        for possible values:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    add_seasonality: Dict with args for NeuralProphet.add_seasonality().
        Dict can have the following keys/values:
            name: string name of the seasonality component.
            period: float number of days in one period.
            fourier_order: int number of Fourier components to use.
            prior_scale: optional float prior scale for this component.
            mode: optional 'additive' or 'multiplicative'
            condition_name: string name of the seasonality condition.
    add_country_holidays: Dict with args for
        NeuralProphet.add_country_holidays(). Dict can have the
            following keys/values: country_name: Name of the country,
            like 'UnitedStates' or 'US'
    growth (str): ['off', 'linear'] to specify
        no trend or a linear trend. Note: 'discontinuous' setting is actually
        not a trend per se. only use if you know what you do.
    changepoints (np.array): List of dates at which to include potential
        changepoints. If not specified, potential changepoints are
        selected automatically.
    n_changepoints (int): Number of potential changepoints to include.
        Changepoints are selected uniformly from the first `changepoint_range`
        proportion of the history. Not used if input `changepoints`
        is supplied. If `changepoints` is not supplied.
    changepoints_range (float): Proportion of history in which trend
        changepoints will be estimated. Defaults to 0.8 for the first 80%.
        Not used if `changepoints` is specified.
    trend_reg (float): Parameter modulating the flexibility of the automatic
        changepoint selection. Large values (~1-100) will limit the
        variability of changepoints. Small values (~0.001-1.0) will allow
        changepoints to change faster. default: 0 will fully fit a trend
        to each segment.
    trend_reg_threshold (bool, float): Allowance for trend to change
        without regularization.
        True: Automatically set to a value that leads to a smooth trend.
        False: All changes in changepoints are regularized

    ## Seasonality Config
    yearly_seasonality (bool, int): Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier/linear terms
        to generate.
    weekly_seasonality (bool, int): Fit monthly seasonality.
        Can be 'auto', True, False, or a number of Fourier/linear terms
        to generate.
    daily_seasonality (bool, int): Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier/linear terms
        to generate.
    seasonality_mode (str): 'additive' (default) or 'multiplicative'.
    seasonality_reg (float): Parameter modulating the strength of the
        seasonality model. Smaller values (~0.1-1) allow the model to
        fit larger seasonal fluctuations, larger values (~1-100) dampen
        the seasonality. default: None, no regularization

    ## AR Config
    n_lags (int): Previous time series steps to include in
        auto-regression. Aka AR-order.
    ar_sparsity (float): [0-1], how much sparsity to enduce in
        the AR-coefficients.
        Should be around (# nonzero components) / (AR order), eg. 3/100 = 0.03

    ## Model Config
    n_forecasts (int): Number of steps ahead of prediction time step
        to forecast.
    num_hidden_layers (int): number of hidden layer to include in
        AR-Net. defaults to 0.
    d_hidden (int): dimension of hidden layers of the AR-Net. Ignored if
        num_hidden_layers == 0.

    ## Train Config
    learning_rate (float): Maximum learning rate setting for 1cycle
        policy scheduler.
        default: None: Automatically sets the learning_rate based on a
        learning rate range test. For manual values, try values ~0.001-10.
    epochs (int): Number of epochs (complete iterations over dataset)
        to train model.
        default: None: Automatically sets the number of epochs based
            on dataset size. For best results also leave batch_size to None.
        For manual values, try ~5-500.
    batch_size (int): Number of samples per mini-batch.
        default: None: Automatically sets the batch_size based on dataset size.
        For best results leave epochs to None. For manual values, try ~1-512.
    loss_func (str, torch.nn.modules.loss._Loss): Type of loss to
        use ['Huber', 'MAE', 'MSE']
    train_speed (int, float) a quick setting to speed up or slow down model
        fitting [-3, -2, -1, 0, 1, 2, 3]
        potentially useful when under-, over-fitting, or simply in a hurry.
        applies epochs *= 2**-train_speed, batch_size *= 2**train_speed,
        learning_rate *= 2**train_speed,
        default None: equivalent to 0.

    ## Data config
    normalize (str): Type of normalization to apply to the time series.
        options: ['auto', 'soft', 'off', 'minmax, 'standardize']
        default: 'auto' uses 'minmax' if variable is binary, else 'soft'
        'soft' scales minimum to 0.1 and the 90th quantile to 0.9
    impute_missing (bool): whether to automatically impute missing dates/values
        imputation follows a linear method up to 10 missing values, more
        are filled with trend.

    References
    ----------
    http://neuralprophet.com
    https://github.com/ourownstory/neural_prophet

    """

    def __init__(
        self,
        # Args due to wrapping
        freq=None,
        add_seasonality=None,
        add_country_holidays=None,
        # Args of neuralprophet
        growth="linear",
        changepoints=None,
        n_changepoints=5,
        changepoints_range=0.8,
        trend_reg=0,
        trend_reg_threshold=False,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        seasonality_mode="additive",
        seasonality_reg=0,
        n_forecasts=1,
        n_lags=0,
        num_hidden_layers=0,
        d_hidden=None,
        ar_sparsity=None,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="Huber",
        train_speed=None,
        normalize="auto",
        impute_missing=True,
    ):
        self.freq = freq
        self.add_seasonality = add_seasonality
        self.add_country_holidays = add_country_holidays

        self.growth = (growth,)
        self.changepoints = (changepoints,)
        self.n_changepoints = (n_changepoints,)
        self.changepoints_range = (changepoints_range,)
        self.trend_reg = (trend_reg,)
        self.trend_reg_threshold = (trend_reg_threshold,)
        self.yearly_seasonality = (yearly_seasonality,)
        self.weekly_seasonality = (weekly_seasonality,)
        self.daily_seasonality = (daily_seasonality,)
        self.seasonality_mode = (seasonality_mode,)
        self.seasonality_reg = (seasonality_reg,)
        self.n_forecasts = (n_forecasts,)
        self.n_lags = (n_lags,)
        self.num_hidden_layers = (num_hidden_layers,)
        self.d_hidden = (d_hidden,)
        self.ar_sparsity = (ar_sparsity,)
        self.learning_rate = (learning_rate,)
        self.epochs = (epochs,)
        self.batch_size = (batch_size,)
        self.loss_func = (loss_func,)
        self.train_speed = (train_speed,)
        self.normalize = (normalize,)
        self.impute_missing = (impute_missing,)

        # import inside method to avoid hard dependency
        from neuralprophet import NeuralProphet as _NeuralProphet

        self._ModelClass = _NeuralProphet

        super(NeuralProphet, self).__init__()

    def _instantiate_model(self):
        self._forecaster = self._ModelClass(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoints_range=self.changepoints_range,
            trend_reg=self.trend_reg,
            trend_reg_threshold=self.trend_reg_threshold,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_reg=self.seasonality_reg,
            n_forecasts=self.n_forecasts,
            n_lags=self.n_lags,
            num_hidden_layers=self.num_hidden_layers,
            d_hidden=self.d_hidden,
            ar_sparsity=self.ar_sparsity,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            loss_func=self.loss_func,
            train_speed=self.train_speed,
            normalize=self.normalize,
            impute_missing=self.impute_missing,
        )
        return self

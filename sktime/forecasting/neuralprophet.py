#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements NeuralProphet forecaster by wrapping neuralprophet."""

__author__ = ["vedantag17"]
__all__ = ["NeuralProphet"]


from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base.adapters import _ProphetAdapter


class NeuralProphet(_ProphetAdapter):
    """NeuralProphet forecaster by wrapping NeuralProphet algorithm [1]_.

    Direct interface to NeuralProphet, using the sktime interface.
    All hyper-parameters are exposed via the constructor.

    Data can be passed in one of the sktime compatible formats.
    Like Prophet, NeuralProphet also supports integer/range and period index:
    * integer/range index is interpreted as days since Jan 1, 2000
    * ``PeriodIndex`` is converted using the ``pandas`` method ``to_timestamp``

    Parameters
    ----------
    freq: str, default=None
        A DatetimeIndex frequency.
    growth: str, default="linear"
        Type of trend, 'linear', 'discontinuous', or 'off'
    changepoints: list, default=None
        List of dates at which to include potential changepoints
    n_changepoints: int, default=10
        Number of potential changepoints to include
    changepoints_range: float, default=0.8
        Proportion of history in which trend changepoints are automatically selected
    yearly_seasonality: bool or int, default=True
        Whether to include yearly seasonality or number of Fourier terms
    weekly_seasonality: bool or int, default=True
        Whether to include weekly seasonality or number of Fourier terms
    daily_seasonality: bool or int, default=False
        Whether to include daily seasonality or number of Fourier terms
    seasonality_mode: str, default='additive'
        'additive' or 'multiplicative'
    seasonality_reg: float, default=0
        Regularization strength for seasonality
    custom_seasonalities: list or None, default=None
        List of dicts for custom seasonality components to add
    add_country_holidays: dict or None, default=None
        Dict with args for adding country holidays
        Dict should contain 'country_name' as key
    holidays: pd.DataFrame, default=None
        DataFrame with holiday dates and features
    holidays_mode: str, default='additive'
        'additive' or 'multiplicative'
    holidays_reg: float, default=0
        Regularization strength for holidays
    trend_reg: float, default=0
        Regularization strength for trend
    trend_reg_threshold: bool, default=False
        Threshold for trend regularization
    learning_rate: float, default=None
        Maximum learning rate (applicable in quasi-Newton optimization)
    epochs: int, default=None
        Number of training epochs
    batch_size: int, default=None
        Number of samples per mini-batch
    loss_func: str, default="Huber"
        Type of loss to use (e.g., "Huber", "MSE", "MAE", etc.)
    alpha: float, default=0.05
        Width of the uncertainty intervals
    uncertainty_samples: int, default=1000
        Number of samples for estimating uncertainty intervals
    verbose: bool, default=False
        Whether to print status information during fitting

    References
    ----------
    .. [1] https://github.com/ourownstory/neural_prophet

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.neuralprophet import NeuralProphet
    >>> # NeuralProphet requires data with a pandas.DatetimeIndex
    >>> y = load_airline().to_timestamp(freq='M')
    >>> forecaster = NeuralProphet(
    ...     seasonality_mode='multiplicative',
    ...     n_changepoints=12,
    ...     add_country_holidays={'country_name': 'US'},
    ...     yearly_seasonality=True)
    >>> forecaster.fit(y)
    NeuralProphet(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    def __init__(
        self,
        # Args due to wrapping
        freq=None,
        add_seasonality=None,
        custom_seasonalities=None,
        add_country_holidays=None,
        growth="linear",
        changepoints=None,
        n_changepoints=10,
        changepoints_range=0.8,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        seasonality_reg=0,
        holidays=None,
        holidays_mode="additive",
        holidays_reg=0,
        trend_reg=0,
        trend_reg_threshold=False,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="Huber",
        alpha=DEFAULT_ALPHA,
        uncertainty_samples=1000,
        verbose=False,
    ):
        self.freq = freq
        self.add_seasonality = add_seasonality
        self.custom_seasonalities = custom_seasonalities
        self.add_country_holidays = add_country_holidays

        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoints_range = changepoints_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_reg = seasonality_reg
        self.holidays = holidays
        self.holidays_mode = holidays_mode
        self.holidays_reg = holidays_reg
        self.trend_reg = trend_reg
        self.trend_reg_threshold = trend_reg_threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.alpha = alpha
        self.uncertainty_samples = uncertainty_samples
        self.verbose = verbose

        super().__init__()

        # import inside method to avoid hard dependency
        from neuralprophet import NeuralProphet as _NeuralProphet

        self._ModelClass = _NeuralProphet

    def _instantiate_model(self):
        """Instantiate the NeuralProphet model."""
        self._forecaster = self._ModelClass(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoints_range=self.changepoints_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_reg=self.seasonality_reg,
            trend_reg=self.trend_reg,
            trend_reg_threshold=self.trend_reg_threshold,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            loss_func=self.loss_func,
        )

        return self

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
            {
                "n_changepoints": 0,
                "yearly_seasonality": False,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "epochs": 5,
                "uncertainty_samples": 10,
                "verbose": False,
            },
            {
                "growth": "linear",
                "n_changepoints": 5,
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "seasonality_mode": "multiplicative",
                "epochs": 10,
                "loss_func": "MAE",
                "uncertainty_samples": 50,
            },
            {
                "growth": "discontinuous",
                "n_changepoints": 3,
                "yearly_seasonality": 10,
                "weekly_seasonality": 5,
                "seasonality_mode": "additive",
                "seasonality_reg": 0.1,
                "trend_reg": 0.05,
                "epochs": 15,
                "batch_size": 32,
                "uncertainty_samples": 100,
            }
        ]
        return params

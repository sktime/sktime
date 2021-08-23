# -*- coding: utf-8 -*-

"""VECM Forecaster."""

__all__ = ["VECM"]
__author__ = ["Taiwo Owoseni"]


from statsmodels.tsa.vector_ar.vecm import VECM as _VECM
from sktime.forecasting.base.adapters import _StatsModelsAdapter
from sktime.forecasting.base._base import DEFAULT_ALPHA

import numpy as np


class VECM(_StatsModelsAdapter):
    """
    A restricted VAR for nonstationary series that are cointegrated.

    Parameters
    ----------
    dates: np.ndarray, optional (default=None)
        An array-like object of datetime objects
    freq: str, optional (default=None)
        The frequency of the time-series.
        Pandas offset or ‘B’, ‘D’, ‘W’, ‘M’, ‘A’, or ‘Q’.
    missing: str, optional (default='none')
        A string specifying if data is missing
    k_ar_diff: int, (default = 1 )
        Lags in the VAR representation.
    coint_rank: int, (default = 1)
        Cointegration rank
    deterministic: str, (default = "nc")
        "nc" - no deterministic terms
        "co" - constant outside the cointegration relation
        "ci" - constant within the cointegration relation
        "lo" - linear trend outside the cointegration relation
        "li" - linear trend within the cointegration relation
        Combinations of these are possible
        (e.g. "cili" or "colo" for linear trend with intercept).
        When using a constant term you have to choose
        whether you want to restrict it
        to the cointegration relation (i.e. "ci")
        or leave it unrestricted (i.e. "co").
        Don’t use both "ci" and "co".
        The same applies for "li" and "lo" when using a linear term
    seasons: int, (default=0)
        Number of periods in a seasonal cycle. 0 means no seasons.
    first_season: int, (default=0)
        Season of the first observation.


    Example
    -------
    >>> from sktime.forecasting.vecm import VECM
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> index = pd.date_range(start="2005", end="2006-12", freq="M")
    >>> df = pd.DataFrame(np.random.randint(0, 100, size=(23, 2)),
    ... columns=list("AB"),
    ... index=pd.PeriodIndex(index))
    >>> train, test = temporal_train_test_split(df)
    >>> sktime_model = VECM()
    >>> fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    >>> sktime_model.fit(train)
    VECM()
    >>> fc2 = sktime_model.predict(fh=fh)
    """

    _fitted_param_names = ()
    _tags = {
        "scitype:y": "multivariate",
        "y_inner_mtype": "pd.DataFrame",
        "univariate-only": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(
        self,
        dates=None,
        freq=None,
        missing="none",
        k_ar_diff=1,
        coint_rank=1,
        deterministic="nc",
        seasons=0,
        first_season=0,
    ):
        # Model params
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season

        super(VECM, self).__init__()

    def _fit_forecaster(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """
        self._forecaster = _VECM(
            y,
            dates=self.dates,
            freq=self.freq,
            missing=self.missing,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic,
            seasons=self.seasons,
            first_season=self.first_season,
        )
        self._fitted_forecaster = self._forecaster.fit()
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Wrap Statmodel's VAR forecast method.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        return_pred_int : bool, optional (default=False)
        alpha : int or list, optional (default=0.95)

        Returns
        -------
        y_pred : np.ndarray
            Returns series of predicted values.
        """
        fh_int = fh.to_absolute_int(self._y.index[0], self._y.index[-1])
        y_pred = self._fitted_forecaster.predict(steps=fh_int[-1])
        new_arr = []
        for i in fh:
            new_arr.append(y_pred[i - 1])
        return np.array(new_arr)

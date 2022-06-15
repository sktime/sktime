# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""VECM Forecaster."""

__all__ = ["VECM"]
__author__ = ["thayeylolu", "AurumnPegasus"]

import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM as _VECM
from statsmodels.tsa.vector_ar.vecm import VECMResults as _VECMResults

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class VECM(_StatsModelsAdapter):
    r"""
    A VECM model, or Vector Error Correction Model, is a restricted.

    VAR model used for nonstationary series that are cointegrated.r

    Parameters
    ----------
    dates : array_like of datetime, optional
        See :class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel` for more
        information.
    freq : str, optional
        See :class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel` for more
        information.
    missing : str, optional
        See :class:`statsmodels.base.model.Model` for more information.
    k_ar_diff : int
        Number of lagged differences in the model. Equals :math:`k_{ar} - 1` in
        the formula above.
    coint_rank : int
        Cointegration rank, equals the rank of the matrix :math:`\\Pi` and the
        number of columns of :math:`\\alpha` and :math:`\\beta`.
    deterministic : str {``"n"``, ``"co"``, ``"ci"``, ``"lo"``, ``"li"``}
        * ``"n"`` - no deterministic terms
        * ``"co"`` - constant outside the cointegration relation
        * ``"ci"`` - constant within the cointegration relation
        * ``"lo"`` - linear trend outside the cointegration relation
        * ``"li"`` - linear trend within the cointegration relation

        Combinations of these are possible (e.g. ``"cili"`` or ``"colo"`` for
        linear trend with intercept). When using a constant term you have to
        choose whether you want to restrict it to the cointegration relation
        (i.e. ``"ci"``) or leave it unrestricted (i.e. ``"co"``). Do not use
        both ``"ci"`` and ``"co"``. The same applies for ``"li"`` and ``"lo"``
        when using a linear term. See the Notes-section for more information.
    seasons : int, default: 0
        Number of periods in a seasonal cycle. 0 means no seasons.
    first_season : int, default: 0
        Season of the first observation.
    method : str {"ml"}, default: "ml"
        Estimation method to use. "ml" stands for Maximum Likelihood.
    exog_coint : a scalar (float), 1D ndarray of size nobs,
        2D ndarray/pd.DataFrame of size (any, neqs)
        Deterministic terms inside the cointegration relation.
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    #  when changing scitype:y to multivariate or both:
    #   y_inner_mtype should be changed to pd.DataFrame
    # other tags are "safe defaults" which can usually be left as-is
    _tags = {
        "scitype:y": "multivariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": True,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": True,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
    }
    #  in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__ (avoid this if possible)

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        alpha,
        beta,
        gamma,
        sigma_u,
        dates=None,
        freq=None,
        missing="none",
        k_ar_diff=1,
        coint_rank=1,
        deterministic="n",
        seasons=0,
        first_season=0,
        method="ml",
        exog_coint=None,
        exog_coint_fc=None,
        delta_y_1_T=None,
        y_lag1=None,
        delta_x=None,
        model=None,
        names=None,
    ):

        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season
        self.method = method
        self.exog_coint = exog_coint
        self.exog_coint_fc = exog_coint_fc
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_u = sigma_u
        self.delta_y_1_T = delta_y_1_T
        self.y_lag1 = y_lag1
        self.delta_x = delta_x
        self.model = model
        self.names = names

        super(VECM, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """
        Fit forecaster to training data.

        Wrapper for statsmodel's VECM (_VECM) fit method

        Parameters
        ----------
        y : pd.DataFrame, guaranteed to have 2 or more columns
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self._forecaster = _VECM(
            endog=y,
            exog=X,
            exog_coint=self.exog_coint,
            dates=self.dates,
            freq=self.freq,
            missing=self.missing,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic,
            seasons=self.seasons,
            first_season=self.first_season,
        )

        self._fitted_forecaster = self._forecaster.fit(method=self.method)
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Wrapper for statsmodel's VECM (_VECM) predict method

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        self._predictor = _VECMResults(
            endog=self._y,
            exog=self._X,
            exog_coint=self.exog_coint,
            k_ar=self.k_ar_diff,
            coint_rank=self.coint_rank,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            sigma_u=self.sigma_u,
            deterministic=self.deterministic,
            seasons=self.seasons,
            first_season=self.first_season,
            delta_y_1_T=self.delta_y_1_T,
            y_lag1=self.y_lag1,
            delta_x=self.delta_x,
            model=self.model,
            names=self.names,
            dates=self.dates,
        )

        fh_int = fh.to_absolute_int(self._y.index[0], self._y.index[-1])
        steps = fh_int[-1]
        forecast = self._predictor.predict(
            steps=steps, alpha=self.alpha, exog_fc=X, exog_coint_fc=self.exog_coint_fc
        )
        new_arr = []
        for i in fh:
            new_arr.append(forecast[i - 1])
        return np.array(new_arr)

# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""VECM Forecaster."""


__all__ = ["VECM"]
__author__ = ["thayeylolu", "AurumnPegasus"]

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM as _VECM

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
    exog_coint_fc : a scalar (float), 1D ndarray of size nobs,
        2D ndarray/pd.DataFrame of size (any, neqs)
        Forcasted value of exog_coint

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
    >>> _ = sktime_model.fit(train, fh=fh)
    >>> fc2 = sktime_model.predict(fh=fh)
    """

    _tags = {
        "scitype:y": "multivariate",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "univariate-only": False,
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
    }

    def __init__(
        self,
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

        super(VECM, self).__init__()

    def _fit(self, y, fh=None, X=None):
        """
        Fit forecaster to training data.

        Wrapper for statsmodel's VECM (_VECM) fit method

        Parameters
        ----------
        y : pd.DataFrame, guaranteed to have 2 or more columns
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon
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
        """
        Forecast time series at future horizon.

        Wrapper for statsmodel's VECM (_VECM) predict method

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
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
        y_pred_outsample = None
        y_pred_insample = None
        exog_fc = X.values if X is not None else None
        fh_int = fh.to_relative(self.cutoff)

        # out-sample prediction
        if fh_int.max() > 0:
            y_pred_outsample = self._fitted_forecaster.predict(
                steps=fh_int[-1],
                exog_fc=exog_fc,
                exog_coint_fc=self.exog_coint_fc,
            )

        # in-sample prediction by means of residuals
        if fh_int.min() <= 0:

            # .resid returns np.ndarray
            # both values need to be pd DataFrame for subtraction
            y_pred_insample = self._y - pd.DataFrame(self._fitted_forecaster.resid)
            y_pred_insample = y_pred_insample.values

        if y_pred_insample is not None and y_pred_outsample is not None:
            y_pred = np.concatenate([y_pred_outsample, y_pred_insample], axis=0)
        else:
            y_pred = (
                y_pred_insample if y_pred_insample is not None else y_pred_outsample
            )

        index = fh.to_absolute(self.cutoff)
        index.name = self._y.index.name
        y_pred = pd.DataFrame(
            y_pred[fh.to_indexer(self.cutoff), :],
            index=fh.to_absolute(self.cutoff),
            columns=self._y.columns,
        )

        return y_pred

    def _predict_interval(self, fh, X=None, coverage=None):
        """
        Compute/return prediction quantiles for a forecast.

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
        exog_fc = X.values if X is not None else None
        fh_oos = fh.to_out_of_sample(self.cutoff)
        var_names = (
            self._y.index.name
            if self._y.index.name is not None
            else self._y.columns.values
        )
        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        # pred_int = pd.DataFrame(index=int_idx)

        for c in coverage:
            alpha = 1 - c
            _, y_lower, y_upper = self._fitted_forecaster.predict(
                steps=fh_oos[-1],
                exog_fc=exog_fc,
                exog_coint_fc=self.exog_coint_fc,
                alpha=alpha,
            )
            values = []
            for v_idx in range(len(var_names)):
                values.append(y_lower[0][v_idx])
                values.append(y_upper[0][v_idx])
                # pred_int.loc[(var_names[v_idx], c, "lower"), :] = (y_lower[0][v_idx])
                # pred_int.loc[(var_names[v_idx], c, "upper"), :] = (y_upper[0][v_idx])
        pred_int = pd.DataFrame(
            [values], index=fh.to_absolute(self.cutoff), columns=int_idx
        )

        return pred_int

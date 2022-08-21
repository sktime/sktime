# -*- coding: utf-8 -*-
"""Tests the ARDL model."""
__author__ = ["kcc-lion"]

from numpy.testing import assert_allclose
from statsmodels.datasets import danish_data, grunfeld, longley
from statsmodels.tsa.ardl import ARDL as _ARDL
from statsmodels.tsa.ardl import ardl_select_order as _ardl_select_order

from sktime.datasets import load_macroeconomic
from sktime.forecasting.ardl import ARDL
from sktime.forecasting.base import ForecastingHorizon


def test_against_statsmodels():
    """Compare sktime's ARDL interface with statsmodels ARDL."""
    # data
    data = longley.load_pandas().data
    oos = data.iloc[-5:, :]
    data = data.iloc[:-5, :]
    y = data.TOTEMP
    X = data[["GNPDEFL", "GNP"]]
    X_oos = oos[["GNPDEFL", "GNP"]]
    # fit
    sm_ardl = _ARDL(y, 2, X, {"GNPDEFL": 1, "GNP": 2}, trend="c")
    res = sm_ardl.fit()
    ardl_sktime = ARDL(lags=2, order={"GNPDEFL": 1, "GNP": 2}, trend="c")
    ardl_sktime.fit(y=y, X=X, fh=None)
    # predict
    fh = ForecastingHorizon([1, 2, 3])
    start, end = X.shape[0] + fh[0] - 1, X.shape[0] + fh[-1] - 1
    y_pred_stats = sm_ardl.predict(res.params, start=start, end=end, exog_oos=X_oos)
    y_pred = ardl_sktime.predict(fh=fh, X=X_oos)
    return assert_allclose(y_pred, y_pred_stats)


def test_against_statsmodels_2():
    """Compare sktime's ARDL interface with statsmodels ARDL with different data."""
    # data
    data = grunfeld.load_pandas().data
    oos = data.iloc[-5:, :]
    data = data.iloc[:-5, :]
    y = data.value
    X = data[["capital", "invest"]]
    X_oos = oos[["capital", "invest"]]
    # fit
    lags = 1
    trend = "ct"
    order = 2
    sm_ardl = _ARDL(y, lags, X, order=order, trend=trend)
    res = sm_ardl.fit()
    ardl_sktime = ARDL(lags=lags, order=order, trend=trend)
    ardl_sktime.fit(y=y, X=X, fh=None)
    # predict
    fh = ForecastingHorizon([1, 2, 3])
    start, end = X.shape[0] + fh[0] - 1, X.shape[0] + fh[-1] - 1
    y_pred_stats = sm_ardl.predict(res.params, start=start, end=end, exog_oos=X_oos)
    y_pred = ardl_sktime.predict(fh=fh, X=X_oos)
    return assert_allclose(y_pred, y_pred_stats)


def test_against_statsmodels_3():
    """Compare sktime's ARDL interface with statsmodels ARDL with X=None."""
    # data
    data = longley.load_pandas().data
    data = data.iloc[:-5, :]
    y = data.TOTEMP
    X = None
    X_oos = None
    # fit
    sm_ardl = _ARDL(y, lags=2, exog=None, trend="c")
    res = sm_ardl.fit()
    ardl_sktime = ARDL(lags=2, trend="c")
    ardl_sktime.fit(y=y, X=X, fh=None)
    # predict
    fh = ForecastingHorizon([1, 2, 3])
    start, end = y.shape[0] + fh[0] - 1, y.shape[0] + fh[-1] - 1
    y_pred_stats = sm_ardl.predict(res.params, start=start, end=end, exog_oos=X_oos)
    y_pred = ardl_sktime.predict(fh=fh, X=X_oos)
    return assert_allclose(y_pred, y_pred_stats)


def test_against_statsmodels_4():
    """Compare sktime's ARDL interface with statsmodels ARDL."""
    # data
    data = load_macroeconomic()
    data = data.iloc[:-5, :]
    y = data.realgdp
    X = None
    X_oos = None
    # fit
    sm_ardl = _ARDL(y, lags=2, exog=None, trend="c")
    res = sm_ardl.fit()
    ardl_sktime = ARDL(lags=2, trend="c")
    ardl_sktime.fit(y=y, X=X, fh=None)
    # predict
    fh = ForecastingHorizon([1, 2, 3])
    start, end = y.shape[0] + fh[0] - 1, y.shape[0] + fh[-1] - 1
    y_pred_stats = sm_ardl.predict(res.params, start=start, end=end, exog_oos=X_oos)
    y_pred = ardl_sktime.predict(fh=fh, X=X_oos)
    return assert_allclose(y_pred, y_pred_stats)


def test_auto_ardl():
    """Compare sktime's ARDL interface with statsmodels ardl_select_order."""
    # data
    data = longley.load_pandas().data
    oos = data.iloc[-5:, :]
    data = data.iloc[:-5, :]
    y = data.TOTEMP
    X = data[["GNPDEFL", "GNP"]]
    X_oos = oos[["GNPDEFL", "GNP"]]
    maxlag = 2
    maxorder = 2
    trend = "c"
    # fit
    sm_ardl = _ardl_select_order(
        endog=y, maxlag=maxlag, exog=X, maxorder=maxorder, trend=trend
    )
    res = sm_ardl.model.fit()
    ardl_sktime = ARDL(auto_ardl=True, maxlag=maxlag, maxorder=maxorder, trend=trend)
    ardl_sktime.fit(y=y, X=X, fh=None)
    # predict
    fh = ForecastingHorizon([0])
    start, end = X.shape[0] + fh[0] - 1, X.shape[0] + fh[-1] - 1
    y_pred_stats = res.predict(start=start, end=end, exog_oos=X_oos)
    y_pred = ardl_sktime.predict(fh=fh, X=X_oos)
    return assert_allclose(y_pred, y_pred_stats)


def test_against_statsmodels_5():
    """Compare sktime's ARDL interface with statsmodels ARDL."""
    # data
    data = danish_data.load().data
    data[["lrm", "lry", "ibo", "ide"]]
    y = data.lrm
    X = data[["lry", "ibo", "ide"]]
    # fit
    sel_res = _ardl_select_order(
        data.lrm, 3, data[["lry", "ibo", "ide"]], 3, ic="aic", trend="c"
    )
    res = sel_res.model.fit()
    ardl_sktime = ARDL(auto_ardl=True, maxlag=3, maxorder=3, trend="c", ic="aic")
    ardl_sktime.fit(y=y, X=X, fh=None)
    ardl_loglik = ardl_sktime.get_fitted_params()["loglike"]
    sm_loglik = sel_res.model.loglike(res.params)
    return assert_allclose(ardl_loglik, sm_loglik)

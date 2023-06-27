"""Tests the AutoReg model."""
__author__ = ["CTFallon", "mgazian000", "jonathanbechtel"]

import pytest
from numpy.testing import assert_allclose

# from sktime.datasets import
from sktime.forecasting.auto_reg import AutoREG
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation._dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(AutoREG, severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_against_statsmodels():
    """Compare sktime's AutoREG interface with statsmodels AutoREG,
    without exog data."""
    from statsmodels.tsa.ar_model import AutoReg as _AutoReg

    from sktime.datasets import load_airline

    # data
    data = load_airline()
    # fit
    sm_autoreg = _AutoReg(data, 2, trend="c")
    res = sm_autoreg.fit()
    autoreg_sktime = AutoREG(lags=2, trend="c")
    autoreg_sktime.fit(y=data, fh=None)
    # predict
    fh = ForecastingHorizon([x for x in range(1, 13)])
    start, end = data.shape[0] + fh[0] - 1, data.shape[0] + fh[-1] - 1
    y_pred_stats = sm_autoreg.predict(res.params, start=start, end=end)
    y_pred = autoreg_sktime.predict(fh=fh)
    return assert_allclose(y_pred, y_pred_stats)


@pytest.mark.skipif(
    not _check_estimator_deps(AutoREG, severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_against_statsmodels_fit_results():
    """Compare fit stats of sktime's AutoREG interface with statsmodels AutoREG,
    without exog data"""
    from statsmodels.tsa.ar_model import AutoReg as _AutoReg

    from sktime.datasets import load_airline

    # data
    data = load_airline()
    # fit
    sm_autoreg = _AutoReg(data, 2, trend="c")
    res = sm_autoreg.fit()
    autoreg_sktime = AutoREG(lags=2, trend="c")
    autoreg_sktime.fit(y=data, fh=None)
    temp_stats_dict = autoreg_sktime.get_fitted_params()
    stats_dict = {key: temp_stats_dict[key] for key in ["aic", "aicc", "bic", "hqic"]}
    sm_stats_dict = {
        "aic": res.aic,
        "aicc": res.aicc,
        "bic": res.bic,
        "hqic": res.hqic,
    }
    return assert_allclose(list(stats_dict.values()), list(sm_stats_dict.values()))


@pytest.mark.skipif(
    not _check_estimator_deps(AutoREG, severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_against_statsmodels_exog():
    """Compare sktime's autoReg interface with statsmodels autoReg, with exog data."""
    from statsmodels.tsa.ar_model import AutoReg as _AutoReg

    from sktime.datasets import load_longley

    # data
    y, X_og = load_longley()
    X_oos = X_og.iloc[-5:, :]
    y = y.iloc[:-5]
    X = X_og.iloc[:-5, :]
    X = X[["GNPDEFL", "GNP"]]
    X_oos = X_oos[["GNPDEFL", "GNP"]]
    # fit
    sm_autoreg = _AutoReg(endog=y, lags=2, exog=X, trend="c")
    res = sm_autoreg.fit()
    autoreg_sktime = AutoREG(lags=2, trend="c")
    autoreg_sktime.fit(y=y, X=X, fh=None)
    # predict
    fh = ForecastingHorizon([1, 2, 3])
    start, end = X.shape[0] + fh[0] - 1, X.shape[0] + fh[-1] - 1
    y_pred_stats = sm_autoreg.predict(
        res.params, start=start, end=end, exog=X, exog_oos=X_oos
    )
    y_pred = autoreg_sktime.predict(fh=fh, X=X_oos)
    return assert_allclose(y_pred, y_pred_stats)

import pytest
import numpy as np

from sktime.forecasting import ThetaForecaster
from sktime.datasets import load_airline


__author__ = "big-o@github"


# forecast horizons
FHS = ([1], np.arange(1, 5), np.arange(1, 20))


@pytest.mark.parametrize("fh", FHS)
def test_ThetaForecaster_univariate(fh):
    y = np.log1p(load_airline())
    y_train, y_test = y[:-len(fh)], y[-len(fh):]

    m = ThetaForecaster(seasonal_periods=12)
    m.fit(y_train)
    y_pred = m.predict(fh=fh)

    assert y_pred.shape[0] == len(fh)
    assert m.score(y_test, fh=fh) > 0

    errs = m.pred_errs(alpha=0.05)

    # Prediction errors should always increase with the horizon.
    assert errs.is_monotonic_increasing

    # Performance on this particular dataset should be reasonably good.
    assert np.allclose(y_pred, y_test, rtol=0.05)
    assert np.all(y_pred - errs < y_test)
    assert np.all(y_test < y_pred + errs)

    y_pred2, errs2 = m.predict(fh=fh, return_conf_int=True, alpha=0.05)
    assert np.allclose(y_pred, y_pred2)
    assert np.allclose(errs, errs2)

    y_pred3, errs3 = m.predict(fh=fh, return_conf_int=True, alpha=[0.05, 0.2])
    assert np.allclose(y_pred, y_pred2)
    assert np.allclose(errs, errs3[0])

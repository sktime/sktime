"""Tests for the ARIMA estimator and _PmdArimaAdapter."""

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARIMA_pred_quantiles_insample():
    """Test ARIMA predict_quantiles with in-sample fh.

    Failure condition of #4468.
    """
    y = load_airline()
    forecaster = ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12))
    forecaster.fit(y)
    forecaster.predict_quantiles(fh=y.index, X=None, alpha=[0.05, 0.95])


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARIMA_integer_index_not_starting_at_zero():
    """Test ARIMA out-of-sample prediction with an integer index not starting at 0.

    statsmodels >= 0.15 raises "No supported index is available" for such indices,
    failure condition of #11228.
    """
    import numpy as np
    import pandas as pd

    index = pd.Index(np.arange(3, 43), dtype="int64")
    rng = np.random.default_rng(42)
    y = pd.Series(rng.normal(size=40).cumsum(), index=index)
    X = pd.DataFrame({"x": rng.normal(size=45)}, index=np.arange(3, 48))

    forecaster = ARIMA(order=(1, 0, 0))
    forecaster.fit(y.iloc[:35], X=X.iloc[:35], fh=[1, 2])

    y_pred = forecaster.predict(X=X.loc[38:39])
    assert list(y_pred.index) == [38, 39]

    pred_int = forecaster.predict_interval(X=X.loc[38:39])
    assert list(pred_int.index) == [38, 39]

    forecaster.update(y.iloc[35:], X=X.loc[38:42])
    y_pred = forecaster.predict(X=X.loc[43:44])
    assert list(y_pred.index) == [43, 44]


def _longley_arima_estimators():
    from sktime.forecasting.arima import AutoARIMA

    arima = ARIMA(order=(2, 0, 0), suppress_warnings=True)
    auto = AutoARIMA(
        start_p=1,
        max_p=2,
        start_q=0,
        max_q=1,
        d=0,
        seasonal=False,
        suppress_warnings=True,
    )
    return [arima, auto]


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("est_idx", [0, 1])
@pytest.mark.parametrize("fh", [[1], [3], [1, 2, 3], [2, 4], [-1, 0, 1, 2]])
def test_pmdarima_X_longer_than_fh(est_idx, fh):
    """Test pmdarima adapter slices X to the rows pmdarima needs, for any fh.

    Failure condition of #9344: ``X`` containing more rows than required,
    or ``fh`` not starting at 1, gave a row count mismatch in pmdarima.
    """
    import numpy as np

    from sktime.datasets import load_longley
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=5)

    forecaster = _longley_arima_estimators()[est_idx].fit(y_train, X=X_train)

    # X_test is longer than the out-of-sample part of fh, and starts at cutoff + 1
    y_pred = forecaster.predict(fh=fh, X=X_test)
    assert len(y_pred) == len(fh)

    # expected: pmdarima called with exactly the rows for steps 1..n_periods
    fh_oos = [k for k in fh if k > 0]
    n_periods = max(fh_oos)
    pmd_pred = np.asarray(
        forecaster._forecaster.predict(n_periods=n_periods, X=X_test.iloc[:n_periods])
    )
    expected = pmd_pred[[k - 1 for k in fh_oos]]
    np.testing.assert_allclose(y_pred.values[-len(fh_oos) :], expected)

    y_pred_int = forecaster.predict_interval(fh=fh, X=X_test)
    assert len(y_pred_int) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("est_idx", [0, 1])
def test_pmdarima_update_predict_with_X(est_idx):
    """Test update_predict with X equals a manual update/predict loop.

    Failure condition of #9344.
    """
    import pandas as pd

    from sktime.datasets import load_longley
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=5)

    from sktime.split import SlidingWindowSplitter

    forecaster = _longley_arima_estimators()[est_idx].fit(y_train, X=X_train)
    # window_length=1: each update receives only the new observation
    cv = SlidingWindowSplitter(fh=1, window_length=1, step_length=1)
    y_pred = (
        forecaster.clone()
        .fit(y_train, X=X_train)
        .update_predict(y_test[:-1], cv=cv, X=X_test, update_params=True)
    )

    # cutoffs at 1958, 1959, 1960 (splitter needs fh inside y), one step ahead
    expected = []
    manual = forecaster.clone().fit(y_train, X=X_train)
    for i in range(1, len(y_test) - 1):
        manual.update(y_test.iloc[i - 1 : i], X=X_test.iloc[i - 1 : i])
        expected.append(manual.predict(fh=[1], X=X_test.iloc[i : i + 1]))
    expected = pd.concat(expected)

    pd.testing.assert_series_equal(y_pred, expected, check_freq=False)


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pmdarima_X_with_gap_raises():
    """Test X that does not cover every step from the cutoff raises, cf #9344."""
    from sktime.datasets import load_longley
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=5)

    forecaster = ARIMA(order=(2, 0, 0), suppress_warnings=True).fit(y_train, X=X_train)

    with pytest.raises(ValueError, match="every step"):
        forecaster.predict(fh=[1, 2], X=X_test.iloc[1:])


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pmdarima_in_sample_fh_ignores_future_X():
    """Test in-sample predictions do not consume future X rows, cf #9344."""
    import numpy as np

    from sktime.datasets import load_longley
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=5)

    forecaster = ARIMA(order=(2, 0, 0), suppress_warnings=True).fit(y_train, X=X_train)

    long_X = forecaster.predict(fh=[-2, -1], X=X_test)
    short_X = forecaster.predict(fh=[-2, -1], X=X_test.iloc[:1])

    np.testing.assert_allclose(long_X.values, short_X.values)

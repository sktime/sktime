# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TiRex2Forecaster."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.tirex2 import TiRex2Forecaster

pytestmark = pytest.mark.skipif(
    not _check_estimator_deps(TiRex2Forecaster, severity="none"),
    reason="missing soft dependencies for TiRex-2 tests",
)

# a context long enough to be meaningful, short enough to keep tests quick
_N = 64


def _series(n=_N, columns=("y",), start=0):
    """Return a simple sine-based frame with ``n`` rows and given columns."""
    index = pd.RangeIndex(start, start + n)
    data = {c: np.sin(np.arange(n) / 8.0 + i) for i, c in enumerate(columns)}
    return pd.DataFrame(data, index=index)


class _StubModel:
    """Stand-in for the loaded model, exposing only what ``_fit`` reads.

    Deliberately uses a context limit different from the real checkpoint, so
    that a test can tell whether ``_fit`` reads the limit off the model rather
    than relying on a hard-coded constant.
    """

    context_len = 50


def _fitted_stub(y, X=None):
    """Fit a forecaster against ``_StubModel``, avoiding a checkpoint download."""
    forecaster = TiRex2Forecaster()
    forecaster._load_model = _StubModel
    forecaster.fit(y, X=X, fh=[1, 2, 3])
    return forecaster


def test_context_is_truncated_to_model_context_len():
    """Context is truncated to the model's ``context_len``, keeping the tail."""
    y = _series(200)
    forecaster = _fitted_stub(y)

    assert len(forecaster._context) == _StubModel.context_len
    pd.testing.assert_frame_equal(forecaster._context, y.iloc[-50:])


def test_context_is_not_padded_when_shorter_than_limit():
    """A context shorter than ``context_len`` is stored unchanged."""
    y = _series(20)
    forecaster = _fitted_stub(y)

    pd.testing.assert_frame_equal(forecaster._context, y)


def test_split_covariates_known_future_and_past_only():
    """Columns in fit and predict are known-future, fit-only columns are past."""
    forecaster = _fitted_stub(_series(), X=_series(columns=("a", "b")))
    X_future = _series(3, columns=("a",), start=_N)

    past, future = forecaster._split_covariates(X_future, pred_len=3)

    assert list(past.columns) == ["b"]
    assert len(past) == len(forecaster._context)

    assert list(future.columns) == ["a"]
    # known-future covariates carry past values followed by future values
    assert len(future) == len(forecaster._context) + 3


def test_split_covariates_all_known_future():
    """When every fit column is also passed to predict, there are no past-only."""
    forecaster = _fitted_stub(_series(), X=_series(columns=("a",)))

    past, future = forecaster._split_covariates(
        _series(3, columns=("a",), start=_N), pred_len=3
    )

    assert past is None
    assert len(future) == len(forecaster._context) + 3


def test_split_covariates_without_predict_x():
    """Without X in predict, all fit columns are treated as past-only."""
    forecaster = _fitted_stub(_series(), X=_series(columns=("a", "b")))

    past, future = forecaster._split_covariates(None, pred_len=3)

    assert list(past.columns) == ["a", "b"]
    assert future is None


def test_split_covariates_column_order_follows_fit():
    """Covariate column order follows fit, not the order passed to predict."""
    forecaster = _fitted_stub(_series(), X=_series(columns=("a", "b", "c")))
    X_future = _series(3, columns=("c", "a"), start=_N)[["c", "a"]]

    past, future = forecaster._split_covariates(X_future, pred_len=3)

    assert list(future.columns) == ["a", "c"]
    assert list(past.columns) == ["b"]


def test_x_in_predict_but_not_in_fit_raises():
    """Passing X only to predict raises an informative error."""
    forecaster = _fitted_stub(_series())

    with pytest.raises(ValueError, match="X was passed to predict but not to fit"):
        forecaster._split_covariates(_series(3, columns=("a",), start=_N), pred_len=3)


def test_unknown_column_in_predict_x_raises():
    """Columns unseen in fit raise an informative error."""
    forecaster = _fitted_stub(_series(), X=_series(columns=("a",)))

    with pytest.raises(ValueError, match="columns not seen in fit"):
        forecaster._split_covariates(_series(3, columns=("zz",), start=_N), pred_len=3)


def test_short_predict_x_raises():
    """X shorter than the forecasting horizon raises an informative error."""
    forecaster = _fitted_stub(_series(), X=_series(columns=("a",)))

    with pytest.raises(ValueError, match="must cover the full"):
        forecaster._split_covariates(_series(2, columns=("a",), start=_N), pred_len=10)


def test_predict_end_to_end():
    """A plain fit/predict succeeds and returns finite forecasts.

    This is the regression test for the ``torch.compile`` failure: ``tirex-2``
    compiles its forward path unconditionally, which needs a C++ toolchain, so
    the forecaster runs the model under ``set_stance("force_eager")``. Without
    that, this test raises ``InductorError`` on machines without a compiler.
    """
    y = _series(128)
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(y)

    y_pred = forecaster.predict(fh=[1, 2, 3])

    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape == (3, 1)
    assert list(y_pred.index) == [128, 129, 130]
    assert np.isfinite(y_pred.to_numpy()).all()


def test_predict_non_contiguous_fh():
    """A non-contiguous horizon returns exactly the requested steps."""
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(_series(128))

    y_pred = forecaster.predict(fh=[1, 5, 9])

    assert list(y_pred.index) == [128, 132, 136]


def test_multivariate_shape_and_column_order():
    """Multivariate forecasts keep the column names and order seen in fit."""
    columns = ["z", "a", "m"]
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(_series(columns=columns))

    y_pred = forecaster.predict(fh=[1, 2])

    assert y_pred.shape == (2, 3)
    assert list(y_pred.columns) == columns


def test_horizon_beyond_model_limit_warns_and_pads_with_nan():
    """Beyond the model's maximum horizon, forecasts are nan and a warning is raised."""
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(_series(128))

    max_horizon = forecaster.model.future_len
    fh = list(range(1, max_horizon + 11))

    with pytest.warns(UserWarning, match="supports at most"):
        y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == len(fh)
    assert np.isfinite(y_pred.to_numpy()[:max_horizon]).all()
    assert np.isnan(y_pred.to_numpy()[max_horizon:]).all()


@pytest.mark.parametrize("lower,upper", [(0.01, 0.99), (0.05, 0.95)])
def test_quantiles_outside_native_grid_are_clamped(lower, upper):
    """Levels outside the native 0.1 to 0.9 grid clamp to the nearest level."""
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(_series(128))

    pred_q = forecaster.predict_quantiles(fh=[1, 2, 3], alpha=[lower, 0.1, 0.9, upper])

    np.testing.assert_allclose(pred_q[("y", lower)], pred_q[("y", 0.1)])
    np.testing.assert_allclose(pred_q[("y", upper)], pred_q[("y", 0.9)])


def test_predict_quantiles_median_matches_predict():
    """The 0.5 quantile is the point forecast."""
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(_series(128))

    y_pred = forecaster.predict(fh=[1, 2, 3])
    pred_q = forecaster.predict_quantiles(fh=[1, 2, 3], alpha=[0.5])

    np.testing.assert_allclose(pred_q[("y", 0.5)], y_pred["y"])


def test_predict_proba_returns_histogram_qpd():
    """predict_proba returns a HistogramQPD consistent with predict_quantiles."""
    from skpro.distributions import HistogramQPD

    alpha = [0.1, 0.5, 0.9]
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(_series(128))

    pred_dist = forecaster.predict_proba(fh=[1, 3])
    assert isinstance(pred_dist, HistogramQPD)

    from_quantiles = forecaster.predict_quantiles(fh=[1, 3], alpha=alpha)
    from_proba = pred_dist.quantile(alpha=alpha)

    pd.testing.assert_index_equal(from_quantiles.index, from_proba.index)
    np.testing.assert_allclose(from_quantiles, from_proba, atol=1e-6)


def test_predict_with_exogenous_data():
    """Forecasting with past-only and known-future covariates succeeds."""
    forecaster = TiRex2Forecaster(**TiRex2Forecaster.get_test_params()[0])
    forecaster.fit(_series(), X=_series(columns=("a", "b")))

    y_pred = forecaster.predict(fh=[1, 2, 3], X=_series(3, columns=("a",), start=_N))

    assert y_pred.shape == (3, 1)
    assert np.isfinite(y_pred.to_numpy()).all()

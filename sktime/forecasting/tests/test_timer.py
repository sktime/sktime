"""Tests for the TimerForecaster."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.timer import TimerForecaster


@pytest.fixture(autouse=True)
def mock_timer_deps(monkeypatch):
    """Bypass soft dependency checks so unit testing can run without transformers."""
    monkeypatch.setattr(
        TimerForecaster,
        "_tags",
        {**TimerForecaster._tags, "python_dependencies": []},
    )


@pytest.fixture
def mock_timer_model():
    """Mock Timer model with config for unit testing without checkpoint download."""
    config = SimpleNamespace(input_token_len=4)
    model = SimpleNamespace(config=config, device="cpu")
    return model


@pytest.fixture
def timer_forecaster(mock_timer_model, monkeypatch):
    """TimerForecaster fixture with mocked model."""
    forecaster = TimerForecaster(context_length=5)
    monkeypatch.setattr(forecaster, "_load_model", lambda: mock_timer_model)
    return forecaster


def test_timer_update_restricts_cur_y_to_context_length(timer_forecaster):
    """Test that _update restricts self._cur_y to the latest context_length."""
    forecaster = timer_forecaster

    y_train = pd.Series(
        np.arange(10, dtype=np.float32), index=pd.RangeIndex(10), name="a"
    )
    forecaster.fit(y_train)

    y_new = pd.Series(
        np.arange(10, 15, dtype=np.float32), index=pd.RangeIndex(10, 15), name="a"
    )
    forecaster.update(y_new, update_params=False)

    assert len(forecaster._cur_y) == 5
    np.testing.assert_array_equal(
        forecaster._cur_y.to_numpy(),
        np.array([10, 11, 12, 13, 14], dtype=np.float32),
    )
    assert len(forecaster._y_train) == 5
    np.testing.assert_array_equal(
        forecaster._y_train,
        np.array([10, 11, 12, 13, 14], dtype=np.float32),
    )


def test_timer_update_no_context_length_keeps_all_data(mock_timer_model, monkeypatch):
    """Test that _update keeps all data when context_length is None."""
    forecaster = TimerForecaster(context_length=None)
    forecaster.set_tags(python_dependencies=[])
    monkeypatch.setattr(forecaster, "_load_model", lambda: mock_timer_model)

    y_train = pd.Series(
        np.arange(10, dtype=np.float32), index=pd.RangeIndex(10), name="a"
    )
    forecaster.fit(y_train)

    y_new = pd.Series(
        np.arange(10, 15, dtype=np.float32), index=pd.RangeIndex(10, 15), name="a"
    )
    forecaster.update(y_new, update_params=False)

    assert len(forecaster._cur_y) == 15
    np.testing.assert_array_equal(
        forecaster._cur_y.to_numpy(),
        np.arange(15, dtype=np.float32),
    )
    assert len(forecaster._y_train) == 15


def test_timer_update_pads_if_shorter_than_input_token_len(
    mock_timer_model, monkeypatch
):
    """Test that _update pads _y_train if shorter than input_token_len."""
    # input_token_len is 4, context_length is 2
    forecaster = TimerForecaster(context_length=2)
    forecaster.set_tags(python_dependencies=[])
    monkeypatch.setattr(forecaster, "_load_model", lambda: mock_timer_model)

    y_train = pd.Series([1.0, 2.0], index=pd.RangeIndex(2), name="a")
    forecaster.fit(y_train)

    y_new = pd.Series([3.0, 4.0], index=pd.RangeIndex(2, 4), name="a")
    forecaster.update(y_new, update_params=False)

    assert len(forecaster._cur_y) == 2
    np.testing.assert_array_equal(forecaster._cur_y.to_numpy(), [3.0, 4.0])
    # _y_train should be padded to input_token_len (4)
    assert len(forecaster._y_train) == 4
    np.testing.assert_array_equal(forecaster._y_train, [0.0, 0.0, 3.0, 4.0])


def test_timer_predict_after_update_uses_latest_context(monkeypatch):
    """Test that predict after update feeds the updated context window to model."""
    torch = pytest.importorskip("torch")

    captured_inputs = []

    def mock_generate(input_tensor, max_new_tokens):
        captured_inputs.append(input_tensor.clone())
        return torch.zeros((1, max_new_tokens), dtype=torch.float32)

    config = SimpleNamespace(input_token_len=2)
    mock_model = SimpleNamespace(
        config=config,
        device="cpu",
        generate=mock_generate,
    )

    forecaster = TimerForecaster(context_length=4)
    monkeypatch.setattr(forecaster, "_load_model", lambda: mock_model)

    y_train = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.RangeIndex(4), name="val")
    forecaster.fit(y_train)

    y_new = pd.Series([5.0, 6.0], index=pd.RangeIndex(4, 6), name="val")
    forecaster.update(y_new, update_params=False)

    y_pred = forecaster.predict(fh=[1, 2])
    assert len(y_pred) == 2
    assert len(captured_inputs) == 1
    # Should use the updated context: [3.0, 4.0, 5.0, 6.0]
    np.testing.assert_array_equal(
        captured_inputs[0].numpy().squeeze(0),
        np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    )

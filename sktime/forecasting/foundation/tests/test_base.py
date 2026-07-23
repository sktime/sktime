"""Tests for ``BaseFoundationForecaster``."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pickle

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
    clear_foundation_model_cache,
)


class _DummyModel:
    """Minimal model object stored in a model handle."""


class _DummyFoundationForecaster(BaseFoundationForecaster):
    """Dependency-free concrete forecaster used to exercise the base class."""

    _tags = {
        "capability:multivariate": True,
        "capability:pred_int": True,
        "python_dependencies": [],
        "requires-fh-in-fit": False,
    }
    _uses_torch_inference_context = False

    def __init__(self, model_spec=None, result=None):
        if model_spec is None:
            model_spec = FoundationModelSpec(model_path="dummy", device="cpu")

        self.result = result
        self.load_count = 0

        super().__init__(model_spec=model_spec)

    def _update_attrs_in_fit(self, y, X, fh):
        self.fit_hook_args_ = (y, X, fh)

    def _load_model(self):
        self.load_count += 1
        return ModelHandle(model=_DummyModel())

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        self.inference_args_ = {
            "handle": handle,
            "context_y": context_y,
            "context_X": context_X,
            "future_X": future_X,
            "pred_len": pred_len,
            "fh": fh,
            "alpha": alpha,
        }
        if self.result is not None:
            return self.result

        values = np.arange(1, pred_len + 1, dtype=float)
        values = np.repeat(values[:, np.newaxis], context_y.shape[1], axis=1)
        quantiles = None
        if alpha is not None:
            quantiles = {value: values + value for value in alpha}
        return ForecastResult(mean=values, quantiles=quantiles)


class _PointOnlyFoundationForecaster(_DummyFoundationForecaster):
    """Dummy forecaster without probabilistic prediction capability."""

    _tags = {"capability:pred_int": False}


class _DependencyFoundationForecaster(_DummyFoundationForecaster):
    """Dummy forecaster with a dependency tag for dynamic-tag tests."""

    _tags = {"python_dependencies": ["dummy-package"]}


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Keep process-local model handles isolated between tests."""
    clear_foundation_model_cache()
    yield
    clear_foundation_model_cache()


@pytest.fixture
def y():
    """Simple multivariate target used by lifecycle tests."""
    return pd.DataFrame(
        {"temperature": [10.0, 11.0, 12.0], "pressure": [20.0, 21.0, 22.0]},
        index=pd.RangeIndex(3),
    )


def test_constructor_requires_model_spec():
    """The base constructor rejects objects other than FoundationModelSpec."""
    with pytest.raises(TypeError, match="model_spec must be a FoundationModelSpec"):
        BaseFoundationForecaster(model_spec={"model_path": "dummy"})


def test_post_init_normalizes_spec_without_mutating_constructor_spec():
    """Runtime normalization creates an isolated fitted specification."""
    config = {"architecture": {"layers": 2}}
    spec = FoundationModelSpec(
        model_path="dummy",
        config=config,
        device="cpu",
        random_state=42,
    )

    forecaster = _DummyFoundationForecaster(model_spec=spec)
    config["architecture"]["layers"] = 3

    assert forecaster.model_spec is not spec
    assert forecaster.model_spec.config == {"architecture": {"layers": 2}}
    assert forecaster.model_spec.config is not config
    assert forecaster.model_spec.random_state is not None
    assert spec.random_state == 42


def test_post_init_normalizes_none_config_to_empty_dict():
    """A missing model configuration becomes an empty runtime mapping."""
    forecaster = _DummyFoundationForecaster(
        FoundationModelSpec(model_path="dummy", config=None)
    )

    assert forecaster.model_spec.config == {}


@pytest.mark.parametrize(
    "dtype_name",
    [
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int64",
        "bool",
        "complex64",
        "half",
        "double",
        "long",
    ],
)
def test_post_init_resolves_torch_dtype_strings(dtype_name):
    """All valid Torch dtype names are resolved from their string form."""
    torch = pytest.importorskip("torch")
    forecaster = _DummyFoundationForecaster(
        FoundationModelSpec(dtype=f"torch.{dtype_name}")
    )

    assert forecaster.model_spec.dtype is getattr(torch, dtype_name)


@pytest.mark.parametrize("dtype", ["auto", None])
def test_post_init_preserves_non_torch_dtype_values(dtype):
    """Backend-specific and missing dtype values pass through unchanged."""
    forecaster = _DummyFoundationForecaster(FoundationModelSpec(dtype=dtype))

    assert forecaster.model_spec.dtype is dtype


@pytest.mark.parametrize("dtype", ["torch.not_a_dtype", "torch.Tensor"])
def test_post_init_rejects_invalid_torch_dtype_strings(dtype):
    """Invalid or non-dtype Torch attributes produce a clear error."""
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="Unknown Torch dtype"):
        _DummyFoundationForecaster(FoundationModelSpec(dtype=dtype))


@pytest.mark.parametrize(
    "ignore_deps, expected", [(False, ["dummy-package"]), (True, [])]
)
def test_ignore_deps_updates_dependency_tag(ignore_deps, expected):
    """Dependency checks are cleared only when explicitly disabled."""
    spec = FoundationModelSpec(ignore_deps=ignore_deps)

    forecaster = _DependencyFoundationForecaster(spec)

    assert forecaster.get_tag("python_dependencies") == expected


def test_fit_stores_context_and_calls_fit_hook(y):
    """Fit snapshots y and X and invokes the extension hook before loading."""
    X = pd.DataFrame({"holiday": [0, 1, 0]}, index=y.index)
    expected_y = y.copy()
    expected_X = X.copy()
    forecaster = _DummyFoundationForecaster()

    returned = forecaster.fit(y, X=X, fh=[1, 2])
    pd.testing.assert_frame_equal(forecaster.fit_hook_args_[0], expected_y)
    pd.testing.assert_frame_equal(forecaster.fit_hook_args_[1], expected_X)
    assert list(forecaster.fit_hook_args_[2]) == [1, 2]

    y.iloc[0, 0] = -1
    X.iloc[0, 0] = -1

    assert returned is forecaster
    pd.testing.assert_frame_equal(forecaster.context_y_, expected_y)
    pd.testing.assert_frame_equal(forecaster.context_X_, expected_X)
    assert isinstance(forecaster.model_handle_, ModelHandle)
    assert forecaster.load_count == 1


def test_fit_without_X_stores_none(y):
    """Fit preserves the absence of exogenous context."""
    forecaster = _DummyFoundationForecaster().fit(y)

    assert forecaster.context_X_ is None
    assert forecaster.fit_hook_args_[1] is None


def test_predict_formats_sparse_horizon_and_forwards_inputs(y):
    """Predict selects sparse horizon rows and passes context to inference."""
    past_X = pd.DataFrame({"holiday": [0, 1, 0]}, index=y.index)
    future_X = pd.DataFrame({"holiday": [1, 0, 1]}, index=pd.RangeIndex(3, 6))
    forecaster = _DummyFoundationForecaster().fit(y, X=past_X)

    actual = forecaster.predict(fh=[1, 3], X=future_X)
    expected = pd.DataFrame(
        {"temperature": [1.0, 3.0], "pressure": [1.0, 3.0]},
        index=pd.Index([3, 5]),
    )

    pd.testing.assert_frame_equal(actual, expected)
    args = forecaster.inference_args_
    assert args["handle"] is forecaster.model_handle_
    pd.testing.assert_frame_equal(args["context_y"], forecaster.context_y_)
    pd.testing.assert_frame_equal(args["context_X"], forecaster.context_X_)
    pd.testing.assert_frame_equal(args["future_X"], future_X)
    assert args["pred_len"] == 3
    assert args["alpha"] is None


def test_predict_quantiles_formats_variables_and_alpha(y):
    """Quantile prediction returns the sktime variable/alpha column layout."""
    forecaster = _DummyFoundationForecaster().fit(y)

    actual = forecaster.predict_quantiles(fh=[1, 3], alpha=[0.1, 0.9])
    expected = pd.DataFrame(
        [
            [1.1, 1.9, 1.1, 1.9],
            [3.1, 3.9, 3.1, 3.9],
        ],
        index=pd.Index([3, 5]),
        columns=pd.MultiIndex.from_product([["temperature", "pressure"], [0.1, 0.9]]),
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert forecaster.inference_args_["alpha"] == (0.1, 0.9)
    assert forecaster.inference_args_["pred_len"] == 3


def test_inference_must_return_forecast_result(y):
    """A concrete adapter gets a clear error for returning an invalid result."""
    forecaster = _DummyFoundationForecaster(result=np.ones(2)).fit(y)

    with pytest.raises(TypeError, match="_inference must return ForecastResult"):
        forecaster.predict(fh=[1, 2])


def test_model_cache_reuses_handle_for_equal_loading_spec(y):
    """Equal loading specifications share one process-local model handle."""
    spec_1 = FoundationModelSpec(
        model_path="dummy",
        device="cpu",
        load_extra_kwargs={"options": {"layers": [1, 2]}},
        predict_extra_kwargs={"temperature": 0.1},
    )
    spec_2 = FoundationModelSpec(
        model_path="dummy",
        device="cpu",
        load_extra_kwargs={"options": {"layers": [1, 2]}},
        predict_extra_kwargs={"temperature": 0.9},
    )
    first = _DummyFoundationForecaster(spec_1).fit(y)
    second = _DummyFoundationForecaster(spec_2).fit(y)

    assert second.model_handle_ is first.model_handle_
    assert first.load_count == 1
    assert second.load_count == 0


def test_model_cache_separates_different_loading_spec(y):
    """A loading-relevant spec change produces a separate model handle."""
    first = _DummyFoundationForecaster(
        FoundationModelSpec(model_path="model-a", device="cpu")
    ).fit(y)
    second = _DummyFoundationForecaster(
        FoundationModelSpec(model_path="model-b", device="cpu")
    ).fit(y)

    assert second.model_handle_ is not first.model_handle_
    assert first.load_count == second.load_count == 1


def test_model_cache_includes_load_affecting_config(y):
    """Different model configurations produce separate cached handles."""
    first = _DummyFoundationForecaster(
        FoundationModelSpec(
            model_path="dummy", device="cpu", config={"hidden_size": 32}
        )
    ).fit(y)
    second = _DummyFoundationForecaster(
        FoundationModelSpec(
            model_path="dummy", device="cpu", config={"hidden_size": 64}
        )
    ).fit(y)

    assert second.model_handle_ is not first.model_handle_
    assert first.load_count == second.load_count == 1


def test_model_cache_normalizes_config_objects(y):
    """Equivalent config objects with ``to_dict`` share a cached handle."""

    class _Config:
        def __init__(self, hidden_size):
            self.hidden_size = hidden_size

        def to_dict(self):
            return {"hidden_size": self.hidden_size}

    first = _DummyFoundationForecaster(
        FoundationModelSpec(model_path="dummy", config=_Config(32))
    ).fit(y)
    second = _DummyFoundationForecaster(
        FoundationModelSpec(model_path="dummy", config=_Config(32))
    ).fit(y)

    assert second.model_handle_ is first.model_handle_
    assert first.load_count == 1
    assert second.load_count == 0


def test_pickle_drops_model_handle_and_reloads_lazily(y):
    """Serialization excludes backend state and prediction reloads it on demand."""
    forecaster = _DummyFoundationForecaster().fit(y)
    restored = pickle.loads(pickle.dumps(forecaster))

    assert restored.model_handle_ is None
    clear_foundation_model_cache()

    actual = restored.predict(fh=[2])

    expected = pd.DataFrame(
        {"temperature": [2.0], "pressure": [2.0]}, index=pd.Index([4])
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert restored.model_handle_ is not None
    assert restored.load_count == 2


def test_quantile_implementation_respects_capability_tag():
    """The shared quantile method is exposed only for probabilistic adapters."""
    assert _DummyFoundationForecaster._has_implementation_of("_predict_quantiles")
    assert not _PointOnlyFoundationForecaster._has_implementation_of(
        "_predict_quantiles"
    )


def test_update_model_spec_replaces_active_spec():
    """Derived fit settings replace the active spec without mutating the old one."""
    original = FoundationModelSpec(model_path="dummy", device="cpu")
    forecaster = _DummyFoundationForecaster(original)

    forecaster._update_model_spec(load_extra_kwargs={"context_length": 32})

    assert forecaster.model_spec is not original
    assert original.load_extra_kwargs == {}
    assert forecaster.model_spec.load_extra_kwargs == {"context_length": 32}


def test_torch_inference_context_is_seeded_and_restores_rng(y):
    """Torch inference is deterministic, uses eval mode, and restores RNG state."""
    torch = pytest.importorskip("torch")

    class _TorchModel:
        device = "cpu"

        def __init__(self):
            self.eval_count = 0

        def eval(self):
            self.eval_count += 1
            return self

    class _TorchFoundationForecaster(_DummyFoundationForecaster):
        _uses_torch_inference_context = True

        def _load_model(self):
            self.load_count += 1
            return ModelHandle(model=_TorchModel())

        def _inference(self, **kwargs):
            self.inference_mode_enabled_ = torch.is_inference_mode_enabled()
            values = torch.rand(kwargs["pred_len"], 2).numpy()
            return ForecastResult(mean=values)

    spec = FoundationModelSpec(model_path="torch-dummy", random_state=42)
    forecaster = _TorchFoundationForecaster(spec).fit(y)
    state_before = torch.random.get_rng_state()

    first = forecaster.predict(fh=[1, 2])
    state_after = torch.random.get_rng_state()
    second = forecaster.predict(fh=[1, 2])

    pd.testing.assert_frame_equal(first, second)
    assert torch.equal(state_before, state_after)
    assert forecaster.model_handle_.model.eval_count == 2
    assert forecaster.inference_mode_enabled_

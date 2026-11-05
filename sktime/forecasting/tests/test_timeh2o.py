"""Offline contract tests for the TimeH2O adapter; no gated weights required."""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.timeh2o import TimeH2OForecaster


class _OfflineTimeH2OForecaster(TimeH2OForecaster):
    """Use the substituted inference boundary for adapter-specific tests."""

    _tags = {"python_dependencies": [], "python_version": None}


class _Model:
    """Deterministic substitute for the gated, optional inference boundary."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def forecast(self, context, horizon, future_covariates=None, **kwargs):
        # Distinguish targets, horizon positions, and requested quantiles.
        point = context[:, -1, None] + np.arange(1, horizon + 1)
        if future_covariates is not None:
            assert len(future_covariates[0]) == context.shape[1] + horizon
            point = point + np.asarray(future_covariates[0][-horizon:]).astype(float)
        alpha = kwargs.get("quantile_levels", [0.1, 0.5, 0.9])
        return types.SimpleNamespace(
            point=point,
            quantiles=point[..., None] + np.asarray(alpha),
        )


@pytest.fixture
def forecaster_class(monkeypatch):
    """Replace only the optional upstream model and its network loading."""
    monkeypatch.setitem(sys.modules, "h2ofm", types.SimpleNamespace(TimeH2O=_Model))

    def make_forecaster(**kwargs):
        options = {"model_path": "offline-test-checkpoint", "license_accepted": True}
        options.update(kwargs)
        return _OfflineTimeH2OForecaster(**options)

    return make_forecaster


def test_covariates_alignment_and_mixed_dtypes(forecaster_class):
    """Preserve numeric rows when other covariates are categorical."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="y")
    X = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0], "holiday": ["a", "b", "a", "b"]})
    f = forecaster_class().fit(y, X=X)
    future = pd.DataFrame(
        {"holiday": ["c", "b", "c"], "price": [10.0, 20.0, 30.0]}, index=[4, 5, 6]
    )
    np.testing.assert_allclose(f.predict([1, 3], X=future), [15, 37])


@pytest.mark.parametrize(
    "future,match",
    [
        (None, "X"),
        (pd.DataFrame({"x": [1.0, 2.0]}, index=[4, 6]), "every|all|missing"),
        (pd.DataFrame({"other": [1.0, 2.0, 3.0]}, index=[4, 5, 6]), "columns"),
    ],
)
def test_invalid_future_covariates(forecaster_class, future, match):
    """Reject missing intermediate rows and incompatible covariate columns."""
    f = forecaster_class().fit(
        pd.Series([1.0, 2.0, 3.0, 4.0]), X=pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    )
    with pytest.raises(ValueError, match=match):
        f.predict([1, 3], X=future)


@pytest.mark.parametrize(
    "kwargs,stage,error",
    [
        ({"model_kwargs": {"device": "cpu"}}, "fit", TypeError),
        ({"forecast_kwargs": {"horizon": 2}}, "predict", TypeError),
        ({"forecast_kwargs": {"quantile_levels": [0.5]}}, "fit", ValueError),
        ({"forecast_kwargs": {"past_covariates": [1, 2]}}, "fit", ValueError),
    ],
)
def test_reserved_options_rejected(forecaster_class, kwargs, stage, error):
    """Options cannot silently override adapter-controlled arguments."""
    if stage == "fit":
        with pytest.raises(error):
            forecaster_class(**kwargs).fit(pd.Series([1.0, 2.0, 3.0]))
    else:
        f = forecaster_class(**kwargs).fit(pd.Series([1.0, 2.0, 3.0]))
        with pytest.raises(error):
            f.predict([1])


def test_nullable_numeric_inputs(forecaster_class, monkeypatch):
    """Nullable numeric arrays must not become upstream categorical labels."""

    class NumericModel(_Model):
        def forecast(self, context, *args, future_covariates=None, **kwargs):
            assert context.dtype.kind == "f"
            assert np.isnan(context[0, 1])
            assert pd.api.types.is_numeric_dtype(future_covariates[0])
            assert pd.isna(future_covariates[0].iloc[1])
            return super().forecast(
                context, *args, future_covariates=future_covariates, **kwargs
            )

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", NumericModel)
    y = pd.DataFrame(
        {
            "a": pd.Series([1, None, 3, 4], dtype="Int64"),
            "b": pd.Series([10, None, 30, 40], dtype="Float64"),
        }
    )
    X = pd.DataFrame({"x": pd.Series([1, None, 3, 4], dtype="Int64")})
    f = forecaster_class().fit(y, X=X)
    future = pd.DataFrame({"x": pd.Series([10], index=[4], dtype="Int64")})
    np.testing.assert_array_equal(f.predict([1], X=future), [[15, 51]])


def test_numeric_categorical_covariates(forecaster_class, monkeypatch):
    """Numeric category labels must retain their categorical meaning upstream."""

    class CategoricalModel(_Model):
        def forecast(self, context, horizon, future_covariates=None, **kwargs):
            assert isinstance(future_covariates[0].dtype, pd.CategoricalDtype)
            return super().forecast(context, horizon, **kwargs)

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", CategoricalModel)
    dtype = pd.CategoricalDtype([1, 2, 3])
    X = pd.DataFrame({"label": pd.Series([1, 2, 1, 2], dtype=dtype)})
    future = pd.DataFrame({"label": pd.Series([3], dtype=dtype, index=[4])})
    f = forecaster_class().fit(pd.Series([1.0, 2.0, 3.0, 4.0]), X=X)
    np.testing.assert_array_equal(f.predict([1], X=future), [5])


def test_different_numeric_categories_remain_categorical(forecaster_class, monkeypatch):
    """Category changes must not turn integer labels into numeric covariates."""

    class CategoricalModel(_Model):
        def forecast(self, context, horizon, future_covariates=None, **kwargs):
            covariate = future_covariates[0]
            assert not pd.api.types.is_numeric_dtype(covariate)
            assert list(covariate) == [1, 2, 1, 2, 3]
            return super().forecast(context, horizon, **kwargs)

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", CategoricalModel)
    X = pd.DataFrame({"label": pd.Categorical([1, 2, 1, 2])})
    future = pd.DataFrame({"label": pd.Categorical([3])}, index=[4])
    f = forecaster_class().fit(pd.Series([1.0, 2.0, 3.0, 4.0]), X=X)
    np.testing.assert_array_equal(f.predict([1], X=future), [5])


def test_options_forwarding_and_ownership(forecaster_class, monkeypatch):
    """Use load/inference options without adding keys to the caller's dicts."""

    class ConfiguredModel(_Model):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "local-checkpoint"
            assert kwargs == {
                "device": "cpu",
                "batch_size": 8,
                "local_files_only": True,
                "cache_dir": "local-cache",
            }
            return cls()

        def forecast(self, *args, **kwargs):
            assert kwargs["context_length"] == 16
            return super().forecast(*args, **kwargs)

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", ConfiguredModel)
    model_kwargs = {"local_files_only": True, "cache_dir": "local-cache"}
    forecast_kwargs = {"context_length": 16}
    f = forecaster_class(
        model_path="local-checkpoint",
        license_accepted=True,
        device="cpu",
        batch_size=8,
        model_kwargs=model_kwargs,
        forecast_kwargs=forecast_kwargs,
    ).fit(pd.Series([1.0, 4.0]))
    np.testing.assert_array_equal(f.predict([1]), [5])
    f.predict_quantiles([1], alpha=[0.1, 0.9])
    assert model_kwargs == {"local_files_only": True, "cache_dir": "local-cache"}
    assert forecast_kwargs == {"context_length": 16}


def test_repeated_fit_reuses_loaded_model(forecaster_class, monkeypatch):
    """Repeated fits with one checkpoint reuse the loaded inference model."""
    calls = []

    class CountingModel(_Model):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(path)
            return cls()

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", CountingModel)
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    first = forecaster_class(model_path="cached-checkpoint").fit(y)
    second = forecaster_class(model_path="cached-checkpoint").fit(y)
    assert first.model_ is second.model_
    assert calls == ["cached-checkpoint"]


def test_license_acceptance_required_for_checkpoint():
    """Restricted pretrained weights require explicit license acceptance."""
    with pytest.raises(ValueError, match="license_accepted=True"):
        TimeH2OForecaster(model_path="h2oai/timeh2o-v1").fit(pd.Series([1.0, 2.0]))


def test_gated_checkpoint_error_has_actionable_message(forecaster_class, monkeypatch):
    """A missing Hugging Face grant explains the required user action."""
    from huggingface_hub.errors import GatedRepoError

    class GatedModel(_Model):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            import httpx

            response = httpx.Response(
                403,
                request=httpx.Request("GET", "https://huggingface.co/h2oai/timeh2o-v1"),
            )
            raise GatedRepoError("access denied", response=response)

    monkeypatch.setattr(sys.modules["h2ofm"], "TimeH2O", GatedModel)
    with pytest.raises(ValueError, match="huggingface.co/h2oai/timeh2o-v1"):
        forecaster_class(model_path="h2oai/timeh2o-v1", license_accepted=True).fit(
            pd.Series([1.0, 2.0, 3.0])
        )


def test_predict_proba_uses_upstream_quantile_grid(forecaster_class):
    """Distribution quantiles agree with the upstream forecast quantiles."""
    pytest.importorskip("skpro")
    f = forecaster_class().fit(
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    )
    dist = f.predict_proba([1, 2])
    expected = f.predict_quantiles([1, 2], alpha=[0.1, 0.5, 0.9])
    for alpha in [0.1, 0.5, 0.9]:
        np.testing.assert_allclose(
            dist.ppf(alpha), expected.loc[:, [("a", alpha), ("b", alpha)]], atol=1e-6
        )


def test_model_path_none_runs_without_checkpoint(monkeypatch):
    """The untrained test model needs no gated access or download."""
    h2ofm = pytest.importorskip("h2ofm")

    def reject_download(*args, **kwargs):
        raise AssertionError("untrained model attempted checkpoint download")

    monkeypatch.setattr(h2ofm.TimeH2O, "from_pretrained", reject_download)
    y = pd.Series(np.sin(np.arange(32) / 10), name="target")
    f = TimeH2OForecaster(model_path=None, device="cpu").fit(y)
    pred = f.predict([1, 3])
    assert np.isfinite(pred).all()
    assert pred.shape == (2,)


def test_untrained_model_golden_output():
    """Pin adapter output and requested quantile order on the offline model."""
    pytest.importorskip("h2ofm")
    # Reproduce the golden values with h2ofm==0.43.1 and CPU:
    # y = pd.Series(np.arange(1, 33, dtype=float), name="y")
    # f = TimeH2OForecaster(model_path=None, device="cpu").fit(y)
    # print(f.predict([1, 2, 3]).to_numpy())
    # print(f.predict_quantiles([1, 2, 3], alpha=[0.1, 0.5, 0.9]).to_numpy())
    # The fixed RNG seed and tiny private-module config live in
    # _make_untrained_model; no gated checkpoint is downloaded.
    y = pd.Series(np.arange(1, 33, dtype=float), name="y")
    f = TimeH2OForecaster(model_path=None, device="cpu").fit(y)
    pred = f.predict([1, 2, 3])
    quantiles = f.predict_quantiles([1, 2, 3], alpha=[0.1, 0.5, 0.9])
    np.testing.assert_allclose(pred, [15.560987, 16.808578, 17.096317], atol=1e-3)
    np.testing.assert_allclose(
        quantiles,
        [
            [3.881908, 15.560987, 27.938889],
            [1.134266, 16.808578, 31.144780],
            [6.755712, 17.096317, 28.135172],
        ],
        atol=1e-3,
    )


def test_untrained_model_forecast_options():
    """Additional upstream forecast options affect output as requested."""
    pytest.importorskip("h2ofm")
    y = pd.Series(np.arange(1, 33, dtype=float))
    f = TimeH2OForecaster(
        model_path=None,
        device="cpu",
        forecast_kwargs={
            "flip_equivariant": False,
            "clip_nonnegative": True,
            "snap_integer": True,
        },
    ).fit(y)
    pred = f.predict([1, 2, 3])
    assert (pred >= 0).all()
    np.testing.assert_array_equal(pred, np.round(pred))


def test_untrained_model_interpolates_requested_quantiles():
    """h2ofm interpolates intermediate levels and extrapolates tail levels."""
    pytest.importorskip("h2ofm")
    y = pd.Series(np.arange(1, 33, dtype=float), name="y")
    f = TimeH2OForecaster(model_path=None, device="cpu").fit(y)
    q = f.predict_quantiles([1], alpha=[0.05, 0.1, 0.15, 0.2, 0.9, 0.95])
    np.testing.assert_allclose(
        q[("y", 0.15)], (q[("y", 0.1)] + q[("y", 0.2)]) / 2, atol=1e-4
    )
    assert q[("y", 0.05)].iloc[0] < q[("y", 0.1)].iloc[0]
    assert q[("y", 0.95)].iloc[0] > q[("y", 0.9)].iloc[0]


@pytest.mark.skipif(
    os.environ.get("SKTIME_TIMEH2O_INTEGRATION") != "1",
    reason="Opt in after accepting HF model terms and installing h2ofm",
)
def test_timeh2o_real_checkpoint():
    """Exercise the actual gated checkpoint only when explicitly requested."""
    pytest.importorskip("h2ofm")

    y = pd.Series(np.sin(np.arange(64) / 10), name="target")
    f = TimeH2OForecaster(device="cpu", license_accepted=True).fit(y)
    pred = f.predict([1, 3])
    quantiles = f.predict_quantiles([1, 3], alpha=[0.1, 0.5, 0.9])
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred, quantiles[("target", 0.5)])

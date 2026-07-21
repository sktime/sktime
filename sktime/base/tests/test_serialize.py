# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for sktime serialization utilities."""

__author__ = ["geetu040"]

import json
import pickle
from io import BytesIO
from zipfile import ZipFile, is_zipfile

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.base import BaseObject, load
from sktime.base._serialize import (
    _KerasArtifactBackend,
    _LightningCheckpointArtifactBackend,
    _PretrainedArtifactBackend,
    _TorchStateDictArtifactBackend,
)


# test fixtures
class _TestObject(BaseObject):
    """Simple object using standard pickle serialization."""

    def __init__(self, value=42):
        self.value = value
        super().__init__()


class _ObjectWithCache(BaseObject):
    """Object with a cache excluded from serialization."""

    _tags = {"serialization:skip": ("cache_",)}

    def __init__(self, value=42):
        self.value = value
        self.cache_ = lambda x: x
        super().__init__()


class _DummyPretrainedArtifact:
    """Dependency-free save_pretrained/from_pretrained test double."""

    def __init__(self, value):
        self.value = value

    def save_pretrained(self, path):
        """Save artifact data."""
        data = json.dumps({"value": self.value})
        (path / "model.json").write_text(data, encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path):
        """Load artifact data."""
        data = (path / "model.json").read_text(encoding="utf-8")
        return cls(json.loads(data)["value"])


class _ObjectWithNativeArtifact(BaseObject):
    """Object with an attribute saved as a native artifact."""

    _tags = {"serialization:native_artifacts": ("model_",)}

    def __init__(self, value="fitted", include_artifact=True):
        self.value = value
        self.include_artifact = include_artifact
        self.model_ = _DummyPretrainedArtifact(value) if include_artifact else None
        super().__init__()


class _ObjectWithUnsupportedArtifact(BaseObject):
    """Object selecting an artifact with no serialization backend."""

    _tags = {"serialization:native_artifacts": ("artifact_",)}

    def __init__(self):
        self.artifact_ = object()
        super().__init__()


# test helpers
def _save_to_file(obj, tmp_path):
    """Save an object to a temporary zip file and return its path."""
    save_path = tmp_path / "object"
    archive = obj.save(save_path)
    archive.close()
    return save_path.with_suffix(".zip")


def _roundtrip(obj, storage, tmp_path):
    """Round-trip an object in memory or through a file."""
    serial = obj.save() if storage == "memory" else _save_to_file(obj, tmp_path)
    return load(serial)


def _artifact_record(obj):
    """Return the class record stored in a native artifact index."""
    cls = type(obj)
    return {"class": f"{cls.__module__}.{cls.__qualname__}"}


def _assert_torch_state_equal(left, right):
    """Assert that two Torch modules contain the same state."""
    import torch

    left_state = left.state_dict()
    right_state = right.state_dict()

    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key].cpu(), right_state[key].cpu())


# standard serialization tests
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_standard_serialization_in_memory(serialization_format):
    """Objects without serialization tags use the existing pickle container."""
    if serialization_format == "cloudpickle":
        pytest.importorskip("cloudpickle")

    obj = _TestObject(value="saved")
    cls, serial = obj.save(serialization_format=serialization_format)

    assert cls is _TestObject
    assert not is_zipfile(BytesIO(serial))
    assert load((cls, serial)).value == "saved"


def test_standard_serialization_to_file(tmp_path):
    """Standard file saves contain only metadata and the pickled object."""
    obj = _TestObject(value="saved")
    zip_path = _save_to_file(obj, tmp_path)

    with ZipFile(zip_path) as file:
        assert set(file.namelist()) == {"_metadata", "_obj"}

    assert load(zip_path).value == "saved"


# serialization tag tests
@pytest.mark.parametrize("storage", ["memory", "file"])
def test_skipped_attribute_is_not_serialized(storage, tmp_path):
    """Skipped attributes are omitted without changing the original object."""
    obj = _ObjectWithCache()
    original_cache = obj.cache_

    loaded = _roundtrip(obj, storage, tmp_path)

    assert not hasattr(loaded, "cache_")
    assert obj.cache_ is original_cache


@pytest.mark.parametrize("storage", ["memory", "file"])
def test_native_artifact_roundtrip(storage, tmp_path):
    """Native artifacts round-trip in memory and through a file."""
    obj = _ObjectWithNativeArtifact(value="model weights")
    original_model = obj.model_

    loaded = _roundtrip(obj, storage, tmp_path)

    assert loaded.model_.value == "model weights"
    assert obj.model_ is original_model


def test_native_artifact_archive_layout(tmp_path):
    """Native artifacts are stored under _artifacts with an index."""
    zip_path = _save_to_file(_ObjectWithNativeArtifact(), tmp_path)

    with ZipFile(zip_path) as file:
        files = {name for name in file.namelist() if not name.endswith("/")}
        index = json.loads(file.read("_artifacts/index.json"))
        stored_obj = pickle.loads(file.read("_obj"))

    assert files == {
        "_metadata",
        "_obj",
        "_artifacts/index.json",
        "_artifacts/model_/model.json",
    }
    assert index["model_"]["backend"] == "pretrained"
    assert index["model_"]["path"] == "model_"
    assert not hasattr(stored_obj, "model_")


def test_none_native_artifact_is_omitted(tmp_path):
    """A native artifact set to None does not create an _artifacts directory."""
    obj = _ObjectWithNativeArtifact(include_artifact=False)
    zip_path = _save_to_file(obj, tmp_path)

    with ZipFile(zip_path) as file:
        assert set(file.namelist()) == {"_metadata", "_obj"}

    loaded = load(zip_path)
    assert not hasattr(loaded, "model_")


def test_unsupported_artifact_raises_and_restores_object(tmp_path):
    """An unsupported artifact gives a clear error and remains on the source."""
    obj = _ObjectWithUnsupportedArtifact()
    original_artifact = obj.artifact_

    with pytest.raises(TypeError, match="No native serialization backend"):
        obj.save(tmp_path / "object")

    assert obj.artifact_ is original_artifact


# input validation tests
@pytest.mark.parametrize("serialization_format", [None, "json", "joblib"])
def test_invalid_serialization_format_raises(serialization_format):
    """Unsupported serialization formats raise a ValueError."""
    with pytest.raises(ValueError, match="not yet supported"):
        _TestObject().save(serialization_format=serialization_format)


def test_invalid_save_path_raises():
    """The save path must be a string, pathlib Path, or None."""
    with pytest.raises(TypeError, match="string or a Path object"):
        _TestObject().save(42)


def test_invalid_memory_container_raises():
    """An in-memory serialization container must have exactly two items."""
    with pytest.raises(ValueError, match="tuple of size 2"):
        load((_TestObject, b"data", b"extra"))


def test_invalid_load_type_raises():
    """Load rejects unsupported input types."""
    with pytest.raises(TypeError, match="serial must either"):
        load(42)


# native artifact backend tests
PRETRAINED_AVAILABLE = _check_soft_dependencies(
    ["transformers", "torch"], severity="none"
)
PEFT_AVAILABLE = _check_soft_dependencies(
    ["peft", "transformers", "torch"], severity="none"
)
KERAS_AVAILABLE = _check_soft_dependencies("tensorflow", severity="none")
LIGHTNING_AVAILABLE = _check_soft_dependencies(["lightning", "torch"], severity="none")
TORCH_AVAILABLE = _check_soft_dependencies("torch", severity="none")


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="requires transformers and torch")
def test_pretrained_artifact_backend(tmp_path):
    """Pretrained models are saved and loaded with their native API."""
    from transformers import BertConfig, BertModel

    config = BertConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        vocab_size=32,
    )
    model = BertModel(config)
    backend = _PretrainedArtifactBackend()
    path = tmp_path / "pretrained"
    path.mkdir()

    assert backend.supports(model)
    backend.save(model, path, estimator=None, name="model")
    loaded = backend.load(
        path,
        _artifact_record(model),
        estimator=None,
        name="model",
    )

    assert (path / "config.json").exists()
    _assert_torch_state_equal(model, loaded)


@pytest.mark.skipif(not PEFT_AVAILABLE, reason="requires peft, transformers and torch")
def test_pretrained_artifact_backend_peft(tmp_path, monkeypatch):
    """PEFT adapters retain their special base-model loading path."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import BertConfig, BertModel

    config = BertConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        vocab_size=32,
    )
    model = get_peft_model(
        BertModel(config),
        LoraConfig(target_modules=["query", "value"]),
    )

    class _ArtifactFactory:
        @staticmethod
        def _get_native_artifact_load_kwargs(name):
            return {"model": BertModel(config)}

    backend = _PretrainedArtifactBackend()
    path = tmp_path / "peft"
    path.mkdir()

    assert backend.supports(model)
    backend.save(model, path, estimator=None, name="model")
    monkeypatch.setattr(backend, "_load_class", lambda record: type(model))
    loaded = backend.load(
        path,
        {},
        estimator=_ArtifactFactory(),
        name="model",
    )

    expected = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    actual = {k: v for k, v in loaded.state_dict().items() if "lora_" in k}
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


@pytest.mark.skipif(not KERAS_AVAILABLE, reason="requires tensorflow")
def test_keras_artifact_backend(tmp_path):
    """Keras models are saved and loaded as .keras files."""
    import numpy as np
    from tensorflow import keras

    model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])
    model(np.zeros((1, 2)))
    backend = _KerasArtifactBackend()
    estimator = _TestObject()
    path = tmp_path / "keras"
    path.mkdir()

    assert backend.supports(model)
    backend.save(model, path, estimator=estimator, name="model")
    loaded = backend.load(
        path,
        _artifact_record(model),
        estimator=estimator,
        name="model",
    )

    assert (path / "model.keras").exists()
    for expected, actual in zip(model.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="requires lightning and torch")
def test_lightning_artifact_backend(tmp_path):
    """Lightning modules are saved and loaded from checkpoints."""
    from lightning.pytorch.demos.boring_classes import BoringModel

    model = BoringModel()
    backend = _LightningCheckpointArtifactBackend()
    path = tmp_path / "lightning"
    path.mkdir()

    assert backend.supports(model)
    backend.save(model, path, estimator=None, name="model")
    loaded = backend.load(
        path,
        _artifact_record(model),
        estimator=None,
        name="model",
    )

    assert (path / "model.ckpt").exists()
    _assert_torch_state_equal(model, loaded)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
def test_torch_state_dict_artifact_backend(tmp_path):
    """Torch modules are rebuilt and loaded from CPU state dictionaries."""
    import torch

    class _ArtifactFactory:
        @staticmethod
        def _create_torch_artifact(name):
            assert name == "network"
            return torch.nn.Linear(2, 1)

    model = torch.nn.Linear(2, 1)
    backend = _TorchStateDictArtifactBackend()
    estimator = _ArtifactFactory()
    path = tmp_path / "torch"
    path.mkdir()

    assert backend.supports(model)
    backend.save(model, path, estimator=estimator, name="network")
    loaded = backend.load(
        path,
        _artifact_record(model),
        estimator=estimator,
        name="network",
    )

    assert (path / "state_dict.pt").exists()
    _assert_torch_state_equal(model, loaded)

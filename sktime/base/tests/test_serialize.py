# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for sktime estimator serialization, version metadata, and warnings."""

import pickle
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from sktime import __version__ as sktime_version
from sktime.base import load
from sktime.forecasting.naive import NaiveForecaster


def test_save_includes_version_metadata():
    """Test that saving an estimator includes the sktime version in metadata."""
    forecaster = NaiveForecaster()

    # 1. In-memory save
    serial_mem = forecaster.save()
    assert isinstance(serial_mem, tuple)
    meta_mem, _ = serial_mem
    assert isinstance(meta_mem, dict)
    assert meta_mem["class"] == NaiveForecaster
    assert meta_mem["sktime_version"] == sktime_version

    # 2. File-based save
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_forecaster"
        forecaster.save(save_path)

        zip_path = Path(f"{save_path}.zip")
        assert zip_path.exists()

        with ZipFile(zip_path, "r") as zip_file:
            meta_file = pickle.loads(zip_file.open("_metadata").read())

        assert isinstance(meta_file, dict)
        assert meta_file["class"] == NaiveForecaster
        assert meta_file["sktime_version"] == sktime_version


def test_load_same_version_no_warning():
    """Test that loading an estimator saved with the same version emits no warning."""
    forecaster = NaiveForecaster()
    serial = forecaster.save()

    with pytest.warns(None) as record:
        loaded = load(serial)

    # Filter for UserWarning
    user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0
    assert isinstance(loaded, NaiveForecaster)


def test_load_different_version_emits_warning():
    """Test that loading an estimator saved with a different version warns user."""
    forecaster = NaiveForecaster()
    _, stored_bytes = forecaster.save()

    # Create serial tuple with different version
    fake_meta = {
        "class": NaiveForecaster,
        "sktime_version": "0.0.1.fake",
    }
    fake_serial = (fake_meta, stored_bytes)

    with pytest.warns(UserWarning, match="saved with sktime version 0.0.1.fake"):
        loaded = load(fake_serial)

    assert isinstance(loaded, NaiveForecaster)


def test_load_legacy_metadata_emits_warning():
    """Test that loading an estimator saved without version metadata warns user."""
    forecaster = NaiveForecaster()
    _, stored_bytes = forecaster.save()

    # Legacy serial tuple where first element was raw class type
    legacy_serial = (NaiveForecaster, stored_bytes)

    with pytest.warns(UserWarning, match="saved without sktime version metadata"):
        loaded = load(legacy_serial)

    assert isinstance(loaded, NaiveForecaster)

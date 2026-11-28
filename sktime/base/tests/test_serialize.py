# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for file serialization through the public BaseObject API."""

from pathlib import Path

import pytest

from sktime.base import BaseObject, load


@pytest.mark.parametrize("path_type", [str, Path])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
@pytest.mark.parametrize("filename", ["model", "model.v1", "model.v1.2", "model.zip"])
def test_save_returns_created_archive(
    tmp_path, path_type, serialization_format, filename
):
    """Test that save returns its created archive for names containing dots."""
    if serialization_format == "cloudpickle":
        pytest.importorskip("cloudpickle")

    obj = BaseObject().set_tags(serialization_test_marker="new")
    save_path = tmp_path / filename
    expected_archive = tmp_path / f"{filename}.zip"

    with obj.save(
        path_type(save_path), serialization_format=serialization_format
    ) as archive:
        assert Path(archive.filename) == expected_archive
        assert set(archive.namelist()) == {"_metadata", "_obj"}
        restored = load(Path(archive.filename))
        assert restored.get_tag("serialization_test_marker") == "new"


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_save_dotted_name_does_not_return_existing_archive(
    tmp_path, serialization_format
):
    """Test that a shorter existing archive cannot replace the returned object."""
    if serialization_format == "cloudpickle":
        pytest.importorskip("cloudpickle")

    old = BaseObject().set_tags(serialization_test_marker="old")
    with old.save(tmp_path / "model", serialization_format=serialization_format):
        pass

    new = BaseObject().set_tags(serialization_test_marker="new")
    with new.save(
        tmp_path / "model.v1", serialization_format=serialization_format
    ) as archive:
        restored = load(Path(archive.filename))
        assert restored.get_tag("serialization_test_marker") == "new"

    original = load(tmp_path / "model.zip")
    assert original.get_tag("serialization_test_marker") == "old"

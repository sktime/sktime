"""Tests for the FPP3 dataset loader's download/extraction safety."""

import io
import os
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from sktime.datasets._fpp3_loaders import (
    _DOWNLOAD_TIMEOUT,
    _decompress_file_to_temp,
    _is_within_directory,
    _safe_extract_tar,
)

__author__ = ["WAHIB-EL-KHADIRI"]


def _make_tar(tmp_path, members):
    """Build a tar.gz file at tmp_path containing the given (name, is_link) members."""
    archive_path = tmp_path / "archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for name, is_link in members:
            if is_link:
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar.addfile(info)
            else:
                info = tarfile.TarInfo(name=name)
                data = b"hello"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return archive_path


def test_is_within_directory_accepts_nested_path(tmp_path):
    """A normal, nested member path resolves inside the target directory."""
    target = tmp_path / "sub" / "file.csv"
    assert _is_within_directory(tmp_path, target)


def test_is_within_directory_rejects_parent_traversal(tmp_path):
    """A `..`-traversal member path resolves outside the target directory."""
    outside = tmp_path.parent / "outside.csv"
    assert not _is_within_directory(tmp_path, outside)


def test_safe_extract_tar_rejects_path_traversal_member(tmp_path):
    """A tar member using `../` to escape the extraction directory is rejected."""
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    archive_path = _make_tar(tmp_path, [("../../outside.csv", False)])

    with tarfile.open(archive_path) as tar:
        with pytest.raises((RuntimeError, tarfile.OutsideDestinationError)):
            _safe_extract_tar(tar, str(extract_dir))

    assert not (tmp_path / "outside.csv").exists()
    assert not (tmp_path.parent / "outside.csv").exists()


def test_safe_extract_tar_rejects_symlink_member_on_legacy_fallback(tmp_path):
    """The manual fallback path (pre-PEP-706 Pythons) rejects link members.

    Forces the legacy fallback by removing `tarfile.data_filter`, which is how
    an interpreter without PEP 706 support presents.
    """
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    archive_path = _make_tar(tmp_path, [("link.csv", True)])

    had_filter = hasattr(tarfile, "data_filter")
    original = getattr(tarfile, "data_filter", None)
    if had_filter:
        del tarfile.data_filter
    try:
        with tarfile.open(archive_path) as tar:
            with pytest.raises(RuntimeError, match="link member"):
                _safe_extract_tar(tar, str(extract_dir))
    finally:
        if had_filter:
            tarfile.data_filter = original


def test_safe_extract_tar_does_not_mistake_internal_typeerror_for_no_filter(tmp_path):
    """A `TypeError` from *inside* extraction must propagate, not silently
    retry with an unfiltered `extractall`.

    Probing `tarfile.data_filter` instead of catching `TypeError` around the
    extraction is what makes this hold: the earlier version would have treated
    any internal `TypeError` as "this interpreter has no filter support" and
    re-extracted the archive unfiltered over a partially populated directory.
    """
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    archive_path = _make_tar(tmp_path, [("data/file.csv", False)])

    if not hasattr(tarfile, "data_filter"):
        pytest.skip("interpreter has no PEP 706 filter support")

    unfiltered_calls = []

    with tarfile.open(archive_path) as tar:
        # Raise only for the *filtered* call, and record (without raising) any
        # unfiltered one. An implementation that treats the TypeError as
        # "filter unsupported" falls through to the unfiltered path and
        # swallows the error; this asserts it does not.
        def fake_extractall(*args, **kwargs):
            if "filter" in kwargs:
                raise TypeError("boom from inside extraction")
            unfiltered_calls.append(kwargs)

        with patch.object(tar, "extractall", side_effect=fake_extractall):
            with pytest.raises(TypeError, match="boom from inside extraction"):
                _safe_extract_tar(tar, str(extract_dir))

    assert unfiltered_calls == [], (
        "internal TypeError was mistaken for missing filter support and "
        "retried with an unfiltered extractall"
    )


def test_safe_extract_tar_extracts_normal_members(tmp_path):
    """A well-behaved archive extracts normally through the safe path."""
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    archive_path = _make_tar(tmp_path, [("data/file.csv", False)])

    with tarfile.open(archive_path) as tar:
        _safe_extract_tar(tar, str(extract_dir))

    assert (extract_dir / "data" / "file.csv").exists()


def test_decompress_file_to_temp_passes_explicit_timeout():
    """`requests.get` is always called with an explicit (connect, read) timeout."""
    mock_response = MagicMock()
    mock_response.content = b""
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response) as mock_get:
        with patch(
            "sktime.datasets._fpp3_loaders.tarfile.open",
            side_effect=tarfile.ReadError("not a tar file"),
        ):
            with pytest.raises(tarfile.ReadError):
                _decompress_file_to_temp(datafile="doesnotmatter.tar.gz")

    assert mock_get.called
    _, kwargs = mock_get.call_args
    assert kwargs.get("timeout") == _DOWNLOAD_TIMEOUT


def test_decompress_file_to_temp_cleans_up_on_extraction_failure():
    """The temp directory created for the download is removed if extraction fails."""
    mock_response = MagicMock()
    mock_response.content = b""
    mock_response.raise_for_status = MagicMock()

    created_dirs = []
    import tempfile as tempfile_module

    original_mkdtemp = tempfile_module.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created_dirs.append(path)
        return path

    with patch("requests.get", return_value=mock_response):
        with patch(
            "sktime.datasets._fpp3_loaders.tarfile.open",
            side_effect=tarfile.ReadError("not a tar file"),
        ):
            with patch(
                "sktime.datasets._fpp3_loaders.tempfile.mkdtemp",
                side_effect=tracking_mkdtemp,
            ):
                with pytest.raises(tarfile.ReadError):
                    _decompress_file_to_temp(datafile="doesnotmatter.tar.gz")

    assert len(created_dirs) == 1
    assert not os.path.exists(created_dirs[0])

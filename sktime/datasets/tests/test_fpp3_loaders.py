"""Tests for the FPP3 dataset loader internals.

These tests are fully mocked and do not touch the network.
"""

import io
import os
import tarfile
from unittest import mock

import pytest

from sktime.datasets._fpp3_loaders import (
    _decompress_file_to_temp,
    _safe_extract_tar,
)


def _make_tar_bytes(member_names):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in member_names:
            data = b"dummy dataset content"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf.getvalue()


def _mock_response(tar_bytes):
    response = mock.Mock()
    response.content = tar_bytes
    response.raise_for_status = mock.Mock()
    return response


def _track_mkdtemp(monkeypatch):
    """Record every temp dir created by the module under test."""
    import sktime.datasets._fpp3_loaders as fpp3

    created = []
    real_mkdtemp = __import__("tempfile").mkdtemp

    def spy(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(fpp3.tempfile, "mkdtemp", spy)
    return created


def test_decompress_passes_timeout_to_requests(tmp_path, monkeypatch):
    created = _track_mkdtemp(monkeypatch)
    tar_bytes = _make_tar_bytes(["aus_accommodation.rda"])
    with mock.patch("requests.get", return_value=_mock_response(tar_bytes)) as get:
        temp_dir = _decompress_file_to_temp(
            datafile="fpp3_1.0.1.tar.gz", temp_folder=str(tmp_path), robust=True
        )
    assert temp_dir in created
    assert os.path.isdir(temp_dir)
    assert os.path.isfile(os.path.join(temp_dir, "aus_accommodation.rda"))
    for call in get.call_args_list:
        assert call.kwargs["timeout"] == (10, 60)


def test_safe_extract_rejects_path_traversal(tmp_path):
    tar_path = tmp_path / "evil.tar.gz"
    tar_path.write_bytes(_make_tar_bytes(["../../outside.txt"]))
    with pytest.raises(RuntimeError, match="outside"):
        _safe_extract_tar(str(tar_path), str(tmp_path))
    assert not (tmp_path.parent / "outside.txt").exists()


def test_decompress_rejects_path_traversal_and_cleans_up(tmp_path, monkeypatch):
    created = _track_mkdtemp(monkeypatch)
    tar_bytes = _make_tar_bytes(["../../outside.txt"])
    with mock.patch("requests.get", return_value=_mock_response(tar_bytes)):
        with pytest.raises(RuntimeError, match="outside"):
            _decompress_file_to_temp(
                datafile="fpp3_1.0.1.tar.gz", temp_folder=str(tmp_path), robust=True
            )
    assert len(created) == 1
    assert not os.path.exists(created[0])
    assert not (tmp_path.parent / "outside.txt").exists()


def test_decompress_failed_download_robust_false_cleans_up(tmp_path, monkeypatch):
    import requests

    created = _track_mkdtemp(monkeypatch)
    with mock.patch("requests.get", side_effect=requests.exceptions.ConnectTimeout):
        result = _decompress_file_to_temp(
            datafile="fpp3_1.0.1.tar.gz", temp_folder=str(tmp_path), robust=False
        )
    assert result is None
    assert len(created) == 1
    assert not os.path.exists(created[0])


def test_decompress_failed_download_robust_true_cleans_up(tmp_path, monkeypatch):
    import requests

    created = _track_mkdtemp(monkeypatch)
    with mock.patch("requests.get", side_effect=requests.exceptions.ConnectTimeout):
        with pytest.raises(RuntimeError, match="Failed to download"):
            _decompress_file_to_temp(
                datafile="fpp3_1.0.1.tar.gz",
                archivedir="fpp3",
                temp_folder=str(tmp_path),
                robust=True,
            )
    assert len(created) == 1
    assert not os.path.exists(created[0])

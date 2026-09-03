# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for isolated environment managers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.env_managers import (
    BaseEnvironmentManager,
    UvEnvironmentManager,
    dependency_env_key,
    env_python,
)
from sktime.utils.env_managers._base import resolve_run_target

__author__ = ["jgyasu"]


class _DummyManager(BaseEnvironmentManager):
    """Minimal manager that points at the current interpreter."""

    def __init__(self, python=None):
        self.python = Path(python or sys.executable)
        self.seen_requirements = None
        self.seen_python = None
        super().__init__()

    def get_python_executable(self, requirements=None, python=None):
        self.seen_requirements = list(requirements or [])
        self.seen_python = python
        return self.python


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_dependency_env_key_is_order_invariant():
    """Same requirements hash regardless of order or duplicates."""
    assert dependency_env_key(["b", "a"]) == dependency_env_key(["a", "b", "a"])
    assert dependency_env_key(["a"]) != dependency_env_key(["b"])


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_env_python_platform_path():
    """Python executable lives in bin/ or Scripts/ depending on platform."""
    env_dir = Path("dummy_env")
    path = env_python(env_dir)
    if sys.platform == "win32":
        assert path == env_dir / "Scripts" / "python.exe"
    else:
        assert path == env_dir / "bin" / "python"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
@pytest.mark.parametrize(
    "target, expected",
    [
        (lambda: None, "callable"),
        (Path("script.py"), "script"),
        ("script.py", "script"),
        ("sktime.benchmarking._worker", "module"),
    ],
)
def test_resolve_run_target(target, expected):
    """Classify scripts, modules, and callables."""
    assert resolve_run_target(target) == expected


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_resolve_existing_file_as_script(tmp_path):
    """An existing file without a .py suffix is still a script."""
    script = tmp_path / "runner"
    script.write_text("print(1)\n", encoding="utf-8")
    assert resolve_run_target(str(script)) == "script"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_run_module_builds_command(monkeypatch):
    """Module targets are launched with ``python -m``."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("sktime.utils.env_managers._base.subprocess.run", fake_run)

    manager = _DummyManager()
    manager.run(
        "sktime.benchmarking._worker",
        requirements=["pandas"],
        input=b"payload",
        args=["--verbose"],
        check=False,
    )

    assert captured["cmd"] == [
        str(Path(sys.executable)),
        "-m",
        "sktime.benchmarking._worker",
        "--verbose",
    ]
    assert captured["kwargs"]["input"] == b"payload"
    assert manager.seen_requirements == ["pandas"]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_run_script_builds_command(monkeypatch, tmp_path):
    """Script targets are launched as ``python script.py``."""
    script = tmp_path / "job.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("sktime.utils.env_managers._base.subprocess.run", fake_run)

    _DummyManager().run(script, args=["--flag"], input="stdin-text")

    assert captured["cmd"] == [str(Path(sys.executable)), str(script), "--flag"]
    assert captured["kwargs"]["input"] == "stdin-text"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers")
    or not _check_soft_dependencies("cloudpickle", severity="none"),
    reason="run test only if env_managers changed and cloudpickle is present",
)
def test_run_callable_builds_payload(monkeypatch):
    """Callables are serialized and executed via the function worker."""
    import cloudpickle

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["payload"] = cloudpickle.loads(kwargs["input"])
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("sktime.utils.env_managers._base.subprocess.run", fake_run)

    def add(x, y, z=0):
        return x + y + z

    manager = _DummyManager()
    manager.run(add, args=(1, 2), kwargs={"z": 3}, input=10)

    assert captured["cmd"] == [
        str(Path(sys.executable)),
        "-m",
        "sktime.utils.env_managers._func_worker",
    ]
    assert captured["payload"]["args"] == (1, 2)
    assert captured["payload"]["kwargs"] == {"z": 3}
    assert captured["payload"]["input"] == 10
    assert "cloudpickle" in manager.seen_requirements


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_uv_manager_requires_uv_executable(tmp_path, monkeypatch):
    """Missing uv executable raises a clear error."""
    monkeypatch.setattr("sktime.utils.env_managers._uv.shutil.which", lambda _: None)
    manager = UvEnvironmentManager(envs_dir=tmp_path, uv_executable=None)
    with pytest.raises(RuntimeError, match="uv"):
        manager.get_python_executable(["pandas"])


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_uv_manager_reuses_ready_environment(tmp_path, monkeypatch):
    """A ready environment is reused without calling uv."""
    manager = UvEnvironmentManager(
        envs_dir=tmp_path,
        uv_executable="/fake/uv",
        base_requirements=["cloudpickle"],
    )
    requirements = ["pandas"]
    env_dir = tmp_path / manager._env_key(requirements)
    python_path = env_python(env_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    env_dir.joinpath(".env_ready").write_text(
        manager._env_key(requirements),
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called for a ready env")

    monkeypatch.setattr("sktime.utils.env_managers._uv.subprocess.run", fail_run)
    assert manager.get_python_executable(requirements) == python_path


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_uv_manager_create_env_invokes_uv(tmp_path, monkeypatch):
    """Creating an environment calls uv venv and uv pip install."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if len(cmd) >= 3 and cmd[1] == "venv":
            Path(cmd[2]).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sktime.utils.env_managers._uv.subprocess.run", fake_run)

    manager = UvEnvironmentManager(
        envs_dir=tmp_path,
        python="3.12",
        uv_executable="/fake/uv",
        base_requirements=["cloudpickle"],
        editable=[tmp_path / "pkg"],
    )
    env_dir = tmp_path / manager._env_key(["pandas"])
    manager._create_env(env_dir, ["pandas"])

    assert calls[0][:3] == ["/fake/uv", "venv", str(env_dir)]
    assert "--python" in calls[0]
    assert calls[1][:3] == ["/fake/uv", "pip", "install"]
    assert "-e" in calls[1]
    assert "cloudpickle" in calls[1]
    assert "pandas" in calls[1]
    assert (env_dir / ".env_ready").exists()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_base_environment_manager_is_abstract():
    """BaseEnvironmentManager cannot be instantiated without get_python_executable."""
    with pytest.raises(TypeError):
        BaseEnvironmentManager()


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_env_key_differs_by_python_version(tmp_path):
    """Same requirements with different Python specs get different env keys."""
    manager = UvEnvironmentManager(envs_dir=tmp_path, uv_executable="/fake/uv")
    reqs = ["numpy"]
    assert manager._env_key(reqs, python="3.11") != manager._env_key(
        reqs, python="3.12"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_create_env_uses_per_call_python(tmp_path, monkeypatch):
    """Per-call python overrides the manager default in ``uv venv``."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if len(cmd) >= 3 and cmd[1] == "venv":
            Path(cmd[2]).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sktime.utils.env_managers._uv.subprocess.run", fake_run)

    manager = UvEnvironmentManager(
        envs_dir=tmp_path,
        python="3.12",
        uv_executable="/fake/uv",
    )
    env_dir = tmp_path / manager._env_key(["pandas"], python="3.11")
    manager._create_env(env_dir, ["pandas"], python="3.11")

    assert calls[0][calls[0].index("--python") + 1] == "3.11"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.env_managers"),
    reason="run test only if env_managers module has changed",
)
def test_run_forwards_python_to_get_python_executable(monkeypatch):
    """``run`` passes ``python`` through to ``get_python_executable``."""

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("sktime.utils.env_managers._base.subprocess.run", fake_run)

    manager = _DummyManager()
    manager.run("script.py", requirements=["numpy"], python="3.10")
    assert manager.seen_python == "3.10"

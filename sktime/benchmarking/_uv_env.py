# ruff: noqa: S603

"""UV virtual environment management for isolated benchmark execution."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
from inspect import isclass
from pathlib import Path

from sktime.base import BaseEstimator
from sktime.benchmarking._benchmarking_dataclasses import TaskObject

logger = logging.getLogger(__name__)

_SKTIME_ROOT = Path(__file__).resolve().parents[2]
_READY_MARKER = ".benchmark_ready"
_REQUIREMENTS_FILE = "requirements.txt"
# Always installed in isolated worker envs (IPC between parent and subprocess).
_BENCHMARK_ENV_REQUIREMENTS = ["cloudpickle"]


def _env_python(env_dir: Path) -> Path:
    """Return the Python executable path for a virtual environment."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _normalize_python_dependencies(deps) -> list[str]:
    """Coerce a ``python_dependencies`` tag value to a flat list of strings."""
    if deps is None:
        return []
    if isinstance(deps, str):
        return [deps] if deps else []
    if isinstance(deps, list):
        result = []
        for dep in deps:
            if isinstance(dep, str) and dep:
                result.append(dep)
            elif isinstance(dep, list) and dep:
                # disjunction: pick the first alternative, same as registry.deps
                first = dep[0]
                if isinstance(first, str) and first:
                    result.append(first)
        return result
    return []


def _object_python_dependencies(obj) -> list[str]:
    """Collect ``python_dependencies`` from a sktime object, class, or dataset."""
    if obj is None:
        return []

    if isclass(obj) and issubclass(obj, BaseEstimator):
        return _normalize_python_dependencies(obj.get_class_tag("python_dependencies"))

    if hasattr(obj, "get_class_tag"):
        return _normalize_python_dependencies(obj.get_class_tag("python_dependencies"))

    if isclass(obj) and hasattr(obj, "get_class_tag"):
        return _normalize_python_dependencies(obj.get_class_tag("python_dependencies"))

    return []


def _dataset_loader_dependencies(dataset_loader) -> list[str]:
    """Collect dependencies required by a dataset loader."""
    if dataset_loader is None:
        return []

    if isclass(dataset_loader):
        return _object_python_dependencies(dataset_loader)

    if hasattr(dataset_loader, "get_class_tag"):
        return _object_python_dependencies(dataset_loader)

    return []


def collect_pair_requirements(estimator: BaseEstimator, task: TaskObject) -> list[str]:
    """Collect PEP 440 dependency strings for a task-estimator pair.

    Dependencies are derived from ``python_dependencies`` tags on the
    estimator and task components (scorers, splitters, dataset loaders).
    """
    requirements: list[str] = []
    requirements.extend(_object_python_dependencies(estimator))

    for scorer in task.scorers:
        requirements.extend(_object_python_dependencies(scorer))

    for component in (
        task.cv_splitter,
        task.cv_global,
        task.cv_global_temporal,
        task.cv_X,
    ):
        requirements.extend(_object_python_dependencies(component))

    requirements.extend(_dataset_loader_dependencies(task.data))

    return sorted(set(requirements))


def dependency_env_key(requirements: list[str]) -> str:
    """Return a stable hash key for a dependency set."""
    payload = "\n".join(sorted(set(requirements)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class UvEnvironmentManager:
    """Create and reuse ``uv`` virtual environments keyed by dependency set.

    Environments are stored under ``envs_dir`` and reused when the same
    dependency set is requested again.
    """

    def __init__(
        self,
        envs_dir: str | Path | None = None,
        python: str | None = None,
        uv_executable: str | None = None,
        sktime_root: str | Path | None = None,
    ):
        self.envs_dir = Path(envs_dir or Path.cwd() / ".benchmark_envs")
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.uv_executable = uv_executable or shutil.which("uv")
        self.sktime_root = Path(sktime_root or _SKTIME_ROOT)

    def get_python_executable(self, requirements: list[str]) -> Path:
        """Get or create an environment for ``requirements`` and return its Python."""
        if self.uv_executable is None:
            raise RuntimeError(
                "isolated benchmark execution requires the `uv` executable on PATH"
            )

        env_dir = self.envs_dir / dependency_env_key(requirements)
        env_python = _env_python(env_dir)

        if self._is_ready(env_dir, requirements):
            logger.debug("Reusing benchmark environment at %s", env_dir)
            return env_python

        logger.info("Creating benchmark environment at %s", env_dir)
        self._create_env(env_dir, requirements)
        return env_python

    def _is_ready(self, env_dir: Path, requirements: list[str]) -> bool:
        env_python = _env_python(env_dir)
        marker = env_dir / _READY_MARKER
        if not env_python.exists() or not marker.exists():
            return False

        stored = marker.read_text(encoding="utf-8").strip()
        return stored == dependency_env_key(requirements)

    def _create_env(self, env_dir: Path, requirements: list[str]) -> None:
        if env_dir.exists():
            shutil.rmtree(env_dir)

        create_cmd = [self.uv_executable, "venv", str(env_dir)]
        if self.python is not None:
            create_cmd.extend(["--python", self.python])

        subprocess.run(create_cmd, check=True, capture_output=True, text=True)

        env_python = _env_python(env_dir)
        install_cmd = [
            self.uv_executable,
            "pip",
            "install",
            "--python",
            str(env_python),
            "-e",
            str(self.sktime_root),
            *_BENCHMARK_ENV_REQUIREMENTS,
        ]
        if requirements:
            install_cmd.extend(requirements)

        subprocess.run(install_cmd, check=True, capture_output=True, text=True)

        env_dir.joinpath(_REQUIREMENTS_FILE).write_text(
            "\n".join([*_BENCHMARK_ENV_REQUIREMENTS, *requirements])
            + ("\n" if requirements or _BENCHMARK_ENV_REQUIREMENTS else ""),
            encoding="utf-8",
        )
        env_dir.joinpath(_READY_MARKER).write_text(
            dependency_env_key(requirements),
            encoding="utf-8",
        )

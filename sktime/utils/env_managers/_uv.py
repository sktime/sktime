# ruff: noqa: S603
"""UV virtual environment manager."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from sktime.utils.env_managers._base import (
    BaseEnvironmentManager,
    dependency_env_key,
    env_python,
)

__author__ = ["jgyasu"]

logger = logging.getLogger(__name__)

_READY_MARKER = ".env_ready"
_REQUIREMENTS_FILE = "requirements.txt"


class UvEnvironmentManager(BaseEnvironmentManager):
    """Create and reuse ``uv`` virtual environments keyed by dependency set.

    Environments are stored under ``envs_dir`` and reused when the same
    dependency set is requested again. The parent process never activates
    these environments; callers use ``get_python_executable`` or ``run``.

    Parameters
    ----------
    envs_dir : str or path-like, optional (default=None)
        Directory for storing virtual environments. Defaults to
        ``".sktime_envs"`` in the current working directory.
    python : str, optional (default=None)
        Default Python interpreter specification passed to
        ``uv venv --python``. Overridden by the ``python`` argument of
        ``get_python_executable`` or ``run``.
    uv_executable : str, optional (default=None)
        Path to the ``uv`` executable. Defaults to the first ``uv`` on
        ``PATH``.
    base_requirements : list of str, optional (default=None)
        Requirement strings installed in every environment created by
        this manager, in addition to per-call ``requirements``.
    editable : list of str or path-like, optional (default=None)
        Local paths installed editable (``uv pip install -e``) in every
        environment created by this manager.
    """

    def __init__(
        self,
        envs_dir: str | Path | None = None,
        python: str | None = None,
        uv_executable: str | None = None,
        base_requirements: list[str] | None = None,
        editable: list[str | Path] | None = None,
    ):
        """Store manager settings and ensure ``envs_dir`` exists."""
        self.envs_dir = Path(envs_dir or Path.cwd() / ".sktime_envs")
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.uv_executable = uv_executable or shutil.which("uv")
        self.base_requirements = list(base_requirements or [])
        self.editable = [Path(path) for path in (editable or [])]

    def get_python_executable(
        self,
        requirements: list[str] | None = None,
        python: str | None = None,
    ) -> Path:
        """Get or create an environment for ``requirements`` and return its Python.

        Parameters
        ----------
        requirements : list of str, optional (default=None)
            PEP 440 requirement strings that select or create the environment.
            Combined with ``base_requirements`` and ``editable`` when hashing
            and installing.
        python : str, optional (default=None)
            Python interpreter specification for this environment, e.g.
            ``"3.11"`` or a path. ``None`` uses ``self.python``.

        Returns
        -------
        pathlib.Path
            Path to the environment's Python executable.

        Raises
        ------
        RuntimeError
            If the ``uv`` executable is not available.
        """
        if self.uv_executable is None:
            raise RuntimeError(
                "UvEnvironmentManager requires the `uv` executable on PATH"
            )

        requirements = list(requirements or [])
        python = self._resolve_python(python)
        env_dir = self.envs_dir / self._env_key(requirements, python=python)
        env_python_path = env_python(env_dir)

        if self._is_ready(env_dir, requirements, python=python):
            logger.debug("Reusing environment at %s", env_dir)
            return env_python_path

        logger.info("Creating environment at %s", env_dir)
        self._create_env(env_dir, requirements, python=python)
        return env_python_path

    def _resolve_python(self, python: str | None) -> str | None:
        """Return the per-call Python spec, falling back to the manager default.

        Parameters
        ----------
        python : str or None
            Per-call interpreter specification.

        Returns
        -------
        str or None
            ``python`` if given, otherwise ``self.python``.
        """
        if python is not None:
            return python
        return self.python

    def _env_key(self, requirements: list[str], python: str | None = None) -> str:
        """Return the reuse key for an environment with ``requirements``.

        The key hashes ``base_requirements``, editable installs, the
        per-call ``requirements``, and the Python spec so two environments
        that share ``envs_dir`` but differ in extras or interpreter do not
        collide.

        Parameters
        ----------
        requirements : list of str
            Per-call PEP 440 requirement strings.
        python : str, optional (default=None)
            Resolved Python interpreter specification included in the hash.

        Returns
        -------
        str
            Stable hash used as the environment directory name.
        """
        python = self._resolve_python(python)
        tokens = self._all_requirements(requirements)
        if python is not None:
            tokens = [*tokens, f"python:{python}"]
        return dependency_env_key(tokens)

    def _all_requirements(self, requirements: list[str]) -> list[str]:
        """Combine manager-level and per-call install tokens.

        Parameters
        ----------
        requirements : list of str
            Per-call PEP 440 requirement strings.

        Returns
        -------
        list of str
            ``base_requirements``, then ``-e <path>`` for each editable
            package, then ``requirements``.
        """
        editable = [f"-e {path}" for path in self.editable]
        return [*self.base_requirements, *editable, *requirements]

    def _is_ready(
        self,
        env_dir: Path,
        requirements: list[str],
        python: str | None = None,
    ) -> bool:
        """Return whether ``env_dir`` is a complete env for ``requirements``.

        An environment is ready when its Python executable exists and the
        ``.env_ready`` marker stores the same key as ``_env_key``.

        Parameters
        ----------
        env_dir : pathlib.Path
            Candidate environment directory.
        requirements : list of str
            Per-call PEP 440 requirement strings.
        python : str, optional (default=None)
            Resolved Python interpreter specification.

        Returns
        -------
        bool
            ``True`` if the environment can be reused as-is.
        """
        env_python_path = env_python(env_dir)
        marker = env_dir / _READY_MARKER
        if not env_python_path.exists() or not marker.exists():
            return False

        stored = marker.read_text(encoding="utf-8").strip()
        return stored == self._env_key(
            requirements, python=self._resolve_python(python)
        )

    def _create_env(
        self,
        env_dir: Path,
        requirements: list[str],
        python: str | None = None,
    ) -> None:
        """Create a ``uv`` virtual environment and install requirements.

        Deletes ``env_dir`` if it already exists, runs ``uv venv``, then
        ``uv pip install`` when there is anything to install. Writes
        ``requirements.txt`` and a ``.env_ready`` marker on success.

        Parameters
        ----------
        env_dir : pathlib.Path
            Directory that will hold the new virtual environment.
        requirements : list of str
            Per-call PEP 440 requirement strings, installed in addition
            to ``base_requirements`` and ``editable``.
        python : str, optional (default=None)
            Interpreter specification passed to ``uv venv --python``.
        """
        if env_dir.exists():
            shutil.rmtree(env_dir)

        python = self._resolve_python(python)
        create_cmd = [self.uv_executable, "venv", str(env_dir)]
        if python is not None:
            create_cmd.extend(["--python", python])

        subprocess.run(create_cmd, check=True, capture_output=True, text=True)

        install_cmd = self._install_command(env_dir, requirements)
        if install_cmd is not None:
            subprocess.run(install_cmd, check=True, capture_output=True, text=True)

        requirement_text = "\n".join(self._all_requirements(requirements))
        if requirement_text:
            requirement_text += "\n"
        env_dir.joinpath(_REQUIREMENTS_FILE).write_text(
            requirement_text,
            encoding="utf-8",
        )
        env_dir.joinpath(_READY_MARKER).write_text(
            self._env_key(requirements, python=python),
            encoding="utf-8",
        )

    def _install_command(
        self, env_dir: Path, requirements: list[str]
    ) -> list[str] | None:
        """Build the ``uv pip install`` command for ``env_dir``.

        Parameters
        ----------
        env_dir : pathlib.Path
            Virtual environment whose interpreter should receive the
            packages.
        requirements : list of str
            Per-call PEP 440 requirement strings.

        Returns
        -------
        list of str or None
            Full ``uv pip install --python ...`` argument vector, or
            ``None`` when there are no packages to install.
        """
        install_items: list[str] = []
        for path in self.editable:
            install_items.extend(["-e", str(path)])
        install_items.extend(self.base_requirements)
        install_items.extend(requirements)
        if not install_items:
            return None

        return [
            self.uv_executable,
            "pip",
            "install",
            "--python",
            str(env_python(env_dir)),
            *install_items,
        ]

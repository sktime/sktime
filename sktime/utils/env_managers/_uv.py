# ruff: noqa: S603
"""UV virtual environment manager."""

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
_DEFAULT_ENVS_DIR = ".temp_envs"


class UvEnvironmentManager(BaseEnvironmentManager):
    """Create and reuse ``uv`` virtual environments keyed by dependency set.

    Environments are stored under ``envs_dir`` and reused when the same
    dependency set and Python spec are requested again. The parent
    process never activates these environments; callers use
    ``get_python_executable`` or ``run``.

    Implements ``get_python_executable`` by creating a ``uv`` venv and
    installing packages with ``uv pip install``. Shared launch logic
    (``run``, ``_prepare_run``, ``_resolve_python``) comes from
    ``BaseEnvironmentManager``.

    Parameters
    ----------
    envs_dir : str or pathlib.Path, optional (default=None)
        Directory for storing virtual environments. A ``str`` is a
        filesystem path, relative to the current working directory
        unless absolute. ``None`` uses ``.temp_envs`` in the current
        working directory.
    python : str, optional (default=None)
        Default interpreter passed to ``uv venv --python``. Valid
        values are a version request (``"3.11"``, ``"3.12.9"``), an
        implementation pin (``"cpython@3.11"``), or an absolute path
        to an interpreter. Overridden by the ``python`` argument of
        ``get_python_executable`` or ``run``. ``None`` uses the
        ``uv venv`` default.
    uv_executable : str, optional (default=None)
        Filesystem path to the ``uv`` executable, or the name of an
        executable on ``PATH``. ``None`` uses the first ``uv`` on
        ``PATH``.
    base_requirements : list of str, optional (default=None)
        PEP 440 specifier strings installed in every environment this
        manager creates, in addition to per-call ``requirements``.
        Same format as the ``python_dependencies`` tag.
        Example: ``["cloudpickle"]``. ``None`` means no extra packages.
    editable : list of str or pathlib.Path, optional (default=None)
        Local project directories installed editable
        (``uv pip install -e``) in every environment. Each entry is
        the project root that contains ``pyproject.toml``, not the
        ``pyproject.toml`` file itself. ``None`` means no editable
        installs.
    """

    def __init__(
        self,
        envs_dir: str | Path | None = None,
        python: str | None = None,
        uv_executable: str | None = None,
        base_requirements: list[str] | None = None,
        editable: list[str | Path] | None = None,
    ):
        """Store manager settings and ensure ``envs_dir`` exists.

        Parameters
        ----------
        envs_dir : str or pathlib.Path, optional (default=None)
            Directory for storing virtual environments. A ``str`` is a
            filesystem path, relative to the current working directory
            unless absolute. ``None`` uses ``.temp_envs`` in the current
            working directory.
        python : str, optional (default=None)
            Default interpreter passed to ``uv venv --python``. Valid
            values are a version request (``"3.11"``, ``"3.12.9"``), an
            implementation pin (``"cpython@3.11"``), or an absolute path
            to an interpreter. Overridden by the ``python`` argument of
            ``get_python_executable`` or ``run``. ``None`` uses the
            ``uv venv`` default.
        uv_executable : str, optional (default=None)
            Filesystem path to the ``uv`` executable, or the name of an
            executable on ``PATH``. ``None`` uses the first ``uv`` on
            ``PATH``.
        base_requirements : list of str, optional (default=None)
            PEP 440 specifier strings installed in every environment this
            manager creates, in addition to per-call ``requirements``.
            Same format as the ``python_dependencies`` tag.
            Example: ``["cloudpickle"]``. ``None`` means no extra packages.
        editable : list of str or pathlib.Path, optional (default=None)
            Local project directories installed editable
            (``uv pip install -e``) in every environment. Each entry is
            the project root that contains ``pyproject.toml``, not the
            ``pyproject.toml`` file itself. ``None`` means no editable
            installs.
        """
        self.envs_dir = Path(envs_dir or Path.cwd() / _DEFAULT_ENVS_DIR)
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.uv_executable = uv_executable or shutil.which("uv")
        self.base_requirements = list(base_requirements or [])
        self.editable = [Path(path) for path in (editable or [])]
        super().__init__()

    def get_python_executable(
        self,
        requirements: list[str] | None = None,
        python: str | None = None,
    ) -> Path:
        """Get or create an environment for ``requirements`` and return its Python.

        Parameters
        ----------
        requirements : list of str, optional (default=None)
            Package requirements, same format as the
            ``python_dependencies`` tag: a list of PEP 440 specifier
            strings. Combined with ``base_requirements`` and ``editable``
            when hashing and installing. ``None`` means no extra packages.
        python : str, optional (default=None)
            Interpreter for this environment: version request
            (``"3.11"``), implementation pin (``"cpython@3.11"``), or
            absolute path. ``None`` uses ``self.python``.

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

# ruff: noqa: S603
"""Base class for isolated environment managers."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

from sktime.base._base import BaseObject

__author__ = ["jgyasu"]

_FUNC_WORKER_MODULE = "sktime.utils.env_managers._func_worker"


def env_python(env_dir: str | Path) -> Path:
    """Return the Python executable path for a virtual environment.

    Parameters
    ----------
    env_dir : str or path-like
        Root directory of a virtual environment.

    Returns
    -------
    pathlib.Path
        ``Scripts/python.exe`` on Windows, otherwise ``bin/python``.
    """
    env_dir = Path(env_dir)
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def dependency_env_key(requirements: Sequence[str]) -> str:
    """Return a stable hash key for a dependency set.

    The key is order-invariant and ignores duplicate requirement strings.
    It is used as the directory name for an isolated environment.

    Parameters
    ----------
    requirements : sequence of str
        PEP 440 requirement strings, plus any extra install tokens such as
        ``"-e /path/to/pkg"``.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 digest of the sorted,
        de-duplicated requirement set.
    """
    payload = "\n".join(sorted(set(requirements)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_run_target(target: str | Path | Callable) -> str:
    """Classify ``target`` as ``"callable"``, ``"script"``, or ``"module"``.

    Parameters
    ----------
    target : str, path-like, or callable
        Object that ``BaseEnvironmentManager.run`` will execute.

        * callable (and not a ``str`` / ``Path``) → ``"callable"``
        * ``pathlib.Path``, existing file, or ``*.py`` path → ``"script"``
        * any other string → ``"module"``

    Returns
    -------
    str
        One of ``"callable"``, ``"script"``, or ``"module"``.
    """
    if callable(target) and not isinstance(target, (str, Path)):
        return "callable"

    if isinstance(target, Path):
        return "script"

    target_str = str(target)
    path = Path(target_str)
    if path.suffix == ".py" or path.exists():
        return "script"

    return "module"


class BaseEnvironmentManager(BaseObject):
    """Base class for managers that create isolated Python environments.

    Concrete subclasses create or reuse an environment for a dependency set
    and return its interpreter. The shared ``run`` method then executes a
    script, module, or callable with that interpreter.
    """

    @abstractmethod
    def get_python_executable(
        self,
        requirements: list[str] | None = None,
        python: str | None = None,
    ) -> Path:
        """Get or create an environment for ``requirements`` and return its Python.

        Parameters
        ----------
        requirements : list of str, optional (default=None)
            PEP 440 requirement strings. ``None`` is treated as an empty
            requirement set.
        python : str, optional (default=None)
            Python interpreter specification for this environment, e.g.
            ``"3.11"`` or a path. ``None`` uses the manager default.

        Returns
        -------
        pathlib.Path
            Path to the environment's Python executable.
        """

    def run(
        self,
        target: str | Path | Callable,
        *,
        requirements: list[str] | None = None,
        python: str | None = None,
        args: Sequence | None = None,
        kwargs: dict | None = None,
        input=None,
        **run_kwargs,
    ) -> subprocess.CompletedProcess:
        """Run a script, module, or callable in an isolated environment.

        Parameters
        ----------
        target : str, path-like, or callable
            What to execute in the environment:

            * callable — serialized and executed in the environment
            * existing file path, ``pathlib.Path``, or ``*.py`` path — run as a
              Python script
            * any other string — run as ``python -m target``

        requirements : list of str, optional (default=None)
            PEP 440 requirement strings that select or create the environment.
        python : str, optional (default=None)
            Python interpreter specification for this environment, e.g.
            ``"3.11"`` or a path. ``None`` uses the manager default.
            Different ``python`` values produce different environments
            even when ``requirements`` are the same. Do not combine a
            different ``python`` with a callable ``target``; see Notes.
        args : sequence, optional (default=None)
            Extra positional arguments. For scripts and modules these are
            passed as command-line arguments. For callables they are passed
            as positional arguments to the function.
        kwargs : dict, optional (default=None)
            Keyword arguments passed to ``target`` when it is a callable.
            Ignored for scripts and modules.
        input : bytes, str, or object, optional
            For scripts and modules, passed as stdin to ``subprocess.run``.
            For callables, passed as the first positional argument.
        **run_kwargs : dict
            Additional keyword arguments forwarded to ``subprocess.run``.
            Defaults are ``capture_output=True`` and ``check=False``.

        Returns
        -------
        subprocess.CompletedProcess
            Result of the subprocess invocation. When ``target`` is a
            callable, ``stdout`` contains a cloudpickled return value.

        Raises
        ------
        ModuleNotFoundError
            If ``target`` is a callable and ``cloudpickle`` is not installed
            in the parent environment.

        Notes
        -----
        Callables are serialized with ``cloudpickle`` in the parent and
        unpickled in the child. That requires the child interpreter to use
        the same Python version as the parent. Passing a different
        ``python`` for a callable target will fail.

        Use a script or module target when environments should use
        different Python versions.
        """
        kind = resolve_run_target(target)
        reqs = list(requirements or [])
        args = list(args or [])
        func_kwargs = dict(kwargs or {})

        if kind == "callable" and "cloudpickle" not in reqs:
            reqs.append("cloudpickle")

        env_python_path = self.get_python_executable(reqs, python=python)
        cmd, stdin = self._prepare_run(
            env_python_path,
            target,
            kind=kind,
            args=args,
            kwargs=func_kwargs,
            input=input,
        )

        defaults = {"capture_output": True, "check": False}
        defaults.update(run_kwargs)
        return subprocess.run(cmd, input=stdin, **defaults)

    def _prepare_run(
        self,
        env_python_path: Path,
        target: str | Path | Callable,
        *,
        kind: str,
        args: list,
        kwargs: dict,
        input,
    ) -> tuple[list[str], bytes | str | None]:
        """Build the subprocess command and stdin payload for ``run``.

        Parameters
        ----------
        env_python_path : pathlib.Path
            Python interpreter of the isolated environment.
        target : str, path-like, or callable
            Script, module, or callable to execute.
        kind : str
            Target kind from ``resolve_run_target``: ``"callable"``,
            ``"script"``, or ``"module"``.
        args : list
            Extra positional arguments. Command-line args for scripts and
            modules; function args for callables.
        kwargs : dict
            Keyword arguments for a callable target. Ignored otherwise.
        input : bytes, str, or object, optional
            Stdin for scripts and modules. For callables, included in the
            serialized payload as the first function argument when not
            ``None``.

        Returns
        -------
        cmd : list of str
            Argument vector passed to ``subprocess.run``.
        stdin : bytes, str, or None
            Value passed as ``input`` to ``subprocess.run``. For callables
            this is a cloudpickled payload; otherwise it is ``input``.

        Raises
        ------
        ModuleNotFoundError
            If ``kind`` is ``"callable"`` and ``cloudpickle`` is not
            installed in the parent environment.
        """
        if kind == "callable":
            from sktime.utils.dependencies import _check_soft_dependencies

            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            payload = {
                "func": target,
                "args": tuple(args),
                "kwargs": kwargs,
            }
            if input is not None:
                payload["input"] = input
            return (
                [str(env_python_path), "-m", _FUNC_WORKER_MODULE],
                cloudpickle.dumps(payload),
            )

        stdin = input
        if kind == "script":
            return [str(env_python_path), str(target), *args], stdin
        return [str(env_python_path), "-m", str(target), *args], stdin

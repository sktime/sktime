# ruff: noqa: S603
"""Base class for isolated environment managers."""

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
    target : str, pathlib.Path, or callable
        Object that ``BaseEnvironmentManager.run`` will execute.

        * callable (and not a ``str`` / ``Path``) → ``"callable"``
        * ``pathlib.Path``, a ``str`` ending in ``.py``, or a ``str``
          that is an existing file path → ``"script"``
        * any other ``str`` → ``"module"`` (executed with ``python -m``)

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
    """Base class for isolated environment managers.

    An environment manager creates or reuses isolated child environments for a
    dependency set and Python interpreter, then runs a target in that
    environment.

    Concrete subclasses implement environment creation and dependency installation.
    A child must implement ``get_python_executable``, which returns the Python
    executable of an environment that has ``requirements`` installed
    and uses the requested ``python`` interpreter.

    Children that accept a default interpreter should store it as
    ``self.python`` (``str`` or ``None``). ``_resolve_python`` then
    uses that as the fallback when a call does not pass ``python``.

    The following methods are shared and should not be overridden
    unless the child launches processes differently from
    ``subprocess.run``:

    * ``run`` — classify ``target``, obtain the interpreter, start the
      child process
    * ``_prepare_run`` — build the child argv and stdin payload
    * ``_resolve_python`` — per-call ``python``, else ``self.python``

    A typical child:

    1. Creates or reuses an isolated environment for ``requirements``
       and ``python``.
    2. Installs ``requirements`` into that environment.
    3. Returns the path to that environment's Python executable from
       ``get_python_executable``.

    ``run`` then starts a subprocess with that executable.
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
            Package requirements to install in the environment, same
            format as the ``python_dependencies`` tag: a list of PEP 440
            specifier strings. Package names are PyPI / ``pip install`` names,
            not import names.

            Valid examples:

            * ``None`` or ``[]``: no extra packages
            * ``["numpy"]``: ``numpy``
            * ``["numpy>=1.26,<2"]``: ``numpy`` 1.26 or higher, and below 2
            * ``["numpy>=1.20.0", "pandas>=1.3.0"]``: both constraints
            * ``["scikit-learn"]: PyPI package name, not import name

        python : str, optional (default=None)
            Interpreter for this environment. Valid values:

            * ``None``: use ``self.python`` if the child set it, otherwise
              the backend default (for ``uv``, the ``uv venv`` default)
            * version request, e.g. ``"3.11"`` or ``"3.12.9"``
            * implementation pin, e.g. ``"cpython@3.11"``
            * absolute path to an interpreter, e.g.
               ``"/usr/bin/python3.11"`` or
               ``"C:/Python311/python.exe"``

        Returns
        -------
        pathlib.Path
            Path to the environment's Python executable.
        """

    def _resolve_python(self, python: str | None = None) -> str | None:
        """Return the per-call Python spec, falling back to ``self.python``.

        Parameters
        ----------
        python : str or None, optional (default=None)
            Per-call interpreter specification. Same valid values as
            ``get_python_executable``'s ``python`` argument.

        Returns
        -------
        str or None
            ``python`` if given, otherwise ``self.python`` if that
            attribute is set, otherwise ``None``.
        """
        if python is not None:
            return python
        return getattr(self, "python", None)

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
        """Run ``target`` in a child process using an isolated environment.

        Obtains that environment's Python via ``get_python_executable``,
        then starts a subprocess. The child command is:

        * script: ``<env_python> <script_path> *args``
        * module: ``<env_python> -m <module> *args``
        * callable: ``<env_python> -m sktime.utils.env_managers._func_worker``
          with a cloudpickled payload on stdin

        Parameters
        ----------
        target : str, pathlib.Path, or callable
            What the child process executes.

            * callable: a Python callable, serialized
              with ``cloudpickle`` in the parent and run in the child.
            * ``pathlib.Path``: always a script. The child runs
              ``<env_python> <path>``.
            * ``str`` ending in ``.py``, or a ``str`` that is an existing
              file path (relative to the current working directory unless
              absolute): treated as a script path.
            * any other ``str``: a module name. The child runs
              ``<env_python> -m <target>``.

        requirements : list of str, optional (default=None)
            Package requirements for the environment, same format as the
            ``python_dependencies`` tag: a list of PEP 440 specifier
            strings. Package names are PyPI / ``pip install`` names.
            ``None`` means no extra packages.
            Example: ``["numpy>=1.26,<2", "pandas>=1.3.0"]``.
        python : str, optional (default=None)
            Interpreter for this environment. Valid values are a version
            request (``"3.11"``, ``"3.12.9"``), an implementation pin
            (``"cpython@3.11"``), or an absolute path to an interpreter.
            ``None`` uses ``self.python`` or the backend default.
            Different ``python`` values produce different environments
            even when ``requirements`` are the same. Do not combine a
            different ``python`` with a callable ``target``; see Notes.
        args : sequence, optional (default=None)
            Extra positional arguments. Meaning depends on ``target``.

            * script: appended after the script path. The child sees
              them as ``sys.argv[1:]``.
            * module: appended after ``-m <module>``. The child sees
              them as ``sys.argv[1:]``.
            * callable: passed as ``*args`` to the function. If
              ``input`` is not ``None``, the call is
              ``func(input, *args, **kwargs)``; otherwise
              ``func(*args, **kwargs)``.
        kwargs : dict, optional (default=None)
            Keyword arguments for a callable ``target``
            (``func(..., **kwargs)``). Ignored for scripts and modules
            (there is no CLI equivalent; pass flag strings via
            ``args``, e.g. ``["--horizon", "12"]``). Not the same as
            ``run_kwargs``, which go to ``subprocess.run``.
        input : bytes, str, or object, optional (default=None)
            Data fed into the child. Meaning depends on ``target``.

            * script or module: stdin of ``subprocess.run``. ``bytes``
              work with the default settings. A ``str`` requires
              ``text=True`` in ``run_kwargs``, otherwise
              ``subprocess.run`` raises ``TypeError``.
            * callable: optional first positional argument. Included
              in the cloudpickled payload only when this value is not
              ``None``. Omitting ``input`` and passing
              ``input=None`` are therefore the same, and both call
              ``func(*args, **kwargs)``. To pass ``None`` as a real
              argument, put it in ``args`` instead.

            For a callable, process stdin is the pickle, not this
            value. Prefer ``args`` / ``kwargs`` alone for callables.
        **run_kwargs : dict
            Additional keyword arguments forwarded to ``subprocess.run``
            (e.g. ``text``, ``check``, ``cwd``, ``env``). Defaults are
            ``capture_output=True`` and ``check=False``. Distinct from
            ``kwargs``, which are for a callable ``target`` only.

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

        Examples
        --------
        For script or module: ``args`` become ``sys.argv[1:]``, ``input`` is
        stdin. ``kwargs`` is ignored.

        >>> manager.run(  # doctest: +SKIP
        ...     "train.py",
        ...     args=["--horizon", "12"],
        ...     input=b"payload",
        ... )
        >>> manager.run(  # doctest: +SKIP
        ...     "sktime.benchmarking._worker",
        ...     args=["--verbose"],
        ... )
        >>> manager.run(  # doctest: +SKIP
        ...     "train.py", input="payload", text=True
        ... )

        For callable: ``args`` / ``kwargs`` are the function call. ``input``
        is an optional first argument when it is not ``None``. This is done
        to keep the signature of ``run`` consistent.

        >>> def add(offset, x, y, z=0):  # doctest: +SKIP
        ...     return offset + x + y + z
        >>> manager.run(  # doctest: +SKIP
        ...     add, args=(1, 2), kwargs={"z": 3}, input=10
        ... )
        >>> # child calls add(10, 1, 2, z=3)
        >>> manager.run(add, args=(1, 2), kwargs={"z": 3})  # doctest: +SKIP
        >>> # child calls add(1, 2, z=3)
        """
        kind = resolve_run_target(target)
        reqs = list(requirements or [])
        args = list(args or [])
        func_kwargs = dict(kwargs or {})

        if kind == "callable" and not any(
            str(req).strip().lower().startswith("cloudpickle") for req in reqs
        ):
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

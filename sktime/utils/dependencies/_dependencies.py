"""Utility to check soft dependency imports, and raise warnings or errors."""

from importlib.metadata import metadata
from importlib.util import find_spec

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from skbase.utils.dependencies._dependencies import (
    _check_soft_dependencies,
    _raise_at_severity,
)

__all__ = [
    "_check_dl_dependencies",
    "_check_mlflow_dependencies",
    "_check_soft_dependencies",
    "_raise_at_severity",
    "_get_lowest_compatible_python_version",
]


def _check_dl_dependencies(msg=None, severity="error"):
    """Check if deep learning dependencies are installed.

    Parameters
    ----------
    msg : str, optional, default= default message (msg below)
        error message to be returned in the ``ModuleNotFoundError``, overrides default

    severity : str, "error" (default), "warning", "none"
        whether the check should raise an error, a warning, or nothing

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
          function returns False if one of packages is not installed, otherwise True

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install deep learning dependencies

    Returns
    -------
    boolean - whether all packages are installed, only if no exception is raised
    """
    if not isinstance(msg, str):
        msg = (
            "tensorflow is required for deep learning functionality in `sktime`. "
            "To install these dependencies, run: `pip install sktime[dl]`"
        )
    if find_spec("tensorflow") is not None:
        return True
    else:
        _raise_at_severity(msg, severity, caller="_check_dl_dependencies")
        return False


def _check_mlflow_dependencies(msg=None, severity="error"):
    """Check if `mlflow` and its dependencies are installed.

    Parameters
    ----------
    msg: str, optional, default= default message (msg below)
        error message to be returned when ``ModuleNotFoundError`` is raised.
    severity: str, either of "error", "warning" or "none"
        behaviour for raising errors or warnings
        "error" - raises a ``ModuleNotFound`` if mlflow-related packages are not found.
        "warning" - raises a warning message if any mlflow-related package is not
            installed also returns False. In case all packages are present,
            returns True.
        "none" - does not raise any exception or warning and simply returns True
            if all packages are installed otherwise return False.

    Raise
    -----
    ModuleNotFoundError
        User Friendly error with a suggested action to install mlflow dependencies

    Returns
    -------
    boolean - whether all mlflow-related packages are installed.
    """
    if not isinstance(msg, str):
        msg = (
            "`mlflow` is an extra dependency and is not included "
            "in the base sktime installation. "
            "Please run `pip install mlflow` "
            "or `pip install sktime[mlflow]` to install the package."
        )

    # we allow mlflow and mlflow-skinny, at least one must be present
    MLFLOW_DEPS = [["mlflow", "mlflow-skinny"]]

    return _check_soft_dependencies(MLFLOW_DEPS, msg=msg, severity=severity)


def _get_lowest_compatible_python_version(estimator):
    """Get the lowest Python version compatible with an estimator and sktime.

    Parameters
    ----------
    estimator : sktime estimator class

    Returns
    -------
    str
        Lowest compatible Python version, e.g. "3.11".
    """
    estimator_spec = estimator.get_class_tag("python_version")
    sktime_spec = metadata("sktime")["Requires-Python"]

    estimator_spec_set = (
        SpecifierSet(estimator_spec) if estimator_spec is not None else SpecifierSet()
    )
    sktime_spec_set = SpecifierSet(sktime_spec)

    major = 3
    minor = 0

    while True:
        version = Version(f"{major}.{minor}")

        if version in sktime_spec_set and version in estimator_spec_set:
            return str(version)

        minor += 1

        # Safety guard against infinite loops if constraints are unsatisfiable
        if minor > 100:
            break

    raise RuntimeError(
        f"No compatible Python version found for "
        f"sktime ({sktime_spec}) and estimator ({estimator_spec})."
    )

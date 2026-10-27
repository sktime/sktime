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


def _check_dl_dependencies(msg=None, severity="warning"):
    """Check if deep learning dependencies are installed.

    Modified to NEVER break docs build.
    """

    if not isinstance(msg, str):
        msg = (
            "tensorflow is required for deep learning functionality in `sktime`. "
            "Install via `pip install sktime[dl]`."
        )

    try:
        if find_spec("tensorflow") is not None:
            return True
        else:
            _raise_at_severity(msg, "warning", caller="_check_dl_dependencies")
            return False
    except Exception:
        # Fail-safe for docs build
        return False


def _check_mlflow_dependencies(msg=None, severity="warning"):
    """Check if `mlflow` dependencies are installed.

    Modified to NEVER break docs build.
    """

    if not isinstance(msg, str):
        msg = (
            "`mlflow` is an optional dependency and is not required "
            "for documentation builds. "
            "Install via `pip install sktime[mlflow]` if needed."
        )

    try:
        # Allow either mlflow or mlflow-skinny
        if find_spec("mlflow") is not None or find_spec("mlflow_skinny") is not None:
            return True
        else:
            _raise_at_severity(msg, "warning", caller="_check_mlflow_dependencies")
            return False
    except Exception:
        # Absolute safety for docs build
        return False
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

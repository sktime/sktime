"""Utility to check soft dependency imports, and raise warnings or errors."""

from importlib.util import find_spec

from skbase.utils.dependencies._dependencies import (
    _check_soft_dependencies,
    _get_installed_packages,
    _raise_at_severity,
)

__all__ = [
    "_check_dl_dependencies",
    "_check_mlflow_dependencies",
    "_check_soft_dependencies",
    "_get_installed_packages",
    "_raise_at_severity",
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
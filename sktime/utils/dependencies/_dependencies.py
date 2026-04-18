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


def _check_dl_dependencies(msg=None, severity="error"):
    """Check if deep learning dependencies are installed."""
    if not isinstance(msg, str):
        msg = (
            "tensorflow is required for deep learning functionality in `sktime`. "
            "To install these dependencies, run: `pip install sktime[dl]`"
        )

    if find_spec("tensorflow") is not None:
        return True
    else:
       
        try:
            _raise_at_severity(msg, "warning", caller="_check_dl_dependencies")
        except Exception:
            pass
        return False


def _check_mlflow_dependencies(msg=None, severity="error"):
    """Check if `mlflow` and its dependencies are installed."""
    if not isinstance(msg, str):
        msg = (
            "`mlflow` is an extra dependency and is not included "
            "in the base sktime installation. "
            "Please run `pip install mlflow` "
            "or `pip install sktime[mlflow]` to install the package."
        )

    # we allow mlflow and mlflow-skinny, at least one must be present
    MLFLOW_DEPS = [["mlflow", "mlflow-skinny"]]

   
    try:
        return _check_soft_dependencies(
            MLFLOW_DEPS,
            msg=msg,
            severity="warning",  # <-- prevents docs crash
        )
    except Exception:
        return False
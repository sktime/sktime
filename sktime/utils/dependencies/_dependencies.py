"""Utility to check soft dependency imports, and raise warnings or errors."""

from importlib.util import find_spec

from skbase.utils.dependencies._dependencies import (
    _check_soft_dependencies,
    _raise_at_severity,
)

__all__ = [
    "_check_dl_dependencies",
    "_check_estimator_deps",
    "_check_mlflow_dependencies",
    "_check_soft_dependencies",
    "_raise_at_severity",
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


# Monkeypatch skbase so that any imports of _check_estimator_deps
# from skbase get our version
import skbase.utils.dependencies
import skbase.utils.dependencies._dependencies

_dependencies_module = skbase.utils.dependencies._dependencies
_orig_check_estimator_deps = _dependencies_module._check_estimator_deps


def _check_estimator_deps(obj, msg=None, severity="error"):
    """Check if object/estimator's package & python requirements are met by python env.

    Additionally checks for 'requires_cython' tag compatibility.
    """
    if isinstance(obj, (list, tuple)):
        compatible = True
        for x in obj:
            x_chk = _check_estimator_deps(x, msg=msg, severity=severity)
            compatible = compatible and x_chk
        return compatible

    compatible = _orig_check_estimator_deps(obj, msg=msg, severity=severity)
    if not compatible:
        return False

    # Check requires_cython tag:
    if hasattr(obj, "get_class_tag"):
        requires_cython = obj.get_class_tag("requires_cython", False)
    elif isinstance(obj, type) and hasattr(obj, "get_class_tags"):
        requires_cython = obj.get_class_tags().get("requires_cython", False)
    else:
        requires_cython = False

    if requires_cython:
        from sktime.utils.dependencies.compile import has_compiler

        if not has_compiler():
            obj_name = obj.__name__ if isinstance(obj, type) else obj.__class__.__name__
            msg_cython = (
                f"C compiler is required for {obj_name} functionality "
                "in `sktime`. Please ensure a C compiler (like GCC or MSVC) "
                "is installed and set up in your environment."
            )
            _raise_at_severity(
                msg_cython,
                severity=severity,
                caller="_check_estimator_deps",
            )
            return False

    return True


skbase.utils.dependencies._check_estimator_deps = _check_estimator_deps
skbase.utils.dependencies._dependencies._check_estimator_deps = _check_estimator_deps

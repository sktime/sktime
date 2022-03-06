# -*- coding: utf-8 -*-
"""Utility to check soft dependency imports, and raise warnings or errors."""
import warnings
from importlib import import_module


def _check_soft_dependencies(*packages, severity="error"):
    """Check if requred soft dependencies are installed and raise error or warning.

    Parameters
    ----------
    packages : str
        One or more package names to check
    severity : str, "error" (default) or "warning"
        whether the check should raise an error, or only a warning

    Raises
    ------
    ModuleNotFoundError
        error with informative message, asking to install required soft dependencies
    """
    for package in packages:
        try:
            import_module(package)
        except ModuleNotFoundError as e:
            msg = (
                f"{e}. '{package}' is a soft dependency and not included in the "
                f"sktime installation. Please run: `pip install {package}`. "
                f"To install all soft dependencies, run: `pip install "
                f"sktime[all_extras]`"
            )
            if severity == "error":
                raise ModuleNotFoundError(msg) from e
            elif severity == "warning":
                warnings.warn(msg)
            else:
                raise RuntimeError(
                    "Error in calling _check_soft_dependencies, severity "
                    f'argument must bee "error" or "warning", found "{severity}".'
                )

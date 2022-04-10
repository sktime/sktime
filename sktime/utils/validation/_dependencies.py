# -*- coding: utf-8 -*-
"""Utility to check soft dependency imports, and raise warnings or errors."""

import io
import sys
import warnings
from importlib import import_module


def _check_soft_dependencies(
    *packages, severity="error", object=None, suppress_import_stdout=False
):
    """Check if required soft dependencies are installed and raise error or warning.

    Parameters
    ----------
    packages : str or tuple of str
        One or more package names to check
    severity : str, "error" (default) or "warning"
        whether the check should raise an error, or only a warning
    object : python object or None, default=None
        if self is passed here when _check_soft_dependencies is called within __init__
        the error message is more informative and will refer to the class
    suppress_import_stdout : bool, optional. Default=False
        whether to suppress stdout printout upon import.

    Raises
    ------
    ModuleNotFoundError
        error with informative message, asking to install required soft dependencies
    """
    for package in packages:
        try:
            if suppress_import_stdout:
                # setup text trap, import, then restore
                sys.stdout = io.StringIO()
                import_module(package)
                sys.stdout = sys.__stdout__
            else:
                import_module(package)
        except ModuleNotFoundError as e:
            if object is None:
                msg = (
                    f"{e}. '{package}' is a soft dependency and not included in the "
                    f"sktime installation. Please run: `pip install {package}` to "
                    f"install the {package} package. "
                    f"To install all soft dependencies, run: `pip install "
                    f"sktime[all_extras]`"
                )
            else:
                class_name = type(object).__name__
                msg = (
                    f"{class_name} requires package '{package}' in python "
                    f"environment to be instantiated, but '{package}' was not found. "
                    f"'{package}' is a soft dependency and not included in the base "
                    f"sktime installation. Please run: `pip install {package}` to "
                    f"install the {package} package. "
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


def _check_dl_dependencies(msg=None):
    """Check if deep learning dependencies are installed.

    Parameters
    ----------
    msg : str, optional, default= default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install deep learning dependencies
    """
    if not isinstance(msg, str):
        msg = (
            "tensorflow and tensorflow-probability are required for "
            "deep learning and probabilistic functionality in `sktime`. "
            "To install these dependencies, run: `pip install sktime[dl]`"
        )
    try:
        import_module("tensorflow")
        import_module("tensorflow_probability")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(msg) from e

# -*- coding: utf-8 -*-
from importlib import import_module


def _check_soft_dependencies(*packages):
    """Check if specified soft dependencies are installed.

    Parameters
    ----------
    packages : str or tuple of str
        One or more package names to check

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install required soft dependencies
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
            raise ModuleNotFoundError(msg) from e


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

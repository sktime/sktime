# -*- coding: utf-8 -*-
from importlib import import_module


def _check_soft_deps(*packages):
    """
    Check if the packages which are soft dependencies
    for sktime are installed

    Parameters
    ----------
    packages : single or multiple packages to check

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to
        install all soft dependencies
    """
    for package in packages:
        try:
            import_module(package)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}. \n"
                f"{package} is a soft dependency in sktime and is not included in the sktime installation.\n"  # noqa: E501
                f"Please run `pip install {package}` or to install all soft dependencies run `pip install sktime[all_extras]`"  # noqa: E501
            ) from e

# -*- coding: utf-8 -*-
from importlib import import_module


def _check_soft_dependencies(*packages):
    """
    Check if all soft dependencies are installed and raise appropriate error message
    when not.

    Parameters
    ----------
    packages : str
        One or more package names to check

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install all required soft
        dependencies
    """
    for package in packages:
        try:
            import_module(package)
        except ModuleNotFoundError as e:
            msg = (
                f"{e}. \n"
                f"{package} is a soft dependency and not included in the "
                f"sktime installation.\n"
                f"Please run `pip install {package}`. Alternatively, to install all "
                f"soft dependencies, run `pip install sktime[all_extras]`"
            )
            raise ModuleNotFoundError(msg) from e

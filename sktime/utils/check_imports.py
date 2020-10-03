# -*- coding: utf-8 -*-
from importlib import import_module


def _check_imports(package):
    try:
        import_module(package)
    except ModuleNotFoundError as e:
        raise Exception(
            f"{e}. \n"
            f"{package} is a soft dependency in sktime and is not included in the sktime installation.\n"  # noqa: E501
            f"Please run `pip install {package}` or to install all soft dependencies run `pip install sktime[all_extras]`"  # noqa: E501
        ) from e

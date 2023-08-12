"""Git related utilities to identify changed modules."""

__author__ = ["fkiraly"]
__all__ = []

import importlib.util
import inspect
import subprocess


def get_module_from_class(cls):
    """Get full parent module string from class.

    Parameters
    ----------
    cls : class
        class to get module string from, e.g., NaiveForecaster

    Returns
    -------
    str : module string, e.g., sktime.forecasting.naive
    """
    module = inspect.getmodule(cls)
    return module.__name__ if module else None


def get_path_from_module(module_str):
    r"""Get local path string from module string.

    Parameters
    ----------
    module_str : str
        module string, e.g., sktime.forecasting.naive

    Returns
    -------
    str : local path string, e.g., sktime\forecasting\naive.py
    """
    try:
        module_spec = importlib.util.find_spec(module_str)
        if module_spec is None:
            raise ImportError(
                f"Error in get_path_from_module, module '{module_str}' not found."
            )
        return module_spec.origin
    except Exception as e:
        raise ImportError(f"Error finding module '{module_str}'") from e


def is_module_changed(module_str):
    """Check if a module has changed compared to the main branch.

    Parameters
    ----------
    module_str : str
        module string, e.g., sktime.forecasting.naive
    """
    module_file_path = get_path_from_module(module_str)
    cmd = f"git diff remotes/origin/main -- {module_file_path}"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        return bool(output)
    except subprocess.CalledProcessError:
        return True


def is_class_changed(cls):
    """Check if a class' parent module has changed compared to the main branch.

    Parameters
    ----------
    cls : class
        class to get module string from, e.g., NaiveForecaster

    Returns
    -------
    bool : True if changed, False otherwise
    """
    module_str = get_module_from_class(cls)
    return is_module_changed(module_str)

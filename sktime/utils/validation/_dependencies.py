# -*- coding: utf-8 -*-
"""Utility to check soft dependency imports, and raise warnings or errors."""

import io
import sys
import warnings
from importlib import import_module
from inspect import isclass


def _check_soft_dependencies(
    *packages,
    package_import_alias=None,
    severity="error",
    object=None,
    suppress_import_stdout=False,
):
    """Check if required soft dependencies are installed and raise error or warning.

    Parameters
    ----------
    packages : str or tuple of str
        One or more package names to check. This needs to be the *package* name,
        i.e., the name of the package on pypi, installed by pip install package
    package_import_alias : dict with str keys and values, optional, default=empty
        key-value pairs are package name, import name
        import name is str used in python import, i.e., from import_name import ...
        should be provided if import name differs from package name
    severity : str, "error" (default), "warning", "none"
        behaviour for raising errors or warnings
        "error" - raises a ModuleNotFoundException if one of packages is not installed
        "warning" - raises a warning if one of packages is not installed
            function returns False if one of packages is not installed, otherwise True
        "none" - does not raise exception or warning
            function returns False if one of packages is not installed, otherwise True
    object : python class, object, str, or None, default=None
        if self is passed here when _check_soft_dependencies is called within __init__,
        or a class is passed when it is called at the start of a single-class module,
        the error message is more informative and will refer to the class/object;
        if str is passed, will be used as name of the class/object or module
    suppress_import_stdout : bool, optional. Default=False
        whether to suppress stdout printout upon import.

    Raises
    ------
    ModuleNotFoundError
        error with informative message, asking to install required soft dependencies

    Returns
    -------
    boolean - whether all packages are installed, only if no exception is raised
    """
    if not all(isinstance(x, str) for x in packages):
        raise TypeError("packages must be str or tuple of str")

    if package_import_alias is None:
        package_import_alias = dict()
    msg = "package_import_alias must be a dict with str keys and values"
    if not isinstance(package_import_alias, dict):
        raise TypeError(msg)
    if not all(isinstance(x, str) for x in package_import_alias.keys()):
        raise TypeError(msg)
    if not all(isinstance(x, str) for x in package_import_alias.values()):
        raise TypeError(msg)

    for package in packages:
        # determine the package import
        if package in package_import_alias.keys():
            package_import_name = package_import_alias[package]
        else:
            package_import_name = package
        # attempt import - if not possible, we know we need to raise warning/exception
        try:
            if suppress_import_stdout:
                # setup text trap, import, then restore
                sys.stdout = io.StringIO()
                import_module(package_import_name)
                sys.stdout = sys.__stdout__
            else:
                import_module(package_import_name)
            return True
        # if package cannot be imported, make the user aware of installation requirement
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
                if not isclass(object):
                    class_name = type(object).__name__
                elif isclass(object):
                    class_name = object.__name__
                elif isinstance(object, str):
                    class_name = object
                else:
                    raise TypeError("object must be a class, an object, a str, or None")
                msg = (
                    f"{class_name} requires package '{package}' to be present "
                    f"in the python environment, but '{package}' was not found. "
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
                return False
            elif severity == "none":
                return False
            else:
                raise RuntimeError(
                    "Error in calling _check_soft_dependencies, severity "
                    'argument must be "error", "warning", or "none",'
                    f'found "{severity}".'
                )


def _check_dl_dependencies(msg=None, severity="error"):
    """Check if deep learning dependencies are installed.

    Parameters
    ----------
    msg : str, optional, default= default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default
    severity : str, "error" (default), "warning", "none"
        behaviour for raising errors or warnings
        "error" - raises a ModuleNotFoundException if one of packages is not installed
        "warning" - raises a warning if one of packages is not installed
            function returns False if one of packages is not installed, otherwise True
        "none" - does not raise exception or warning
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
            "tensorflow and tensorflow-probability are required for "
            "deep learning and probabilistic functionality in `sktime`. "
            "To install these dependencies, run: `pip install sktime[dl]`"
        )
    try:
        import_module("tensorflow")
        import_module("tensorflow_probability")
        return True
    except ModuleNotFoundError as e:
        if severity == "error":
            raise ModuleNotFoundError(msg) from e
        elif severity == "warning":
            warnings.warn(msg)
            return False
        elif severity == "none":
            return False
        else:
            raise RuntimeError(
                "Error in calling _check_dl_dependencies, severity "
                f'argument must be "error", "warning", or "none", found "{severity}".'
            )

# -*- coding: utf-8 -*-
"""Utility to check soft dependency imports, and raise warnings or errors."""

__author__ = ["fkiraly", "mloning"]

import io
import sys
import warnings
from importlib import import_module
from inspect import isclass

from packaging.specifiers import InvalidSpecifier, SpecifierSet


def _check_soft_dependencies(
    *packages,
    package_import_alias=None,
    severity="error",
    obj=None,
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
    obj : python class, object, str, or None, default=None
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
        package_import_alias = {}
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
            if obj is None:
                msg = (
                    f"{e}. '{package}' is a soft dependency and not included in the "
                    f"base sktime installation. Please run: `pip install {package}` to "
                    f"install the {package} package. "
                    f"To install all soft dependencies, run: `pip install "
                    f"sktime[all_extras]`"
                )
            else:
                if not isclass(obj):
                    class_name = type(obj).__name__
                elif isclass(obj):
                    class_name = obj.__name__
                elif isinstance(obj, str):
                    class_name = obj
                else:
                    raise TypeError("obj must be a class, an object, a str, or None")
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


def _check_python_version(obj, package=None, msg=None, severity="error"):
    """Check if system python version is compatible with requirements of obj.

    Parameters
    ----------
    obj : sktime estimator, BaseObject descendant
        used to check python version
    package : str, default = None
        if given, will be used in error message as package name
    msg : str, optional, default = default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default
    severity : str, "error" (default), "warning", or "none"
        whether the check should raise an error, a warning, or nothing

    Returns
    -------
    compatible : bool, whether obj is compatible with system python version
        check is using the python_version tag of obj

    Raises
    ------
    ModuleNotFoundError
        User friendly error if obj has python_version tag that is
        incompatible with the system python version. If package is given,
        error message gives package as the reason for incompatibility.
    """
    est_specifier_tag = obj.get_class_tag("python_version", tag_value_default="None")
    if est_specifier_tag in ["None", None]:
        return True

    try:
        est_specifier = SpecifierSet(est_specifier_tag)
    except InvalidSpecifier:
        msg_version = (
            f"wrong format for python_version tag, "
            f'must be PEP 440 compatible specifier string, e.g., "<3.9, >= 3.6.3",'
            f' but found "{est_specifier_tag}"'
        )
        raise InvalidSpecifier(msg_version)

    # python sys version, e.g., "3.8.12"
    sys_version = sys.version.split(" ")[0]

    if sys_version in est_specifier:
        return True
    # now we know that est_version is not compatible with sys_version

    if not isinstance(msg, str):
        msg = (
            f"{type(obj).__name__} requires python version to be {est_specifier},"
            f" but system python version is {sys.version}."
        )

        if package is not None:
            msg += (
                f" This is due to python version requirements of the {package} package."
            )

    if severity == "error":
        raise ModuleNotFoundError(msg)
    elif severity == "warning":
        warnings.warn(msg)
    elif severity == "none":
        return False
    else:
        raise RuntimeError(
            "Error in calling _check_python_version, severity "
            f'argument must be "error", "warning", or "none", found "{severity}".'
        )
    return True


def _check_estimator_deps(obj, msg=None, severity="error"):
    """Check all dependencies of estimator, packages and python.

    Convenience wrapper around _check_python_version and _check_soft_dependencies,
    checking against estimator tags "python_version", "python_dependencies".

    Parameters
    ----------
    obj : sktime estimator, BaseObject descendant
        used to check python version
    msg : str, optional, default = default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default
    severity : str, "error" (default), "warning", or "none"
        behaviour for raising errors or warnings
        "error" - raises a ModuleNotFoundException if environment is incompatible
        "warning" - raises a warning if environment is incompatible
            function returns False if environment is incompatible, otherwise True
        "none" - does not raise exception or warning
            function returns False if environment is incompatible, otherwise True
    Returns
    -------
    compatible : bool, whether obj is compatible with python environment
        False is returned only if no exception is raised by the function
        checks for python version using the python_version tag of obj
        checks for soft dependencies present using the python_dependencies tag of obj

    Raises
    ------
    ModuleNotFoundError
        User friendly error if obj has python_version tag that is
        incompatible with the system python version.
        Compatible python versions are determined by the "python_version" tag of obj.
        User friendly error if obj has package dependencies that are not satisfied.
        Packages are determined based on the "python_dependencies" tag of obj.
    """
    compatible = True
    compatible = compatible and _check_python_version(obj, severity=severity)

    pkg_deps = obj.get_class_tag("python_dependencies", None)
    if pkg_deps is not None and not isinstance(pkg_deps, list):
        pkg_deps = [pkg_deps]
    if pkg_deps is not None:
        pkg_deps_ok = _check_soft_dependencies(*pkg_deps, severity=severity, obj=obj)
        compatible = compatible and pkg_deps_ok

    return compatible

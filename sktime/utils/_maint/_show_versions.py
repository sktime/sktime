#!/usr/bin/env python3 -u
# License: BSD 3 clause
"""Utility methods to print system info for debugging.

adapted from
:func: `sklearn.show_versions`
"""

__author__ = ["mloning", "fkiraly"]
__all__ = ["show_versions"]

import importlib
import platform
import sys


def _get_sys_info():
    """System information.

    Return
    ------
    sys_info : dict
        system and Python version information
    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


# dependencies to print versions of, by default
DEFAULT_DEPS_TO_SHOW = [
    "pip",
    "sktime",
    "sklearn",
    "skbase",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "joblib",
    "numba",
    "statsmodels",
    "pmdarima",
    "statsforecast",
    "tsfresh",
    "tslearn",
    "torch",
    "tensorflow",
    "tensorflow_probability",
]


def _get_deps_info(deps=None):
    """Overview of the installed version of main dependencies.

    Parameters
    ----------
    deps : optional, list of strings with import names
        if None, behaves as deps = ["sktime"]

    Returns
    -------
    deps_info: dict
        version information on libraries in `deps`
        keys are import names, values are PEP 440 version strings
        of the import as present in the current python environment
    """
    if deps is None:
        deps = ["sktime"]

    def get_version(module):
        return module.__version__

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None

    return deps_info


def show_versions():
    """Print python version, OS version, sktime version, selected dependency versions.

    Pretty prints:

    * python version of environment
    * python executable location
    * OS version
    * list of import name and version number for selected python dependencies

    Developer note:
    Python version/executable and OS version are from `_get_sys_info`
    Package versions are retrieved by `_get_deps_info`
    Selected dependencies are as in the DEFAULT_DEPS_TO_SHOW variable
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info(deps=DEFAULT_DEPS_TO_SHOW)

    print("\nSystem:")  # noqa: T001, T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T001, T201

    print("\nPython dependencies:")  # noqa: T001, T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T001, T201

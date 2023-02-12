#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# License: BSD 3 clause

"""Utility methods to print system info for debugging.

adapted from :func:`sklearn.show_versions`
"""

__author__ = ["mloning"]
__all__ = ["show_versions"]

import importlib
import platform
import sys


def _get_sys_info():
    """
    System information.

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


def _get_deps_info():
    """
    Overview of the installed version of main dependencies.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "pip",
        "setuptools",
        "sklearn",
        "sktime",
        "statsmodels",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "joblib",
        "numba",
        "pmdarima",
        "tsfresh",
    ]

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
    """Print useful debugging information."""
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T001, T201
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))  # noqa: T001, T201

    print("\nPython dependencies:")  # noqa: T001, T201
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))  # noqa: T001, T201

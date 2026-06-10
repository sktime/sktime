#!/usr/bin/env python3 -u
# License: BSD 3 clause
"""Utility methods to print system info for debugging."""

__all__ = ["show_versions"]

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
    "scikit-learn",
    "scikit-base",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "joblib",
    "huggingface-hub",
    "numba",
    "pmdarima",
    "pytorch-forecasting",
    "skforecast",
    "skpro",
    "statsforecast",
    "statsmodels",
    "transformers",
    "tsfresh",
    "tslearn",
    "torch",
    "tensorflow",
]


def _get_deps_info(deps=None):
    """Overview of the installed version of main dependencies.

        Uses ``importlib.distributions``.
        Strings in deps are assumed to be PEP 440 package strings,
        e.g., ``scikit-learn``, not ``sklearn``.

    Parameters
    ----------
    deps : optional, list of strings with package names
        if None, behaves as deps = ["sktime"].

    Returns
    -------
    deps_info: dict
        version information on libraries in `deps`
        keys are package names, import names if source is "import",
        and PEP 440 package strings if source is "distributions";
        values are PEP 440 version strings
        of the import as present in the current python environment
    """
    if deps is None:
        deps = ["sktime"]

    from skbase.utils.dependencies._dependencies import (
        _get_installed_packages,
        _norm_pkgname,
    )

    pkgs = _get_installed_packages(lowercase=True)

    deps_info = {}
    for modname in deps:
        modname_norm = _norm_pkgname(modname)
        deps_info[modname] = pkgs.get(modname_norm, None)

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

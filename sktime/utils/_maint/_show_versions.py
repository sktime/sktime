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
]

# core dependencies of sktime
CORE_DEPS = ["pip", "sktime", "sklearn", "skbase", "numpy", "scipy", "pandas"]


def _get_deps_info(deps=None, source="distributions"):
    """Overview of the installed version of main dependencies.

    Parameters
    ----------
    deps : optional, list of strings with package names
        if None, behaves as deps = ["sktime"].

    source : str, optional one of "distributions" (default) or "import"
        source of version information

        * "distributions" - uses importlib.distributions. In this case,
          strings in deps are assumed to be PEP 440 package strings,
          e.g., scikit-learn, not sklearn.
        * "import" - uses the __version__ attribute of the module.
          In this case, strings in deps are assumed to be import names,
          e.g., sklearn, not scikit-learn.

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

    if source == "distributions":
        from sktime.utils.dependencies._dependencies import _get_installed_packages

        KEY_ALIAS = {"sklearn": "scikit-learn", "skbase": "scikit-base"}

        pkgs = _get_installed_packages()

        deps_info = {}
        for modname in deps:
            pkg_name = KEY_ALIAS.get(modname, modname)
            deps_info[modname] = pkgs.get(pkg_name, None)

        return deps_info

    def get_version(module):
        return getattr(module, "__version__", None)

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except ImportError:
            deps_info[modname] = None
        else:
            ver = get_version(mod)
            deps_info[modname] = ver

    return deps_info


def _get_pkgnames_from_deptag(deptag):
    """Extract package names from dependency tag python_dependencies.

    Parameters
    ----------
    deptag : list, tuple, str
        dependency tag as used in sktime estimators.
        Nested lists and tuples of str, in PEP 440 format, are allowed.

    Returns
    -------
    package_names : set of str
        set of all package names occurring in deptag.
        Version bounds are removed.
    """
    from packaging.requirements import Requirement

    def extract_names(item, package_names):
        if isinstance(item, str):
            package_name = Requirement(item).name
            package_names.add(package_name)
        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                extract_names(sub_item)

    package_names = set()
    extract_names(deptag, package_names)  # mutates package_names
    return package_names


def _get_depstrs_from_estimator(estimator):
    """Extract package names from estimator class tags."""
    deps = estimator.get_class_tags()["python_dependencies"]
    return _get_pkgnames_from_deptag(deps)


def show_versions(estimator=None):
    """Print python version, OS version, sktime version, selected dependency versions.

    Pretty prints:

    * python version of environment
    * python executable location
    * OS version
    * list of import name and version number for selected python dependencies

    If no estimator is passed, a list of default dependencies is shown.

    If an estimator is passed, the dependencies of the estimator are shown,
    plus core dependencies of ``sktime``.

    Parameters
    ----------
    estimator : sktime estimator object, optional
        estimator object to show dependencies for

    Notes
    -----
    Python version/executable and OS version are from ``_get_sys_info``
    Package versions are retrieved by ``_get_deps_info``
    Default dependencies are as in the ``DEFAULT_DEPS_TO_SHOW`` variable
    """
    if estimator is None:
        deps = DEFAULT_DEPS_TO_SHOW
    else:
        deps = _get_depstrs_from_estimator(estimator)
        deps = CORE_DEPS + list(deps)

    sys_info = _get_sys_info()
    deps_info = _get_deps_info(deps=deps)

    print("\nSystem:")  # noqa: T001, T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T001, T201

    print("\nPython dependencies:")  # noqa: T001, T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T001, T201

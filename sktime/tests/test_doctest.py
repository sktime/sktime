# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Doctest checks directed through pytest with conditional skipping."""

import importlib
import inspect
import pkgutil
from functools import lru_cache

from sktime.tests.test_all_estimators import ONLY_CHANGED_MODULES
from sktime.tests.test_switch import run_test_module_changed

EXCLUDE_MODULES_STARTING_WITH = ("all", "test", "contrib", "mlflow")


def _all_functions(module_name):
    """Get all functions from a module, including submodules.

    Excludes:

    * modules starting with 'all' or 'test'.
    * if the flag ``ONLY_CHANGED_MODULES`` is set, modules that have not changed,
      compared to the ``main`` branch.

    Parameters
    ----------
    module_name : str
        Name of the module.

    Returns
    -------
    functions_list : list
        List of tuples (function_name, function_object).
    """
    res = _all_functions_cached(module_name, only_changed_modules=ONLY_CHANGED_MODULES)
    # copy the result to avoid modifying the cached result
    return res.copy()


@lru_cache
def _all_functions_cached(module_name, only_changed_modules=False):
    """Get all functions from a module, including submodules.

    Excludes:

    * modules starting with 'all' or 'test'.
    * if ``only_changed_modules`` is ``True``, modules that have not changed,
      compared to the ``main`` branch.

    Parameters
    ----------
    module_name : str
        Name of the module.
    only_changed_modules : bool, optional (default=False)
        If True, only functions from modules that have changed are returned.

    Returns
    -------
    functions_list : list
        List of tuples (function_name, function_object).
    """
    # Import the package
    package = importlib.import_module(module_name)

    # Initialize an empty list to hold all functions
    functions_list = []

    # Walk through the package's modules
    package_path = package.__path__[0]
    for _, modname, _ in pkgutil.walk_packages(
        path=[package_path], prefix=package.__name__ + "."
    ):
        # Skip modules starting with 'all' or 'test'
        if modname.split(".")[-1].startswith(EXCLUDE_MODULES_STARTING_WITH):
            continue

        # Skip modules that have not changed
        if only_changed_modules and not run_test_module_changed(modname):
            continue

        # Import the module
        module = importlib.import_module(modname)

        # Get all functions from the module
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # if function is imported from another module, skip it
            if obj.__module__ != module.__name__:
                continue
            # add the function to the list
            functions_list.append((name, obj))

    return functions_list


def pytest_generate_tests(metafunc):
    """Test parameterization routine for pytest.

    Fixtures parameterized
    ----------------------
    func : all functions from sktime, as returned by _all_functions
        if ONLY_CHANGED_MODULES is set, only functions from modules that have changed
    """
    # we assume all four arguments are present in the test below
    funcs_and_names = _all_functions("sktime")

    if len(funcs_and_names) > 0:
        funcs, names = zip(*funcs_and_names)

        metafunc.parametrize("func", funcs, ids=names)
    else:
        metafunc.parametrize("func", [])


def test_all_functions_doctest(func):
    """Run doctest for all functions in sktime."""
    import doctest

    doctest.run_docstring_examples(func, globals())

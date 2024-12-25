"""Utility functions for retrieving objects from modules."""

import importlib
import inspect
import pkgutil
from functools import lru_cache

EXCLUDE_MODULES_STARTING_WITH = ("all", "test", "contrib")


def _all_functions(module_name):
    """Get all functions from a module, including submodules.

    Excludes modules starting with 'all' or 'test'.

    Parameters
    ----------
    module_name : str
        Name of the module.

    Returns
    -------
    functions_list : list
        List of tuples (function_name: str, function_object: function).
    """
    # copy to avoid modifying the cache
    return _all_cond(module_name, inspect.isfunction).copy()


def _all_classes(module_name):
    """Get all classes from a module, including submodules.

    Excludes modules starting with 'all' or 'test'.

    Parameters
    ----------
    module_name : str
        Name of the module.

    Returns
    -------
    classes_list : list
        List of tuples (class_name: str, class_ref: class).
    """
    # copy to avoid modifying the cache
    return _all_cond(module_name, inspect.isclass).copy()


@lru_cache
def _all_cond(module_name, cond):
    """Get all objects from a module satisfying a condition.

    The condition should be a hashable callable,
    of signature ``condition(obj) -> bool``.

    Excludes modules starting with 'all' or 'test'.

    Parameters
    ----------
    module_name : str
        Name of the module.
    cond : callable
        Condition to satisfy.
        Signature: ``condition(obj) -> bool``,
        passed as predicate to ``inspect.getmembers``.

    Returns
    -------
    functions_list : list
        List of tuples (function_name, function_object).
    """
    # Import the package
    package = importlib.import_module(module_name)

    # Initialize an empty list to hold all objects
    obj_list = []

    # Walk through the package's modules
    package_path = package.__path__[0]
    for _, modname, _ in pkgutil.walk_packages(
        path=[package_path], prefix=package.__name__ + "."
    ):
        # Skip modules starting with 'all' or 'test'
        if modname.split(".")[-1].startswith(EXCLUDE_MODULES_STARTING_WITH):
            continue

        # Import the module
        module = importlib.import_module(modname)

        # Get all objects from the module
        for name, obj in inspect.getmembers(module, cond):
            # if object is imported from another module, skip it
            if obj.__module__ != module.__name__:
                continue
            # add the object to the list
            obj_list.append((name, obj))

    return obj_list

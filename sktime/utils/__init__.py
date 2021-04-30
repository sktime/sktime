# -*- coding: utf-8 -*-
__author__ = "Markus Löning"
__all__ = ["all_estimators"]

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path


def _get_name(obj):
    return obj.__class__.__name__


def all_estimators(estimator_types=None, return_names=True, exclude_estimators=None):
    """Get a list of all estimators from sktime.

    This function crawls the module and gets all classes that inherit
    from sktime's and sklearn's base classes.

    Not included are: the base classes themselves, classes defined in test
    modules.

    Parameters
    ----------
    estimator_types : string, list of string, optional (default=None)
        Which kind of estimators should be returned.
        - If None, no filter is applied and all estimators are returned.
        - Possible values are 'classifier', 'regressor', 'transformer' and
        'forecaster' to get estimators only of these specific types, or a list of
        these to get the estimators that fit at least one of the types.
    return_names : bool, optional (default=True)
        If True, return estimators as list of (name, estimator) tuples.
        If False, return list of estimators.
    exclude_estimators : str, list of str, optional (default=None)
        Names of estimators to exclude.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual class.

    References
    ----------
    Modified version from scikit-learn's `all_estimators()`.
    """

    # lazy import to avoid circular imports
    import warnings
    from sktime.tests._config import (
        VALID_ESTIMATOR_TYPES,
        VALID_ESTIMATOR_BASE_TYPE_LOOKUP,
    )

    MODULES_TO_IGNORE = ("tests", "setup", "contrib", "benchmarking")
    all_estimators = []
    ROOT = str(Path(__file__).parent.parent)  # sktime package root directory

    def _is_abstract(klass):
        if not (hasattr(klass, "__abstractmethods__")):
            return False
        if not len(klass.__abstractmethods__):
            return False
        return True

    def _is_private_module(module):
        return "._" in module

    def _is_ignored_module(module):
        module_parts = module.split(".")
        return any(part in MODULES_TO_IGNORE for part in module_parts)

    def _is_base_class(name):
        return name.startswith("_") or name.startswith("Base")

    def _is_estimator(name, klass):
        # Check if klass is subclass of base estimators, not an base class itself and
        # not an abstract class
        return (
            issubclass(klass, VALID_ESTIMATOR_TYPES)
            and klass not in VALID_ESTIMATOR_TYPES
            and not _is_abstract(klass)
            and not _is_base_class(name)
        )

    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for _, module_name, _ in pkgutil.walk_packages(path=[ROOT], prefix="sktime."):

            # Filter modules
            if _is_ignored_module(module_name) or _is_private_module(module_name):
                continue

            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)

            # Filter classes
            estimators = [
                (name, klass) for name, klass in classes if _is_estimator(name, klass)
            ]
            all_estimators.extend(estimators)

    # Drop duplicates
    all_estimators = set(all_estimators)

    # Filter based on given estimator types
    def _is_in_estimator_types(estimator, estimator_types):
        return any(
            [
                issubclass(estimator, estimator_type)
                for estimator_type in estimator_types
            ]
        )

    def _check_estimator_types(estimator_types):
        if not isinstance(estimator_types, list):
            estimator_types = [estimator_types]  # make iterable

        def _get_err_msg(estimator_type):
            return (
                f"Parameter `estimator_type` must be None, a string or a list of "
                f"strings. Valid string values are: "
                f"{tuple(VALID_ESTIMATOR_BASE_TYPE_LOOKUP.keys())}, but found: "
                f"{repr(estimator_type)}"
            )

        for i, estimator_type in enumerate(estimator_types):
            if not isinstance(estimator_type, (type, str)):
                raise ValueError(
                    "Please specify `estimator_types` as a list of str or " "types."
                )
            if isinstance(estimator_type, str):
                if estimator_type not in VALID_ESTIMATOR_BASE_TYPE_LOOKUP.keys():
                    raise ValueError(_get_err_msg(estimator_type))
                estimator_type = VALID_ESTIMATOR_BASE_TYPE_LOOKUP[estimator_type]
                estimator_types[i] = estimator_type
            elif isinstance(estimator_type, type):
                pass
            else:
                raise ValueError(_get_err_msg(estimator_type))
        return estimator_types

    if estimator_types is not None:
        estimator_types = _check_estimator_types(estimator_types)
        all_estimators = [
            (name, estimator)
            for name, estimator in all_estimators
            if _is_in_estimator_types(estimator, estimator_types)
        ]

    # Filter based on given exclude list
    if exclude_estimators is not None:
        if not isinstance(exclude_estimators, list):
            exclude_estimators = [exclude_estimators]  # make iterable
        if not all([isinstance(estimator, str) for estimator in exclude_estimators]):
            raise ValueError(
                "Please specify `exclude_estimators` as a list of strings."
            )
        all_estimators = [
            (name, estimator)
            for name, estimator in all_estimators
            if name not in exclude_estimators
        ]

    # Drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    all_estimators = sorted(all_estimators, key=itemgetter(0))

    # Return with string names or only estimator classes
    if return_names:
        return all_estimators
    else:
        return [estimator for (name, estimator) in all_estimators]


def _has_tag(Estimator, tag):
    """Check whether an Estimator has the given tag or not.

    Parameters
    ----------
    Estimator : Estimator class
    tag : str
        An Estimator tag like "skip-inverse-transform"

    Returns
    -------
    bool
    """
    # Check if tag is in all tags
    return Estimator._all_tags().get(tag, False)

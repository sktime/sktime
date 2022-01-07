# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Registry lookup methods.

This module exports the following methods for registry lookup:

all_estimators(estimator_types, filter_tags)
    lookup and filtering of estimators

all_tags(estimator_types)
    lookup and filtering of estimator tags
"""

import inspect
import pkgutil
from copy import deepcopy
from importlib import import_module
from operator import itemgetter
from pathlib import Path

import pandas as pd

from sktime.base import BaseEstimator
from sktime.registry._base_classes import (
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    TRANSFORMER_MIXIN_LIST,
)
from sktime.registry._tags import ESTIMATOR_TAG_REGISTER

VALID_TRANSFORMER_TYPES = tuple(TRANSFORMER_MIXIN_LIST)
VALID_ESTIMATOR_BASE_TYPES = tuple(BASE_CLASS_LIST)

VALID_ESTIMATOR_TYPES = (
    BaseEstimator,
    *VALID_ESTIMATOR_BASE_TYPES,
    *VALID_TRANSFORMER_TYPES,
)


def all_estimators(
    estimator_types=None,
    filter_tags=None,
    exclude_estimators=None,
    return_names=True,
    as_dataframe=False,
):
    """Get a list of all estimators from sktime.

    This function crawls the module and gets all classes that inherit
    from sktime's and sklearn's base classes.

    Not included are: the base classes themselves, classes defined in test
    modules.

    Parameters
    ----------
    estimator_types: string, list of string, optional (default=None)
        Which kind of estimators should be returned.
        if None, no filter is applied and all estimators are returned.
        if str or list of str, strings define scitypes specified in search
                only estimators that are of (at least) one of the scitypes are returned
            possible str values are entries of registry.BASE_CLASS_REGISTER (first col)
                for instance 'classifier', 'regressor', 'transformer', 'forecaster'
    return_names: bool, optional (default=True)
        If True, return estimators as list of (name, estimator class) tuples.
        If False, return list of estimators classes.
    filter_tags: dict of (str or list of str), optional (default=None)
        subsets the returned estimators as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"
    exclude_estimators: str, list of str, optional (default=None)
        Names of estimators to exclude.
    as_dataframe: bool, optional (default=False)
                if False, return is as described below;
                if True, return is converted into a DataFrame for pretty display

    Returns
    -------
    estimators: list of class, if return_names=False,
            or list of tuples (str, class), if return_names=True
        if list of estimators:
            entries are estimator classes matching the query,
            in alphabetical order of class name
        if list of tuples:
            list of (name, class) matching the query,
            in alphabetical order of class name, where
            ``name`` is the estimator class name as string
            ``class`` is the actual class

    References
    ----------
    Modified version from scikit-learn's `all_estimators()`.
    """
    import warnings

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
        warnings.simplefilter("module", category=ImportWarning)
        for _, module_name, _ in pkgutil.walk_packages(path=[ROOT], prefix="sktime."):

            # Filter modules
            if _is_ignored_module(module_name) or _is_private_module(module_name):
                continue

            try:
                module = import_module(module_name)
                classes = inspect.getmembers(module, inspect.isclass)

                # Filter classes
                estimators = [
                    (name, klass)
                    for name, klass in classes
                    if _is_estimator(name, klass)
                ]
                all_estimators.extend(estimators)
            except ModuleNotFoundError as e:
                # Skip missing soft dependencies
                if "soft dependency" not in str(e):
                    raise e
                warnings.warn(str(e), ImportWarning)

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

    if filter_tags is not None:
        all_estimators = [
            (n, est) for (n, est) in all_estimators if _check_tag_cond(est, filter_tags)
        ]

    # remove names if return_names=False
    if not return_names:
        all_estimators = [estimator for (name, estimator) in all_estimators]
        columns = ["estimator"]
    else:
        columns = ["name", "estimator"]

    # convert to pd.DataFrame if as_dataframe=True
    if as_dataframe:
        all_estimators = pd.DataFrame(all_estimators, columns=columns)

    return all_estimators


def _check_tag_cond(estimator, filter_tags=None, as_dataframe=True):
    """Check whether estimator satisfies filter_tags condition.

    Parameters
    ----------
    estimator: BaseEstimator, an sktime estimator
    filter_tags: dict of (str or list of str), default=None
        subsets the returned estimators as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"
    as_dataframe: bool, default=False
                if False, return is as described below;
                if True, return is converted into a DataFrame for pretty display

    Returns
    -------
    cond_sat: bool, whether estimator satisfies condition in filter_tags
    """
    if not isinstance(filter_tags, dict):
        raise TypeError("filter_tags must be a dict")

    cond_sat = True

    for (key, value) in filter_tags.items():
        if not isinstance(value, list):
            value = [value]
        cond_sat = cond_sat and estimator.get_class_tag(key) in set(value)

    return cond_sat


def all_tags(
    estimator_types=None,
    as_dataframe=False,
):
    """Get a list of all tags from sktime.

    Retrieves tags directly from `_tags`, offers filtering functionality.

    Parameters
    ----------
    estimator_types: string, list of string, optional (default=None)
        Which kind of estimators should be returned.
        - If None, no filter is applied and all estimators are returned.
        - Possible values are 'classifier', 'regressor', 'transformer' and
        'forecaster' to get estimators only of these specific types, or a list of
        these to get the estimators that fit at least one of the types.
    as_dataframe: bool, optional (default=False)
                if False, return is as described below;
                if True, return is converted into a DataFrame for pretty display

    Returns
    -------
    tags: list of tuples (a, b, c, d),
        in alphabetical order by a
        a : string - name of the tag as used in the _tags dictionary
        b : string - name of the scitype this tag applies to
                    must be in _base_classes.BASE_CLASS_SCITYPE_LIST
        c : string - expected type of the tag value
            should be one of:
                "bool" - valid values are True/False
                "int" - valid values are all integers
                "str" - valid values are all strings
                ("str", list_of_string) - any string in list_of_string is valid
                ("list", list_of_string) - any individual string and sub-list is valid
        d : string - plain English description of the tag
    """

    def is_tag_for_type(tag, estimator_types):
        tag_types = tag[1]
        if isinstance(tag_types, str):
            tag_types = [tag_types]
        elif not isinstance(tag_types, list):
            raise ValueError(
                "Error in ESTIMATOR_TAG_REGISTER, "
                "2nd entries of register tuples must be list or list of str"
            )
        if isinstance(estimator_types, str):
            estimator_types = [estimator_types]

        tag_types = set(tag_types)
        estimator_types = set(estimator_types)
        is_valid_tag_for_type = len(tag_types.intersection(estimator_types)) > 0

        return is_valid_tag_for_type

    all_tags = ESTIMATOR_TAG_REGISTER

    if estimator_types is not None:
        # checking, but not using the return since that is classes, not strings
        _check_estimator_types(estimator_types)
        all_tags = [tag for tag in all_tags if is_tag_for_type(tag, estimator_types)]

    all_tags = sorted(all_tags, key=itemgetter(0))

    # convert to pd.DataFrame if as_dataframe=True
    if as_dataframe:
        columns = ["name", "scitype", "type", "description"]
        all_tags = pd.DataFrame(all_tags, columns=columns)

    return all_tags


def _check_estimator_types(estimator_types):
    """Return list of classes corresponding to type strings."""
    estimator_types = deepcopy(estimator_types)

    if not isinstance(estimator_types, list):
        estimator_types = [estimator_types]  # make iterable

    def _get_err_msg(estimator_type):
        return (
            f"Parameter `estimator_type` must be None, a string or a list of "
            f"strings. Valid string values are: "
            f"{tuple(BASE_CLASS_LOOKUP.keys())}, but found: "
            f"{repr(estimator_type)}"
        )

    for i, estimator_type in enumerate(estimator_types):
        if not isinstance(estimator_type, (type, str)):
            raise ValueError(
                "Please specify `estimator_types` as a list of str or " "types."
            )
        if isinstance(estimator_type, str):
            if estimator_type not in BASE_CLASS_LOOKUP.keys():
                raise ValueError(_get_err_msg(estimator_type))
            estimator_type = BASE_CLASS_LOOKUP[estimator_type]
            estimator_types[i] = estimator_type
        elif isinstance(estimator_type, type):
            pass
        else:
            raise ValueError(_get_err_msg(estimator_type))
    return estimator_types

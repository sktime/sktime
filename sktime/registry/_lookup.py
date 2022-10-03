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

__author__ = ["fkiraly", "mloning", "katiebuc", "miraep8", "xloem"]
# all_estimators is also based on the sklearn utility of the same name


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
    return_tags=None,
    suppress_import_stdout=True,
):
    """Get a list of all estimators from sktime.

    This function crawls the module and gets all classes that inherit
    from sktime's and sklearn's base classes.

    Not included are: the base classes themselves, classes defined in test
    modules.

    Parameters
    ----------
    estimator_types: str, list of str, optional (default=None)
        Which kind of estimators should be returned.
        if None, no filter is applied and all estimators are returned.
        if str or list of str, strings define scitypes specified in search
                only estimators that are of (at least) one of the scitypes are returned
            possible str values are entries of registry.BASE_CLASS_REGISTER (first col)
                for instance 'classifier', 'regressor', 'transformer', 'forecaster'
    return_names: bool, optional (default=True)
        if True, estimator class name is included in the all_estimators()
            return in the order: name, estimator class, optional tags, either as
            a tuple or as pandas.DataFrame columns
        if False, estimator class name is removed from the all_estimators()
            return.
    filter_tags: dict of (str or list of str), optional (default=None)
        For a list of valid tag strings, use the registry.all_tags utility.
        subsets the returned estimators as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"
    exclude_estimators: str, list of str, optional (default=None)
        Names of estimators to exclude.
    as_dataframe: bool, optional (default=False)
        if True, all_estimators will return a pandas.DataFrame with named
            columns for all of the attributes being returned.
        if False, all_estimators will return a list (either a list of
            estimators or a list of tuples, see Returns)
    return_tags: str or list of str, optional (default=None)
        Names of tags to fetch and return each estimator's value of.
        For a list of valid tag strings, use the registry.all_tags utility.
        if str or list of str,
            the tag values named in return_tags will be fetched for each
            estimator and will be appended as either columns or tuple entries.
    suppress_import_stdout : bool, optional. Default=True
        whether to suppress stdout printout upon import.

    Returns
    -------
    all_estimators will return one of the following:
        1. list of estimators, if return_names=False, and return_tags is None
        2. list of tuples (optional estimator name, class, ~optional estimator
                tags), if return_names=True or return_tags is not None.
        3. pandas.DataFrame if as_dataframe = True
        if list of estimators:
            entries are estimators matching the query,
            in alphabetical order of estimator name
        if list of tuples:
            list of (optional estimator name, estimator, optional estimator
            tags) matching the query, in alphabetical order of estimator name,
            where
            ``name`` is the estimator name as string, and is an
                optional return
            ``estimator`` is the actual estimator
            ``tags`` are the estimator's values for each tag in return_tags
                and is an optional return.
        if dataframe:
            all_estimators will return a pandas.DataFrame.
            column names represent the attributes contained in each column.
            "estimators" will be the name of the column of estimators, "names"
            will be the name of the column of estimator class names and the string(s)
            passed in return_tags will serve as column names for all columns of
            tags that were optionally requested.

    Examples
    --------
    >>> from sktime.registry import all_estimators
    >>> # return a complete list of estimators as pd.Dataframe
    >>> all_estimators(as_dataframe=True)
    >>> # return all forecasters by filtering for estimator type
    >>> all_estimators("forecaster")
    >>> # return all forecasters which handle missing data in the input by tag filtering
    >>> all_estimators("forecaster", filter_tags={"handles-missing-data": True})

    References
    ----------
    Modified version from scikit-learn's `all_estimators()`.
    """
    import io
    import sys
    import warnings

    MODULES_TO_IGNORE = ("tests", "setup", "contrib", "benchmarking", "utils", "all")

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

    def _walk(root, exclude=None, prefix=""):
        """Return all modules contained as sub-modules (recursive) as string list.

        Unlike pkgutil.walk_packages, does not import modules on exclusion list.

        Parameters
        ----------
        root : Path
            root path in which to look for submodules
        exclude : tuple of str or None, optional, default = None
            list of sub-modules to ignore in the return, including sub-modules
        prefix: str, optional, default = ""
            this str is appended to all strings in the return

        Yields
        ------
        str : sub-module strings
            iterates over all sub-modules of root
            that do not contain any of the strings on the `exclude` list
            string is prefixed by the string `prefix`
        """

        def _is_ignored_module(module):
            if exclude is None:
                return False
            module_parts = module.split(".")
            return any(part in exclude for part in module_parts)

        for _, module_name, is_pgk in pkgutil.iter_modules(path=[root]):
            if not _is_ignored_module(module_name):
                yield f"{prefix}{module_name}"
                if is_pgk:
                    yield from (
                        f"{prefix}{module_name}.{x}"
                        for x in _walk(f"{root}/{module_name}", exclude=exclude)
                    )

    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("module", category=ImportWarning)
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*has been moved to.*"
        )
        for module_name in _walk(
            root=ROOT, exclude=MODULES_TO_IGNORE, prefix="sktime."
        ):

            # Filter modules
            if _is_private_module(module_name):
                continue

            try:
                if suppress_import_stdout:
                    # setup text trap, import, then restore
                    sys.stdout = io.StringIO()
                    module = import_module(module_name)
                    sys.stdout = sys.__stdout__
                else:
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

    if estimator_types:
        estimator_types = _check_estimator_types(estimator_types)
        all_estimators = [
            (name, estimator)
            for name, estimator in all_estimators
            if _is_in_estimator_types(estimator, estimator_types)
        ]

    # Filter based on given exclude list
    if exclude_estimators:
        exclude_estimators = _check_list_of_str_or_error(
            exclude_estimators, "exclude_estimators"
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

    if filter_tags:
        all_estimators = [
            (n, est) for (n, est) in all_estimators if _check_tag_cond(est, filter_tags)
        ]

    # remove names if return_names=False
    if not return_names:
        all_estimators = [estimator for (name, estimator) in all_estimators]
        columns = ["estimator"]
    else:
        columns = ["name", "estimator"]

    # add new tuple entries to all_estimators for each tag in return_tags:
    if return_tags:
        return_tags = _check_list_of_str_or_error(return_tags, "return_tags")
        # enrich all_estimators by adding the values for all return_tags tags:
        if all_estimators:
            if isinstance(all_estimators[0], tuple):
                all_estimators = [
                    (name, est) + _get_return_tags(est, return_tags)
                    for (name, est) in all_estimators
                ]
            else:
                all_estimators = [
                    tuple([est]) + _get_return_tags(est, return_tags)
                    for est in all_estimators
                ]
        columns = columns + return_tags

    # convert to pandas.DataFrame if as_dataframe=True
    if as_dataframe:
        all_estimators = pd.DataFrame(all_estimators, columns=columns)

    return all_estimators


def _check_list_of_str_or_error(arg_to_check, arg_name):
    """Check that certain arguments are str or list of str.

    Parameters
    ----------
    arg_to_check: argument we are testing the type of
    arg_name: str,
        name of the argument we are testing, will be added to the error if
        ``arg_to_check`` is not a str or a list of str

    Returns
    -------
    arg_to_check: list of str,
        if arg_to_check was originally a str it converts it into a list of str
        so that it can be iterated over.

    Raises
    ------
    TypeError if arg_to_check is not a str or list of str
    """
    # check that return_tags has the right type:
    if isinstance(arg_to_check, str):
        arg_to_check = [arg_to_check]
    if not isinstance(arg_to_check, list) or not all(
        isinstance(value, str) for value in arg_to_check
    ):
        raise TypeError(
            f"Error in all_estimators!  Argument {arg_name} must be either\
             a str or list of str"
        )
    return arg_to_check


def _get_return_tags(estimator, return_tags):
    """Fetch a list of all tags for every_entry of all_estimators.

    Parameters
    ----------
    estimator:  BaseEstimator, an sktime estimator
    return_tags: list of str,
        names of tags to get values for the estimator

    Returns
    -------
    tags: a tuple with all the estimators values for all tags in return tags.
        a value is None if it is not a valid tag for the estimator provided.
    """
    tags = tuple(estimator.get_class_tag(tag) for tag in return_tags)
    return tags


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
                if True, return is converted into a pandas.DataFrame for pretty
                display

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
                if True, return is converted into a pandas.DataFrame for pretty
                display

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
        tag_types = _check_list_of_str_or_error(tag_types, "tag_types")

        if isinstance(estimator_types, str):
            estimator_types = [estimator_types]

        tag_types = set(tag_types)
        estimator_types = set(estimator_types)
        is_valid_tag_for_type = len(tag_types.intersection(estimator_types)) > 0

        return is_valid_tag_for_type

    all_tags = ESTIMATOR_TAG_REGISTER

    if estimator_types:
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

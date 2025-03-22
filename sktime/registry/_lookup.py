# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Registry lookup methods.

This module exports the following methods for registry lookup:

all_estimators(estimator_types, filter_tags)
    lookup and filtering of estimators

all_tags(estimator_types)
    lookup and filtering of estimator tags
"""

__author__ = ["fkiraly", "mloning", "katiebuc", "miraep8", "xloem"]
# all_estimators is also based on the sklearn utility of the same name


from copy import deepcopy
from inspect import isclass
from operator import itemgetter
from pathlib import Path

import pandas as pd
from skbase.lookup import all_objects

from sktime.base import BaseObject
from sktime.registry._base_classes import (
    _get_all_descendants,
    get_base_class_for_str,
    get_obj_scitype_list,
)
from sktime.registry._tags import ESTIMATOR_TAG_REGISTER


def all_estimators(
    estimator_types=None,
    filter_tags=None,
    exclude_estimators=None,
    return_names=True,
    as_dataframe=False,
    return_tags=None,
    suppress_import_stdout=True,
):
    """List all estimators or objects in sktime, by scitype or tag.

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

        if True, estimator class name is included in the ``all_estimators``
        return in the order: name, estimator class, optional tags, either as
        a tuple or as pandas.DataFrame columns

        if False, estimator class name is removed from the ``all_estimators`` return.

    filter_tags: dict of (str or list of str or re.Pattern), optional (default=None)
        For a list of valid tag strings, use the registry.all_tags utility.

        ``filter_tags`` subsets the returned estimators as follows:

        * each key/value pair is statement in "and"/conjunction
        * key is tag name to sub-set on
        * value str or list of string are tag values
        * condition is "key must be equal to value, or in set(value)"

        In detail, he return will be filtered to keep exactly the classes
        where tags satisfy all the filter conditions specified by ``filter_tags``.
        Filter conditions are as follows, for ``tag_name: search_value`` pairs in
        the ``filter_tags`` dict, applied to a class ``klass``:

        - If ``klass`` does not have a tag with name ``tag_name``, it is excluded.
          Otherwise, let ``tag_value`` be the value of the tag with name ``tag_name``.
        - If ``search_value`` is a string, and ``tag_value`` is a string,
          the filter condition is that ``search_value`` must match the tag value.
        - If ``search_value`` is a string, and ``tag_value`` is a list,
          the filter condition is that ``search_value`` is contained in ``tag_value``.
        - If ``search_value`` is a ``re.Pattern``, and ``tag_value`` is a string,
          the filter condition is that ``search_value.fullmatch(tag_value)``
          is true, i.e., the regex matches the tag value.
        - If ``search_value`` is a ``re.Pattern``, and ``tag_value`` is a list,
          the filter condition is that at least one element of ``tag_value``
          matches the regex.
        - If ``search_value`` is iterable, then the filter condition is that
          at least one element of ``search_value`` satisfies the above conditions,
          applied to ``tag_value``.

        Note: ``re.Pattern`` is supported only from ``scikit-base`` version 0.8.0.

    exclude_estimators: str, list of str, optional (default=None)
        Names of estimators to exclude.

    as_dataframe: bool, optional (default=False)

        True: ``all_estimators`` will return a ``pandas.DataFrame`` with named
        columns for all of the attributes being returned.

        False: ``all_estimators`` will return a list (either a list of
        estimators or a list of tuples, see Returns)

    return_tags: str or list of str, optional (default=None)
        Names of tags to fetch and return each estimator's value of.
        For a list of valid tag strings, use the ``registry.all_tags`` utility.
        if str or list of str,
        the tag values named in return_tags will be fetched for each
        estimator and will be appended as either columns or tuple entries.

    suppress_import_stdout : bool, optional. Default=True
        whether to suppress stdout printout upon import.

    Returns
    -------
    all_estimators will return one of the following:

        1. list of estimators, if ``return_names=False``, and ``return_tags`` is None

        2. list of tuples (optional estimator name, class, ~ptional estimator
        tags), if ``return_names=True`` or ``return_tags`` is not ``None``.

        3. ``pandas.DataFrame`` if ``as_dataframe = True``

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
        if ``DataFrame``:
            column names represent the attributes contained in each column.
            "estimators" will be the name of the column of estimators, "names"
            will be the name of the column of estimator class names and the string(s)
            passed in return_tags will serve as column names for all columns of
            tags that were optionally requested.

    Examples
    --------
    >>> from sktime.registry import all_estimators
    >>> # return a complete list of estimators as pd.Dataframe
    >>> all_estimators(as_dataframe=True)  # doctest: +SKIP
    >>> # return all forecasters by filtering for estimator type
    >>> all_estimators("forecaster")  # doctest: +SKIP
    >>> # return all forecasters which handle missing data in the input by tag filtering
    >>> all_estimators("forecaster", filter_tags={"handles-missing-data": True})  # doctest: +SKIP

    References
    ----------
    Modified version of ``scikit-learn``'s ``all_estimators``.
    """  # noqa: E501
    MODULES_TO_IGNORE = (
        "tests",
        "setup",
        "contrib",
        "benchmarking",
        "utils",
        "all",
        "plotting",
        "_split",
        "test_split",
        "registry",
        "normal",
        "_normal",
    )

    ROOT = str(Path(__file__).parent.parent)  # sktime package root directory

    def _coerce_to_str(obj):
        if isinstance(obj, (list, tuple)):
            return [_coerce_to_str(o) for o in obj]
        if isclass(obj):
            obj = obj.get_tag("object_type")
        return obj

    def _coerce_to_list_of_str(obj):
        obj = _coerce_to_str(obj)
        if isinstance(obj, str):
            return [obj]
        return obj

    if estimator_types is not None:
        estimator_types = _coerce_to_list_of_str(estimator_types)
        estimator_types = [x for y in estimator_types for x in _get_all_descendants(y)]
        estimator_types = list(set(estimator_types))

    if estimator_types is not None:
        if filter_tags is None:
            filter_tags = {}
        elif isinstance(filter_tags, str):
            filter_tags = {filter_tags: True}
        else:
            filter_tags = filter_tags.copy()

        if "object_type" in filter_tags:
            obj_field = filter_tags["object_type"]
            obj_field = _coerce_to_list_of_str(obj_field)
            obj_field = obj_field + estimator_types
        else:
            obj_field = estimator_types

        filter_tags["object_type"] = obj_field

    result = all_objects(
        object_types=BaseObject,
        filter_tags=filter_tags,
        exclude_objects=exclude_estimators,
        return_names=return_names,
        as_dataframe=as_dataframe,
        return_tags=return_tags,
        suppress_import_stdout=suppress_import_stdout,
        package_name="sktime",
        path=ROOT,
        modules_to_ignore=MODULES_TO_IGNORE,
    )

    return result


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

    for key, value in filter_tags.items():
        if not isinstance(value, list):
            value = [value]
        cond_sat = cond_sat and estimator.get_class_tag(key) in set(value)

    return cond_sat


def all_tags(
    estimator_types=None,
    as_dataframe=False,
):
    """List all tags in sktime, for objects of a certain type.

    All objects in ``sktime`` are tagged with a set of :term``tag``s.
    This function allows to list all tags, optionally filtered by
    the object :term:`scitype` they apply to.

    Parameters
    ----------
    estimator_types: string, list of string, optional (default=None)
        The object type (:term:`scitype`) for which applicable tags should be listed.

        - If None, no filter is applied and tags for all estimators are returned.
        - Possible values are identifier strings for object scitypes, such as
          ``"forecaster"``, ``"classifier", ``"transformer"``, or lists thereof.
          If list, finds tags applicable to at least one of the listed types.
          Valid scitype strings are in ``sktime.registry.BASE_CLASS_SCITYPE_LIST``.


    as_dataframe: bool, optional (default=False)
        if False, return is as described below;
        if True, return is converted into a pandas.DataFrame for pretty display

    Returns
    -------
    tags: list of tuples (a, b, c, d),
        in alphabetical order by a

        a : string - name of the tag as used in the _tags dictionary

        b : string - name of the scitype this tag applies to,
        as in .BASE_CLASS_SCITYPE.LIST

        c : string - expected type of the tag value, one of:

            * ``"bool"`` - valid values are True/False
            * ``"int"`` - valid values are all integers
            * ``"str"`` - valid values are all strings
            * ``("str", "list_of_string")`` - any string in ``list_of_string`` is valid
            * ``("list", "list_of_string")`` - any string element or sub-list is valid

        d : string - plain English description of the tag
    """

    def is_tag_for_type(tag, estimator_types):
        tag_types = tag[1]
        tag_types = _check_list_of_str_or_error(tag_types, "tag_types")

        if isinstance(estimator_types, str):
            estimator_types = [estimator_types]

        # also retrieve all tags for topmost base classes
        # "estimator" has also been used for object tags, so is always included
        estimator_types += ["estimator", "object"]

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
    """Return list of classes corresponding to type strings.

    Parameters
    ----------
    estimator_types: str, or list of str

    Returns
    -------
    estimator_types: list of classes
        base classes corresponding to scitype strings in estimator_types
    """
    estimator_types = deepcopy(estimator_types)

    if not isinstance(estimator_types, list):
        estimator_types = [estimator_types]  # make iterable

    def _get_err_msg(estimator_type):
        return (
            f"Parameter `estimator_type` must be a string or a list of "
            f"strings. Valid string values are: "
            f"{get_obj_scitype_list()}, but found: "
            f"{repr(estimator_type)}"
        )

    for i, estimator_type in enumerate(estimator_types):
        if not isinstance(estimator_type, (type, str)):
            raise ValueError(
                "Please specify `estimator_types` as a list of str or types."
            )
        if isinstance(estimator_type, str):
            if estimator_type not in get_obj_scitype_list():
                raise ValueError(_get_err_msg(estimator_type))
            estimator_types[i] = get_base_class_for_str(estimator_type)
        elif isinstance(estimator_type, type):
            pass
        else:
            raise ValueError(_get_err_msg(estimator_type))
    return estimator_types

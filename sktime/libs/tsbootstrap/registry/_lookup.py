"""Registry lookup methods.

This module exports the following methods for registry lookup:

all_objects(object_types, filter_tags)
    lookup and filtering of objects
"""

# based on the sktime module of same name

__author__ = ["fkiraly"]
# all_objects is based on the sklearn utility all_estimators


from pathlib import Path

from skbase.base import BaseObject
from skbase.lookup import all_objects as _all_objects

from tsbootstrap.registry._tags import OBJECT_TAG_REGISTER

VALID_OBJECT_TYPE_STRINGS = {x[1] for x in OBJECT_TAG_REGISTER}


def all_objects(
    object_types=None,
    filter_tags=None,
    exclude_objects=None,
    return_names=True,
    as_dataframe=False,
    return_tags=None,
    suppress_import_stdout=True,
):
    """Get a list of all objects from tsbootstrap.

    This function crawls the module and gets all classes that inherit
    from tsbootstrap's and sklearn's base classes.

    Not included are: the base classes themselves, classes defined in test modules.

    Parameters
    ----------
    object_types: str, list of str, optional (default=None)
        Which kind of objects should be returned.
        if None, no filter is applied and all objects are returned.
        if str or list of str, strings define scitypes specified in search
                only objects that are of (at least) one of the scitypes are returned
            possible str values are entries of registry.BASE_CLASS_REGISTER (first col)
    return_names: bool, optional (default=True)
        if True, object class name is included in the all_objects()
            return in the order: name, object class, optional tags, either as
            a tuple or as pandas.DataFrame columns
        if False, object class name is removed from the all_objects()
            return.
    filter_tags: dict of (str or list of str), optional (default=None)
        For a list of valid tag strings, use the registry.all_tags utility.
        subsets the returned objects as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"
    exclude_objects: str, list of str, optional (default=None)
        Names of objects to exclude.
    as_dataframe: bool, optional (default=False)
        if True, all_objects will return a pandas.DataFrame with named
            columns for all of the attributes being returned.
        if False, all_objects will return a list (either a list of
            objects or a list of tuples, see Returns)
    return_tags: str or list of str, optional (default=None)
        Names of tags to fetch and return each object's value of.
        For a list of valid tag strings, use the registry.all_tags utility.
        if str or list of str,
            the tag values named in return_tags will be fetched for each
            object and will be appended as either columns or tuple entries.
    suppress_import_stdout : bool, optional. Default=True
        whether to suppress stdout printout upon import.

    Returns
    -------
    all_objects will return one of the following:
        1. list of objects, if return_names=False, and return_tags is None
        2. list of tuples (optional object name, class, ~optional object
                tags), if return_names=True or return_tags is not None.
        3. pandas.DataFrame if as_dataframe = True
        if list of objects:
            entries are objects matching the query,
            in alphabetical order of object name
        if list of tuples:
            list of (optional object name, object, optional object
            tags) matching the query, in alphabetical order of object name,
            where
            ``name`` is the object name as string, and is an
                optional return
            ``object`` is the actual object
            ``tags`` are the object's values for each tag in return_tags
                and is an optional return.
        if dataframe:
            all_objects will return a pandas.DataFrame.
            column names represent the attributes contained in each column.
            "objects" will be the name of the column of objects, "names"
            will be the name of the column of object class names and the string(s)
            passed in return_tags will serve as column names for all columns of
            tags that were optionally requested.

    Examples
    --------
    >>> from tsbootstrap.registry import all_objects
    >>> # return a complete list of objects as pd.Dataframe
    >>> all_objects(as_dataframe=True)
    >>> # return all bootstrap algorithms by filtering for object type
    >>> all_objects("bootstrap", as_dataframe=True)
    >>> # return all bootstraps which are block bootstraps
    >>> all_objects(
    ...     "bootstrap",
    ...     filter_tags={"bootstrap_type": "block"},
    ...     as_dataframe=True
    ... )

    References
    ----------
    Adapted version of sktime's ``all_estimators``,
    which is an evolution of scikit-learn's ``all_estimators``
    """
    MODULES_TO_IGNORE = (
        "tests",
        "setup",
        "contrib",
        "utils",
        "all",
    )

    result = []
    ROOT = str(
        Path(__file__).parent.parent
    )  # tsbootstrap package root directory

    if isinstance(filter_tags, str):
        filter_tags = {filter_tags: True}
    filter_tags = filter_tags.copy() if filter_tags else None

    if object_types:
        if filter_tags and "object_type" not in filter_tags:
            object_tag_filter = {"object_type": object_types}
        elif filter_tags:
            filter_tags_filter = filter_tags.get("object_type", [])
            if isinstance(object_types, str):
                object_types = [object_types]
            object_tag_update = {
                "object_type": object_types + filter_tags_filter
            }
            filter_tags.update(object_tag_update)
        else:
            object_tag_filter = {"object_type": object_types}
        if filter_tags:
            filter_tags.update(object_tag_filter)
        else:
            filter_tags = object_tag_filter

    result = _all_objects(
        object_types=[BaseObject],
        filter_tags=filter_tags,
        exclude_objects=exclude_objects,
        return_names=return_names,
        as_dataframe=as_dataframe,
        return_tags=return_tags,
        suppress_import_stdout=suppress_import_stdout,
        package_name="tsbootstrap",
        path=ROOT,
        modules_to_ignore=MODULES_TO_IGNORE,
    )

    return result

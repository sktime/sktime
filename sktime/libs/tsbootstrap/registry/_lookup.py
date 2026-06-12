"""
Registry lookup methods.

This module exports the following methods for registry lookup:

- all_objects(object_types: Optional[Union[str, List[str]]] = None,
             filter_tags: Optional[Dict[str, Union[str, List[str], bool]]] = None,
             exclude_objects: Optional[Union[str, List[str]]] = None,
             return_names: bool = True,
             as_dataframe: bool = False,
             return_tags: Optional[Union[str, List[str]]] = None,
             suppress_import_stdout: bool = True) -> Union[List[Any], List[Tuple]]
    Lookup and filtering of objects in the tsbootstrap registry.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from skbase.base import BaseObject
from skbase.lookup import all_objects as _all_objects

from tsbootstrap.registry._tags import (
    OBJECT_TAG_REGISTER,
    check_tag_is_valid,
)

VALID_OBJECT_TYPE_STRINGS: set = {tag.scitype for tag in OBJECT_TAG_REGISTER}


def all_objects(
    object_types: Optional[Union[str, List[str]]] = None,
    filter_tags: Optional[
        Union[str, Dict[str, Union[str, List[str], bool]]]
    ] = None,
    exclude_objects: Optional[Union[str, List[str]]] = None,
    return_names: bool = True,
    as_dataframe: bool = False,
    return_tags: Optional[Union[str, List[str]]] = None,
    suppress_import_stdout: bool = True,
) -> Union[List[Any], List[Tuple]]:
    """
    Get a list of all objects from tsbootstrap.

    This function crawls the module and retrieves all classes that inherit
    from tsbootstrap's and sklearn's base classes.

    Excluded from retrieval are:
        - The base classes themselves
        - Classes defined in test modules

    Parameters
    ----------
    object_types : Union[str, List[str]], optional (default=None)
        Specifies which types of objects to return.
        - If None, no filtering is applied and all objects are returned.
        - If str or list of str, only objects matching the specified scitypes are returned.
          Valid scitypes are entries in `registry.BASE_CLASS_REGISTER` (first column).

    filter_tags : Union[str, Dict[str, Union[str, List[str], bool]]], optional (default=None)
        Dictionary or string to filter returned objects based on their tags.
        - If a string, it is treated as a boolean tag filter with the value `True`.
        - If a dictionary, each key-value pair represents a filter condition in an "AND" conjunction.
          - Key is the tag name to filter on.
          - Value is a string, list of strings, or boolean that the tag value must match or be within.
        - Only objects satisfying all filter conditions are returned.

    exclude_objects : Union[str, List[str]], optional (default=None)
        Names of objects to exclude from the results.

    return_names : bool, optional (default=True)
        - If True, the object's class name is included in the returned results.
        - If False, the class name is omitted.

    as_dataframe : bool, optional (default=False)
        - If True, returns a pandas.DataFrame with named columns for all returned attributes.
        - If False, returns a list (of objects or tuples).

    return_tags : Union[str, List[str]], optional (default=None)
        - Names of tags to fetch and include in the returned results.
        - If specified, tag values are appended as either columns or tuple entries.

    suppress_import_stdout : bool, optional (default=True)
        Whether to suppress stdout printout upon import.

    Returns
    -------
    Union[List[Any], List[Tuple]]
        Depending on the parameters:
        1. List of objects:
            - Entries are objects matching the query, in alphabetical order of object name.
        2. List of tuples:
            - Each tuple contains (optional object name, object class, optional object tags).
            - Ordered alphabetically by object name.
        3. pandas.DataFrame:
            - Columns represent the returned attributes.
            - Includes "objects", "names", and any specified tag columns.

    Examples
    --------
    >>> from tsbootstrap.registry import all_objects
    >>> # Return a complete list of objects as a DataFrame
    >>> all_objects(as_dataframe=True)
    >>> # Return all bootstrap algorithms by filtering for object type
    >>> all_objects("bootstrap", as_dataframe=True)
    >>> # Return all bootstraps which are block bootstraps
    >>> all_objects(
    ...     object_types="bootstrap",
    ...     filter_tags={"bootstrap_type": "block"},
    ...     as_dataframe=True
    ... )

    References
    ----------
    Adapted version of sktime's `all_estimators`,
    which is an evolution of scikit-learn's `all_estimators`.
    """
    MODULES_TO_IGNORE = (
        "tests",
        "setup",
        "contrib",
        "utils",
        "all",
    )

    result: Union[List[Any], List[Tuple]] = []
    ROOT = str(
        Path(__file__).parent.parent
    )  # tsbootstrap package root directory

    # Prepare filter_tags
    if isinstance(filter_tags, str):
        # Ensure the tag expects a boolean value
        tag = next(
            (t for t in OBJECT_TAG_REGISTER if t.name == filter_tags), None
        )
        if not tag:
            raise ValueError(
                f"Tag '{filter_tags}' not found in OBJECT_TAG_REGISTER."
            )
        if tag.value_type != "bool":
            raise ValueError(
                f"Tag '{filter_tags}' does not expect a boolean value."
            )
        filter_tags = {filter_tags: True}
    elif isinstance(filter_tags, dict):
        # Validate each tag in filter_tags
        for key, value in filter_tags.items():
            try:
                if not check_tag_is_valid(key, value):
                    raise ValueError(
                        f"Invalid value '{value}' for tag '{key}'."
                    )
            except KeyError as e:
                raise ValueError(
                    f"Tag '{key}' not found in OBJECT_TAG_REGISTER."
                ) from e
    else:
        filter_tags = None

    if object_types:
        if isinstance(object_types, str):
            object_types = [object_types]
        # Validate object_types
        invalid_types = set(object_types) - VALID_OBJECT_TYPE_STRINGS
        if invalid_types:
            raise ValueError(
                f"Invalid object_types: {invalid_types}. Valid types are {VALID_OBJECT_TYPE_STRINGS}."
            )
        if filter_tags and "object_type" not in filter_tags:
            object_tag_filter = {"object_type": object_types}
            filter_tags.update(object_tag_filter)
        elif filter_tags and "object_type" in filter_tags:
            existing_filter = filter_tags.get("object_type", [])
            if isinstance(existing_filter, str):
                existing_filter = [existing_filter]
            elif isinstance(existing_filter, list):
                pass
            else:
                raise ValueError(
                    f"Unexpected type for 'object_type' filter: {type(existing_filter)}"
                )
            combined_filter = list(set(object_types + existing_filter))
            filter_tags["object_type"] = combined_filter
        else:
            filter_tags = {"object_type": object_types}

    # Retrieve objects using skbase's all_objects
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

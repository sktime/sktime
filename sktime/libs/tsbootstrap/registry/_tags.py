"""
Register of estimator and object tags.

Note for extenders:
    New tags should be entered in `OBJECT_TAG_REGISTER`.
    No other place is necessary to add new tags.

This module exports the following:

- OBJECT_TAG_REGISTER : List[Tag]
    A list of Tag instances, each representing a tag with its attributes.

- OBJECT_TAG_TABLE : List[Dict[str, Any]]
    `OBJECT_TAG_REGISTER` in table form as a list of dictionaries.

- OBJECT_TAG_LIST : List[str]
    List of tag names extracted from `OBJECT_TAG_REGISTER`.

- check_tag_is_valid(tag_name: str, tag_value: Any) -> bool
    Function to validate if a tag value is valid for a given tag name.
"""

from typing import Any, Dict, List, Tuple, Union

from pydantic import BaseModel, field_validator


class Tag(BaseModel):
    """
    Represents a single tag with its properties.

    Attributes
    ----------
    name : str
        Name of the tag as used in the _tags dictionary.
    scitype : str
        Name of the scitype this tag applies to.
    value_type : Union[str, Tuple[str, Union[List[str], str]], List[Union[str, Tuple[str, Union[List[str], str]]]]]
        Expected type(s) of the tag value.
    description : str
        Plain English description of the tag.
    """

    name: str
    scitype: str
    value_type: Union[
        str,
        Tuple[str, Union[List[str], str]],
        List[Union[str, Tuple[str, Union[List[str], str]]]],
    ]
    description: str

    @field_validator("value_type")
    @classmethod
    def validate_value_type(cls, v):
        """
        Validates the `value_type` attribute to ensure it adheres to expected formats.

        Parameters
        ----------
        v : Union[str, Tuple[str, Union[List[str], str]], List[Union[str, Tuple[str, Union[List[str], str]]]]]
            The value to validate.

        Returns
        -------
        Union[str, Tuple[str, Union[List[str], str]], List[Union[str, Tuple[str, Union[List[str], str]]]]]
            The validated value.

        Raises
        ------
        ValueError
            If `v` does not conform to expected types and constraints.
        TypeError
            If `v` is neither a string, a tuple, nor a list.
        """
        valid_base_types = {"bool", "int", "str", "list", "dict"}

        def validate_single_type(single_v):
            if isinstance(single_v, str):
                if single_v not in valid_base_types:
                    raise ValueError(
                        f"Invalid value_type: {single_v}. Must be one of {valid_base_types}."
                    )
            elif isinstance(single_v, tuple):
                if len(single_v) != 2:
                    raise ValueError(
                        "Tuple value_type must have exactly two elements."
                    )
                base, subtype = single_v
                if base not in {"str", "list"}:
                    raise ValueError(
                        "First element of tuple must be 'str' or 'list'."
                    )
                if base == "str":
                    if not isinstance(subtype, list) or not all(
                        isinstance(item, str) for item in subtype
                    ):
                        raise ValueError(
                            "Second element must be a list of strings when base is 'str'."
                        )
                elif base == "list" and not (
                    (
                        isinstance(subtype, list)
                        and all(isinstance(item, str) for item in subtype)
                    )
                    or isinstance(subtype, str)
                ):
                    raise ValueError(
                        "Second element must be a list of strings or 'str' when base is 'list'."
                    )
            else:
                raise TypeError(
                    "Each value_type must be either a string or a tuple."
                )

        if isinstance(v, list):
            if not v:
                raise ValueError("value_type list cannot be empty.")
            for item in v:
                validate_single_type(item)
        else:
            validate_single_type(v)

        return v


# Define the OBJECT_TAG_REGISTER with Tag instances
OBJECT_TAG_REGISTER: List[Tag] = [
    # --------------------------
    # All objects and estimators
    # --------------------------
    Tag(
        name="object_type",
        scitype="object",
        value_type=("str", ["regressor", "transformer"]),
        description="Type of object, e.g., 'regressor', 'transformer'.",
    ),
    Tag(
        name="python_version",
        scitype="object",
        value_type="str",
        description="Python version specifier (PEP 440) for estimator, or None for all versions.",
    ),
    Tag(
        name="python_dependencies",
        scitype="object",
        # Allow both string and list of strings
        value_type=["str", ("list", "str")],
        description="Python dependencies of estimator as string or list of strings.",
    ),
    Tag(
        name="python_dependencies_alias",
        scitype="object",
        value_type="dict",
        description="Alias for Python dependencies if import name differs from package name. Key-value pairs are package name and import name.",
    ),
    # -----------------------
    # BaseTimeSeriesBootstrap
    # -----------------------
    Tag(
        name="bootstrap_type",
        scitype="bootstrap",
        value_type=("list", "str"),
        description="Type(s) of bootstrap the algorithm supports.",
    ),
    Tag(
        name="capability:multivariate",
        scitype="bootstrap",
        value_type="bool",
        description="Whether the bootstrap algorithm supports multivariate data.",
    ),
    # ----------------------------
    # BaseMetaObject reserved tags
    # ----------------------------
    Tag(
        name="named_object_parameters",
        scitype="object",
        value_type="str",
        description="Name of component list attribute for meta-objects.",
    ),
    Tag(
        name="fitted_named_object_parameters",
        scitype="estimator",
        value_type="str",
        description="Name of fitted component list attribute for meta-objects.",
    ),
]

# Create OBJECT_TAG_TABLE as a list of dictionaries
OBJECT_TAG_TABLE: List[Dict[str, Any]] = [
    {
        "name": tag.name,
        "scitype": tag.scitype,
        "value_type": tag.value_type,
        "description": tag.description,
    }
    for tag in OBJECT_TAG_REGISTER
]

# Create OBJECT_TAG_LIST as a list of tag names
OBJECT_TAG_LIST: List[str] = [tag.name for tag in OBJECT_TAG_REGISTER]


def check_tag_is_valid(tag_name: str, tag_value: Any) -> bool:
    """
    Check whether a tag value is valid for a given tag name.

    Parameters
    ----------
    tag_name : str
        The name of the tag to validate.
    tag_value : Any
        The value to validate against the tag's expected type.

    Returns
    -------
    bool
        True if the tag value is valid for the tag name, False otherwise.

    Raises
    ------
    KeyError
        If the tag_name is not found in OBJECT_TAG_REGISTER.
    """
    try:
        tag = next(tag for tag in OBJECT_TAG_REGISTER if tag.name == tag_name)
    except StopIteration as e:
        raise KeyError(
            f"Tag name '{tag_name}' not found in OBJECT_TAG_REGISTER."
        ) from e

    value_type = tag.value_type

    if isinstance(value_type, list):
        # Iterate through each type definition and return True if any matches
        for vt in value_type:
            if isinstance(vt, str):
                if isinstance(tag_value, str):
                    return True
            elif isinstance(vt, tuple):
                base_type, subtype = vt
                if base_type == "str":
                    if isinstance(tag_value, str) and tag_value in subtype:
                        return True
                elif base_type == "list" and isinstance(tag_value, list):
                    if subtype == "str":
                        if all(isinstance(item, str) for item in tag_value):
                            return True
                    elif isinstance(subtype, list) and all(
                        isinstance(item, str) and item in subtype
                        for item in tag_value
                    ):
                        return True
        return False
    elif isinstance(value_type, str):
        expected_type = value_type
        if expected_type == "bool":
            return isinstance(tag_value, bool)
        elif expected_type == "int":
            return isinstance(tag_value, int)
        elif expected_type == "str":
            return isinstance(tag_value, str)
        elif expected_type == "list":
            return isinstance(tag_value, list)
        elif expected_type == "dict":
            return isinstance(tag_value, dict)
        else:
            return False
    elif isinstance(value_type, tuple):
        base_type, subtype = value_type
        if base_type == "str":
            if isinstance(tag_value, str):
                return tag_value in subtype
            return False
        elif base_type == "list":
            if not isinstance(tag_value, list):
                return False
            if isinstance(subtype, list):
                return all(
                    isinstance(item, str) and item in subtype
                    for item in tag_value
                )
            elif subtype == "str":
                return all(isinstance(item, str) for item in tag_value)
        return False
    else:
        return False

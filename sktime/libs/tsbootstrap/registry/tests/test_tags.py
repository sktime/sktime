"""
Tests for the tag registry and tag validation functionality.

This module contains tests to ensure that the `OBJECT_TAG_REGISTER` is correctly
configured and that each tag adheres to the specified structure and type constraints.
"""

from tsbootstrap.registry._tags import (
    OBJECT_TAG_LIST,
    OBJECT_TAG_REGISTER,
    OBJECT_TAG_TABLE,
    Tag,
    check_tag_is_valid,
)


def test_tag_register_type():
    """
    Test the specification of the tag register.

    Ensures that `OBJECT_TAG_REGISTER` is a list of `Tag` instances with the correct attributes and types.

    Raises
    ------
    TypeError
        If `OBJECT_TAG_REGISTER` is not a list or contains non-`Tag` instances.
    ValueError
        If any `Tag` instance does not conform to the expected structure or type constraints.
    """
    # Verify that OBJECT_TAG_REGISTER is a list
    if not isinstance(OBJECT_TAG_REGISTER, list):
        raise TypeError("`OBJECT_TAG_REGISTER` is not a list.")

    # Verify that all elements in OBJECT_TAG_REGISTER are instances of Tag
    if not all(isinstance(tag, Tag) for tag in OBJECT_TAG_REGISTER):
        raise TypeError(
            "Not all elements in `OBJECT_TAG_REGISTER` are `Tag` instances."
        )

    # Iterate through each Tag instance to validate its attributes
    for tag in OBJECT_TAG_REGISTER:
        # Validate the 'name' attribute
        if not isinstance(tag.name, str):
            raise TypeError(f"Tag name '{tag.name}' is not a string.")

        # Validate the 'scitype' attribute
        if not isinstance(tag.scitype, str):
            raise TypeError(f"Tag scitype '{tag.scitype}' is not a string.")

        # Validate the 'value_type' attribute
        if not isinstance(tag.value_type, (str, tuple, list)):
            raise TypeError(
                f"Tag value_type '{tag.value_type}' is not a string, tuple, or list."
            )

        if isinstance(tag.value_type, tuple):
            if len(tag.value_type) != 2:
                raise ValueError(
                    "Tuple `value_type` must have exactly two elements."
                )

            base_type, subtype = tag.value_type

            # Validate the base type
            if base_type not in {"str", "list"}:
                raise ValueError(
                    f"First element of `value_type` tuple must be 'str' or 'list', got '{base_type}'."
                )

            # Validate the subtype based on the base type
            if base_type == "str":
                if not isinstance(subtype, list) or not all(
                    isinstance(item, str) for item in subtype
                ):
                    raise TypeError(
                        "Second element of `value_type` tuple must be a list of strings when base is 'str'."
                    )
            elif base_type == "list" and not (
                (
                    isinstance(subtype, list)
                    and all(isinstance(item, str) for item in subtype)
                )
                or isinstance(subtype, str)
            ):
                raise TypeError(
                    "Second element of `value_type` tuple must be a list of strings or 'str' when base is 'list'."
                )

        elif isinstance(tag.value_type, list):
            if not tag.value_type:
                raise ValueError("`value_type` list cannot be empty.")

            for vt in tag.value_type:
                if isinstance(vt, str):
                    if vt not in {"bool", "int", "str", "list", "dict"}:
                        raise ValueError(
                            f"Invalid value_type in list: {vt}. Must be one of {{'bool', 'int', 'str', 'list', 'dict'}}."
                        )
                elif isinstance(vt, tuple):
                    if len(vt) != 2:
                        raise ValueError(
                            "Tuple in `value_type` list must have exactly two elements."
                        )
                    base, subtype = vt
                    if base not in {"str", "list"}:
                        raise ValueError(
                            "First element of tuple in `value_type` list must be 'str' or 'list'."
                        )
                    if base == "str":
                        if not isinstance(subtype, list) or not all(
                            isinstance(item, str) for item in subtype
                        ):
                            raise TypeError(
                                "Second element of tuple in `value_type` list must be a list of strings when base is 'str'."
                            )
                    elif base == "list" and not (
                        (
                            isinstance(subtype, list)
                            and all(isinstance(item, str) for item in subtype)
                        )
                        or isinstance(subtype, str)
                    ):
                        raise TypeError(
                            "Second element of tuple in `value_type` list must be a list of strings or 'str' when base is 'list'."
                        )
                else:
                    raise TypeError(
                        "`value_type` list elements must be either strings or tuples."
                    )

        # Validate the 'description' attribute
        if not isinstance(tag.description, str):
            raise TypeError(
                f"Tag description '{tag.description}' is not a string."
            )


def test_object_tag_table_structure():
    """
    Test the structure of `OBJECT_TAG_TABLE`.

    Ensures that `OBJECT_TAG_TABLE` is a list of dictionaries, each containing the expected keys and corresponding types.

    Raises
    ------
    TypeError
        If `OBJECT_TAG_TABLE` is not a list or contains elements that are not dictionaries.
    KeyError
        If any dictionary in `OBJECT_TAG_TABLE` is missing required keys.
    TypeError
        If any value in the dictionaries does not match the expected type.
    """
    # Define the expected keys and their types
    expected_keys = {
        "name": str,
        "scitype": str,
        "value_type": (str, tuple, list),
        "description": str,
    }

    # Verify that OBJECT_TAG_TABLE is a list
    if not isinstance(OBJECT_TAG_TABLE, list):
        raise TypeError("`OBJECT_TAG_TABLE` is not a list.")

    # Iterate through each dictionary in OBJECT_TAG_TABLE to validate its structure
    for entry in OBJECT_TAG_TABLE:
        # Verify that each entry is a dictionary
        if not isinstance(entry, dict):
            raise TypeError(
                "Each entry in `OBJECT_TAG_TABLE` must be a dictionary."
            )

        # Check for the presence of all expected keys
        for key, expected_type in expected_keys.items():
            if key not in entry:
                raise KeyError(
                    f"Key '{key}' is missing from an entry in `OBJECT_TAG_TABLE`."
                )

            # Validate the type of each value
            if not isinstance(entry[key], expected_type):
                raise TypeError(
                    f"Value for key '{key}' in `OBJECT_TAG_TABLE` entry is not of type {expected_type}."
                )


def test_object_tag_list():
    """
    Test the contents of `OBJECT_TAG_LIST`.

    Ensures that `OBJECT_TAG_LIST` contains all tag names present in `OBJECT_TAG_REGISTER` and that each name is a string.

    Raises
    ------
    TypeError
        If `OBJECT_TAG_LIST` is not a list or contains non-string elements.
    ValueError
        If any tag name in `OBJECT_TAG_REGISTER` is missing from `OBJECT_TAG_LIST`.
    """
    # Verify that OBJECT_TAG_LIST is a list
    if not isinstance(OBJECT_TAG_LIST, list):
        raise TypeError("`OBJECT_TAG_LIST` is not a list.")

    # Verify that all elements in OBJECT_TAG_LIST are strings
    if not all(isinstance(name, str) for name in OBJECT_TAG_LIST):
        raise TypeError("All elements in `OBJECT_TAG_LIST` must be strings.")

    # Extract all tag names from OBJECT_TAG_REGISTER
    tag_names = {tag.name for tag in OBJECT_TAG_REGISTER}

    # Verify that OBJECT_TAG_LIST contains all tag names
    missing_tags = tag_names - set(OBJECT_TAG_LIST)
    if missing_tags:
        raise ValueError(
            f"The following tags are missing from `OBJECT_TAG_LIST`: {missing_tags}"
        )


def test_check_tag_is_valid():
    """
    Test the `check_tag_is_valid` function.

    Ensures that `check_tag_is_valid` correctly validates tag values based on their expected types.

    Raises
    ------
    AssertionError
        If any test case fails.
    """
    # Define test cases as tuples of (tag_name, tag_value, expected_result)
    test_cases = [
        ("object_type", "regressor", True),
        ("object_type", "transformer", True),
        (
            "object_type",
            "classifier",
            False,
        ),  # Should be False as it's not in the allowed list
        ("object_type", "invalid_type", False),
        ("capability:multivariate", True, True),
        ("capability:multivariate", False, True),
        ("capability:multivariate", "yes", False),
        ("python_version", "3.8.5", True),
        ("python_version", 3.8, False),
        ("python_dependencies", ["numpy", "pandas"], True),
        ("python_dependencies", "numpy", True),
        ("python_dependencies", ["numpy", 123], False),
        ("python_dependencies_alias", {"numpy": "np"}, True),
        ("python_dependencies_alias", "numpy", False),
        ("non_existent_tag", "value", False),  # Should raise KeyError
    ]

    for tag_name, tag_value, expected in test_cases:
        if tag_name == "non_existent_tag":
            try:
                check_tag_is_valid(tag_name, tag_value)
                raise AssertionError(
                    f"Expected KeyError for tag '{tag_name}', but no error was raised."
                )
            except KeyError:
                pass  # Expected behavior
        else:
            result = check_tag_is_valid(tag_name, tag_value)
            if result != expected:
                raise AssertionError(
                    f"check_tag_is_valid({tag_name!r}, {tag_value!r}) returned {result}, expected {expected}."
                )

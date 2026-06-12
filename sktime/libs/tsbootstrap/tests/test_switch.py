# copyright:
# tsbootstrap developers, BSD-3-Clause License (see LICENSE file)
# based on utility from sktime of the same name

"""Switch utility for determining whether tests for a class should be run or not."""

__author__ = ["fkiraly", "astrogilda"]

from typing import Any, List, Optional, Union

from tsbootstrap.utils.dependencies import _check_estimator_dependencies


def run_test_for_class(cls: Union[Any, List[Any], tuple]) -> bool:
    """
    Determine whether tests should be run for a given class or function based on dependency checks.

    This function evaluates whether the provided class/function or a list of them has all required
    soft dependencies present in the current environment. If all dependencies are satisfied, it returns
    `True`, indicating that tests should be executed. Otherwise, it returns `False`.

    Parameters
    ----------
    cls : Union[Any, List[Any], tuple]
        A single class/function or a list/tuple of classes/functions for which to determine
        whether tests should be run. Each class/function should be a descendant of `BaseObject`
        and have the `get_class_tag` method for dependency retrieval.

    Returns
    -------
    bool
        `True` if all provided classes/functions have their required dependencies present.
        `False` otherwise.

    Raises
    ------
    ValueError
        If the severity level provided in dependency checks is invalid.
    TypeError
        If any object in `cls` does not have the `get_class_tag` method or is not a `BaseObject` descendant.
    """
    # Ensure cls is a list for uniform processing
    if not isinstance(cls, (list, tuple)):
        cls = [cls]

    # Define the severity level and message for dependency checks
    # Set to 'none' to silently return False without raising exceptions or warnings
    severity = "none"
    msg: Optional[str] = None  # No custom message

    # Perform dependency checks for all classes/functions
    # If any dependency is not met, the function will return False
    # Since severity is 'none', no exceptions or warnings will be raised
    try:
        all_dependencies_present = _check_estimator_dependencies(
            obj=cls, severity=severity, msg=msg
        )
    except (ValueError, TypeError):
        # Log the error if necessary, or handle it as per testing framework
        # For now, we assume that any exception means dependencies are not met
        all_dependencies_present = False

    return all_dependencies_present

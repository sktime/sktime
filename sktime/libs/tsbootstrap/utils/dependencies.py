"""Utility module for checking soft dependency imports and raising warnings or errors."""

__author__ = ["fkiraly", "astrogilda"]

import logging
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator
from skbase.utils.dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)

# Configure logging for the module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SeverityEnum(str, Enum):
    """
    Enumeration for severity levels.

    Attributes
    ----------
    ERROR : str
        Indicates that a `ModuleNotFoundError` should be raised if dependencies are not met.
    WARNING : str
        Indicates that a warning should be emitted if dependencies are not met.
    NONE : str
        Indicates that no action should be taken if dependencies are not met.
    """

    ERROR = "error"
    WARNING = "warning"
    NONE = "none"


def _check_estimator_dependencies(
    obj: Union[Any, List[Any], tuple],
    severity: Union[str, SeverityEnum] = "error",
    msg: Optional[str] = None,
) -> bool:
    """
    Check if an object or list of objects' package and Python requirements are met by the current environment.

    This function serves as a convenience wrapper around `_check_python_version` and `_check_soft_dependencies`,
    utilizing the estimator tags `"python_version"` and `"python_dependencies"`.

    Parameters
    ----------
    obj : Union[Any, List[Any], tuple]
        An object (instance or class) that is a descendant of `BaseObject`, or a list/tuple of such objects.
        These objects are checked for compatibility with the current Python environment.
    severity : Union[str, SeverityEnum], default="error"
        Determines the behavior when incompatibility is detected:
        - "error": Raises a `ModuleNotFoundError`.
        - "warning": Emits a warning and returns `False`.
        - "none": Silently returns `False` without raising an exception or warning.
    msg : Optional[str], default=None
        Custom error message to be used in the `ModuleNotFoundError`.
        Overrides the default message if provided.

    Returns
    -------
    bool
        `True` if all objects are compatible with the current environment; `False` otherwise.

    Raises
    ------
    ModuleNotFoundError
        If `severity` is set to "error" and incompatibility is detected.
    ValueError
        If an invalid severity level is provided.
    TypeError
        If `obj` is not a `BaseObject` descendant or a list/tuple thereof.
    """

    # Define an inner Pydantic model for validating input parameters
    class DependencyCheckConfig(BaseModel):
        """
        Pydantic model for configuring dependency checks.

        Attributes
        ----------
        severity : SeverityEnum
            Determines the behavior when incompatibility is detected.
        msg : Optional[str]
            Custom error message to be used in the `ModuleNotFoundError`.
        """

        severity: SeverityEnum = Field(
            default=SeverityEnum.ERROR,
            description=(
                "Determines the behavior when incompatibility is detected.\n"
                "- 'error': Raises a `ModuleNotFoundError`.\n"
                "- 'warning': Emits a warning and returns `False`.\n"
                "- 'none': Silently returns `False` without raising an exception or warning."
            ),
        )
        msg: Optional[str] = Field(
            default=None,
            description=(
                "Custom error message to be used in the `ModuleNotFoundError`. "
                "Overrides the default message if provided."
            ),
        )

        @field_validator("severity", mode="before")
        @classmethod
        def validate_severity(
            cls, v: Union[str, SeverityEnum]
        ) -> SeverityEnum:
            """
            Validate and convert the severity level to SeverityEnum.

            Parameters
            ----------
            v : Union[str, SeverityEnum]
                The severity level to validate.

            Returns
            -------
            SeverityEnum
                The validated severity level.

            Raises
            ------
            ValueError
                If the severity level is not one of the defined Enum members.
            """
            if isinstance(v, str):
                try:
                    return SeverityEnum(v.lower())
                except ValueError:
                    raise ValueError(
                        f"Invalid severity level '{v}'. Choose {[level.value for level in SeverityEnum]}"
                    ) from None
            elif isinstance(v, SeverityEnum):
                return v
            else:
                raise TypeError(
                    f"Severity must be a string or an instance of SeverityEnum, got {type(v)}."
                )

    try:
        # Instantiate DependencyCheckConfig to validate severity and msg
        config = DependencyCheckConfig(severity=severity, msg=msg)  # type: ignore
    except ValidationError as ve:
        # Re-raise as a ValueError with detailed message
        raise ValueError(f"Invalid input parameters: {ve}") from ve

    def _check_single_dependency(obj_single: Any) -> bool:
        """
        Check dependencies for a single object.

        Parameters
        ----------
        obj_single : Any
            A single `BaseObject` descendant to check.

        Returns
        -------
        bool
            `True` if the object is compatible; `False` otherwise.
        """
        if not hasattr(obj_single, "get_class_tag"):
            raise TypeError(
                f"Object {obj_single} does not have 'get_class_tag' method."
            )

        compatible = True

        # Check Python version compatibility
        if not _check_python_version(
            obj_single, severity=config.severity.value
        ):
            compatible = False
            message = (
                config.msg or f"Python version incompatible for {obj_single}."
            )
            if config.severity == SeverityEnum.ERROR:
                raise ModuleNotFoundError(message)
            elif config.severity == SeverityEnum.WARNING:
                logger.warning(message)

        # Check soft dependencies
        pkg_deps = obj_single.get_class_tag("python_dependencies", None)
        pkg_alias = obj_single.get_class_tag("python_dependencies_alias", None)

        if pkg_deps:
            if not isinstance(pkg_deps, list):
                pkg_deps = [pkg_deps]
            if not _check_soft_dependencies(
                *pkg_deps,
                severity=config.severity.value,
                obj=obj_single,
                package_import_alias=pkg_alias,
            ):
                compatible = False
                message = (
                    config.msg or f"Missing dependencies for {obj_single}."
                )
                if config.severity == SeverityEnum.ERROR:
                    raise ModuleNotFoundError(message)
                elif config.severity == SeverityEnum.WARNING:
                    logger.warning(message)

        return compatible

    compatible = True

    # If obj is a list or tuple, iterate and check each element
    if isinstance(obj, (list, tuple)):
        for item in obj:
            try:
                item_compatible = _check_single_dependency(item)
                compatible = compatible and item_compatible
                # Early exit if incompatibility detected and severity is ERROR
                if not compatible and config.severity == SeverityEnum.ERROR:
                    break
            except (ModuleNotFoundError, TypeError, ValueError):
                if config.severity == SeverityEnum.ERROR:
                    raise
                elif config.severity == SeverityEnum.WARNING:
                    compatible = False
        return compatible

    # Single object check
    return _check_single_dependency(obj)

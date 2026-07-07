"""Exceptions for foundation-model forecasters."""


class FoundationModelError(Exception):
    """Base exception for foundation-model infrastructure errors."""


class FoundationDependencyError(FoundationModelError, ImportError):
    """Raised when an optional foundation-model dependency is missing."""


class FoundationModelLoadError(FoundationModelError):
    """Raised when loading foundation-model state fails."""

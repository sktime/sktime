"""Import a module/class, return a Mock object if import fails."""

from skbase.utils.dependencies._import import _safe_import

__all__ = ["_safe_import"]

"""Import a module/class, return a Mock object if import fails."""

import importlib
from unittest.mock import MagicMock

from sktime.utils.dependencies import _check_soft_dependencies


def _safe_import(import_path, pkg_name=None):
    """Import a module/class, return a Mock object if import fails.

    Parameters
    ----------
    import_path : str
        The path to the module/class to import.
    pkg_name : str, default=None
        The name of the package to import.
        If None, the first part of the import_path is used.
    """
    if pkg_name is None:
        pkg_name = import_path.split(".")[0]

    if _check_soft_dependencies(pkg_name, severity="none"):
        return importlib.import_module(import_path)

    else:
        mock_obj = MagicMock()
        mock_obj.__call__ = MagicMock(
            return_value=f"Please install {pkg_name} to use this functionality."
        )
        return mock_obj

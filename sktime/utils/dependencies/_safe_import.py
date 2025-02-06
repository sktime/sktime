"""Import a module/class, return a Mock object if import fails."""

import importlib
from unittest.mock import MagicMock

from sktime.utils.dependencies import _check_soft_dependencies


def _safe_import(import_path, pkg_name=None):
    """Import a module/class, return a Mock object if import fails.

    Parameters
    ----------
    import_path : str
        The path to the module/class to import (e.g., "torch.nn.ReLU").
    pkg_name : str, default=None
        The name of the package to import.
        If None, the first part of the import_path is used.

    Returns
    -------
    object
        The imported module, class, or function. If the package is missing,
        returns a MagicMock object that informs the user to install the package.
    """
    if pkg_name is None:
        path_list = import_path.split(".")
        pkg_name = path_list[0]

    if _check_soft_dependencies(pkg_name, severity="none"):
        try:
            if len(path_list) == 1:
                return importlib.import_module(pkg_name)
            module_name, attr_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            return importlib.import_module(import_path)
    else:
        mock_obj = MagicMock()
        mock_obj.__call__ = MagicMock(
            return_value=f"Please install {pkg_name} to use this functionality."
        )
        return mock_obj

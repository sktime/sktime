"""Import a module/class, return a Mock object if import fails."""

import importlib
from unittest.mock import MagicMock

from sktime.utils.dependencies import _check_soft_dependencies


def _safe_import(import_path, pkg_name=None):
    """Import a module/class, return a Mock object if import fails.

    The function supports importing both top-level modules and nested attributes:
    - Top-level module: "torch" -> imports torch
    - Nested module: "torch.nn" -> imports torch.nn
    - Class/function: "torch.nn.Linear" -> imports Linear class from torch.nn

    Parameters
    ----------
    import_path : str
        The path to the module/class to import. Can be:
        - Single module: "torch"
        - Nested module: "torch.nn"
        - Class/attribute: "torch.nn.ReLU"
        Note: The dots in the path determine the import behavior:
        - No dots: Imports as a single module
        - One dot: Imports as a submodule
        - Multiple dots: Last part is treated as an attribute to import
    pkg_name : str, default=None
        The name of the package to check for installation. This is useful when
        the import name differs from the package name, for example:
        - import: "sklearn" -> pkg_name="scikit-learn"
        - import: "cv2" -> pkg_name="opencv-python"
        If None, uses the first part of import_path before the dot.

    Returns
    -------
    object
        One of the following:
        - The imported module if import_path has no dots
        - The imported submodule if import_path has one dot
        - The imported class/function if import_path has multiple dots
        - A MagicMock object that returns an installation message if the
          package is not found

    Examples
    --------
    >>> from sktime.utils.dependencies._safe_import import _safe_import

    >>> # Import a top-level module
    >>> torch = _safe_import("torch")

    >>> # Import a submodule
    >>> nn = _safe_import("torch.nn")

    >>> # Import a specific class
    >>> Linear = _safe_import("torch.nn.Linear")

    >>> # Import with different package name
    >>> cv2 = _safe_import("cv2", pkg_name="opencv-python")
    """
    path_list = import_path.split(".")

    if pkg_name is None:
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
        mock_obj.__str__.return_value = (
            f"Please install {pkg_name} to use this functionality."
        )
        return mock_obj

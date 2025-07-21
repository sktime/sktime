"""Import a module/class, return a Mock object if import fails."""

import importlib
from unittest.mock import MagicMock

from sktime.utils.dependencies import _check_soft_dependencies


def _safe_import(import_path, pkg_name=None):
    """Import a module/class, return a Mock object if import fails.

    Idiomatic usage is ``obj = _safe_import("a.b.c.obj")``.
    The function supports importing both top-level modules and nested attributes:

    - Top-level module: ``"torch"`` -> same as ``import torch``
    - Nested module: ``"torch.nn"`` -> same as``from torch import nn``
    - Class/function: ``"torch.nn.Linear"`` -> same as ``from torch.nn import Linear``

    Parameters
    ----------
    import_path : str
        The path to the module/class to import. Can be:

        - Single module: ``"torch"``
        - Nested module: ``"torch.nn"``
        - Class/attribute: ``"torch.nn.ReLU"``

        Note: The dots in the path determine the import behavior:

        - No dots: Imports as a single module
        - One dot: Imports as a submodule
        - Multiple dots: Last part is treated as an attribute to import

    pkg_name : str, default=None
        The name of the package to check for installation. This is useful when
        the import name differs from the package name, for example:

        - import: ``"sklearn"`` -> ``pkg_name="scikit-learn"``
        - import: ``"cv2"`` -> ``pkg_name="opencv-python"``

        If ``None``, uses the first part of ``import_path`` before the dot.

    Returns
    -------
    object
        If the import path and ``pkg_name`` is present, one of the following:

        - The imported module if ``import_path`` has no dots
        - The imported submodule if ``import_path`` has one dot
        - The imported class/function if ``import_path`` has multiple dots

        If the package or import path are not found:
        a unique ``MagicMock`` object per unique import path.

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
    obj_name = path_list[-1]

    if _check_soft_dependencies(pkg_name, severity="none"):
        try:
            if len(path_list) == 1:
                return importlib.import_module(pkg_name)
            module_name, attr_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            pass

    mock_obj = _create_mock_class(obj_name)
    return mock_obj


class CommonMagicMeta(type):
    def __getattr__(cls, name):
        return MagicMock()

    def __setattr__(cls, name, value):
        pass  # Ignore attribute writes


class MagicAttribute(metaclass=CommonMagicMeta):
    def __getattr__(self, name):
        return MagicMock()

    def __setattr__(self, name, value):
        pass  # Ignore attribute writes

    def __call__(self, *args, **kwargs):
        return self  # Ensures instantiation returns the same object


def _create_mock_class(name: str, bases=()):
    """Create new dynamic mock class similar to MagicMock.

    Parameters
    ----------
    name : str
        The name of the new class.
    bases : tuple, default=()
        The base classes of the new class.

    Returns
    -------
    a new class that behaves like MagicMock, with name ``name``.
        Forwards all attribute access to a MagicMock object stored in the instance.
    """
    return type(name, (MagicAttribute,), {"__metaclass__": CommonMagicMeta})

"""Lookup of classes in a module, by case insensitive name."""

__author__ = ["srupat"]
__all__ = []

from functools import cache
from types import MappingProxyType

from sktime.utils.dependencies import _safe_import


@cache
def _lc_class_dict(module_path, base_class_path=None):
    """Get lower case name to class lookup, for the classes in a module.

    Collects the public classes in the module at ``module_path``, and returns
    a lookup of their lower cased names to the classes themselves.

    The return is cached, as the contents of a module do not change at runtime.

    Parameters
    ----------
    module_path : str
        Import path of the module to collect the classes from,
        e.g., ``"torch.optim"``.
    base_class_path : str, optional
        Import path of a base class, e.g., ``"torch.optim.Optimizer"``.
        If passed, only strict subclasses of that class are collected,
        i.e., the base class itself is excluded, as are objects in the module
        that are not classes, e.g., sub-modules or functions.
        If not passed, all public classes in the module are collected.

    Returns
    -------
    read-only dict, of str to class
        Keys are the lower cased names of the classes, values are the classes.
        Read-only, since the return is cached and shared between calls.
        Empty if ``module_path`` or ``base_class_path`` cannot be imported,
        e.g., if the soft dependency providing them is not present.
    """
    module = _safe_import(module_path, return_object="None")
    if module is None:
        return MappingProxyType({})

    base_class = None
    if base_class_path is not None:
        base_class = _safe_import(base_class_path, return_object="None")
        if not isinstance(base_class, type):
            return MappingProxyType({})

    lookup = {}
    # sorted, so that the entry is deterministic if two names differ only in case
    for name in sorted(vars(module)):
        obj = vars(module)[name]
        if name.startswith("_") or not isinstance(obj, type):
            continue
        if base_class is not None:
            if obj is base_class or not issubclass(obj, base_class):
                continue
        lookup[name.lower()] = obj
    return MappingProxyType(lookup)


def _lookup_class(name, module_path, base_class_path=None, alias_dict=None):
    """Look up a class in a module, by case insensitive name.

    Parameters
    ----------
    name : str
        Name of the class to look up, in any capitalization,
        e.g., ``"adamw"`` or ``"AdamW"`` for ``torch.optim.AdamW``.
    module_path : str
        Import path of the module to look the class up in,
        e.g., ``"torch.optim"``.
    base_class_path : str, optional
        Import path of a base class, e.g., ``"torch.optim.Optimizer"``.
        If passed, only strict subclasses of that class are looked up,
        see ``_lc_class_dict``.
    alias_dict : dict, of str to str, optional
        Aliases to resolve before the lookup, keys and values are both
        interpreted case insensitively.
        Keys are the aliases, values are names of classes in the module.

    Returns
    -------
    class, or None
        The class in the module at ``module_path`` whose name is ``name``,
        case insensitively, after resolving ``alias_dict``.
        None if the module contains no such class.
    """
    key = name.lower()
    if alias_dict is not None:
        key = alias_dict.get(key, key).lower()
    return _lc_class_dict(module_path, base_class_path).get(key, None)

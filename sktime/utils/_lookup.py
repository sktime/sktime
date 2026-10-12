"""Utilities for looking up objects in modules by name or alias."""

__author__ = ["KingLizard1020", "srupat"]
__all__ = ["_lowercase_importer", "_lookup"]

from functools import cache
from types import MappingProxyType

from sktime.utils.dependencies import _safe_import


def _lowercase_importer(module_path):
    """Import ``module_path`` and map lower-case names to canonical names.

    Parameters
    ----------
    module_path : str
        Dotted path of the module to inspect, for example ``"torch.optim"``.

    Returns
    -------
    dict
        Mapping from ``name.lower()`` to the original exported name.

    Raises
    ------
    ModuleNotFoundError
        If ``module_path`` cannot be imported.
    """
    module = _safe_import(module_path, return_object="None")
    if module is None:
        raise ModuleNotFoundError(f"Cannot import {module_path!r}.")
    return {name.lower(): name for name in dir(module)}


def _lookup(alias, module_path, alias_dict=None):
    """Resolve ``alias`` to an object imported from ``module_path``.

    Looks up ``alias`` case-insensitively among names exported by
    ``module_path``. An optional ``alias_dict`` of extra aliases (typically
    lower-case keys to canonical names) is merged in, but only for keys that
    are not already exported by the module, and only when the alias target
    is itself an exported name. Existing ``dir()`` names are never overwritten
    by a bad alias value.

    If a match is found, the object is imported with ``_safe_import``.
    Otherwise a ``ValueError`` is raised.

    Parameters
    ----------
    alias : str
        Name or alias of the object to import.
    module_path : str
        Dotted path of the module to search, for example ``"torch.optim"``.
    alias_dict : dict, optional
        Optional mapping of aliases to canonical names. Keys are matched
        case-insensitively. Entries whose key is already exported by
        ``module_path``, or whose target is not an exported name, are ignored.

    Returns
    -------
    object
        The imported object.

    Raises
    ------
    ValueError
        If ``alias`` cannot be resolved to a name in ``module_path``.
    ModuleNotFoundError
        If ``module_path`` cannot be imported.
    """
    name_map = _lowercase_importer(module_path)
    if alias_dict is not None:
        exported = dict(name_map)
        for key, target in alias_dict.items():
            key_lc = str(key).lower()
            # do not replace a name already exported by the module
            if key_lc in name_map:
                continue
            target_lc = str(target).lower()
            # only accept aliases that point at a real exported name
            if target_lc not in exported:
                continue
            name_map[key_lc] = exported[target_lc]

    if not isinstance(alias, str) or alias.lower() not in name_map:
        raise ValueError(f"{alias!r} is not a valid name in {module_path}.")

    canonical = name_map[alias.lower()]
    obj = _safe_import(f"{module_path}.{canonical}", return_object="None")
    if obj is None:
        # The name may exist only on the live module object, e.g. a test
        # monkeypatch that is visible to ``dir()`` but not as a dotted import.
        parent = _safe_import(module_path, return_object="None")
        obj = getattr(parent, canonical, None) if parent is not None else None
    if obj is None:
        raise ValueError(f"{alias!r} is not a valid name in {module_path}.")
    return obj


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

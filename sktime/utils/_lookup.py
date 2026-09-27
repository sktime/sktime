"""Utilities for looking up objects in modules by name or alias."""

__author__ = ["KingLizard1020"]
__all__ = ["_lowercase_importer", "_lookup"]

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

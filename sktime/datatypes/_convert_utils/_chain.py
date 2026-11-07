# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Search for conversion paths in the graph of defined conversions."""

__author__ = ["adity1raut"]

from collections import deque

from sktime.datatypes._base._chain import ConverterChain, _is_lossy

# cache of conversion paths, keyed by (from_type, to_type, scitype)
_PATH_CACHE = {}
# number of conversions the cache was computed from, for cache invalidation
_PATH_CACHE_SIZE = None


def _conversion_graph(convert_dict, scitype, lossy_ok=True):
    """Return the graph of conversions defined for a scitype.

    Parameters
    ----------
    convert_dict : dict with keys (str, str, str), entries converters
        conversion dictionary, see ``datatypes._convert``
    scitype : str
        scitype to restrict the graph to
    lossy_ok : bool, optional (default=True)
        whether to include lossy conversions in the graph

    Returns
    -------
    adjacency : dict with keys str, entries list of str
        ``adjacency[A]`` are the mtypes that a conversion from ``A`` is
        defined to. Identity conversions are not included.
    """
    adjacency = {}

    for key, converter in convert_dict.items():
        mtype_from, mtype_to, key_scitype = key
        if key_scitype != scitype or mtype_from == mtype_to:
            continue
        if not lossy_ok and _is_lossy(converter):
            continue
        adjacency.setdefault(mtype_from, []).append(mtype_to)

    return adjacency


def _shortest_path(adjacency, from_type, to_type):
    """Return a shortest path from from_type to to_type, by breadth-first search.

    Parameters
    ----------
    adjacency : dict with keys str, entries list of str
        graph of conversions, see ``_conversion_graph``
    from_type, to_type : str
        mtypes to find a path between. Must be distinct.

    Returns
    -------
    list of str, or None
        mtypes on a shortest path, starting with ``from_type`` and ending with
        ``to_type``. None if no path exists.
    """
    if from_type == to_type:
        return None

    predecessor = {from_type: None}
    queue = deque([from_type])

    while queue:
        current = queue.popleft()
        # sorted, to make the path returned deterministic
        for neighbour in sorted(adjacency.get(current, [])):
            if neighbour in predecessor:
                continue
            predecessor[neighbour] = current
            if neighbour == to_type:
                return _reconstruct_path(predecessor, to_type)
            queue.append(neighbour)

    return None


def _reconstruct_path(predecessor, to_type):
    """Reconstruct the path to to_type from a dict of predecessors."""
    path = [to_type]
    while predecessor[path[-1]] is not None:
        path.append(predecessor[path[-1]])
    return list(reversed(path))


def get_conversion_path(from_type, to_type, scitype, convert_dict):
    """Return a conversion path from from_type to to_type.

    A shortest non-lossy path is returned if one exists, otherwise a shortest
    path overall. Lossiness of a conversion is queried via the ``lossy`` tag;
    conversions that do not declare lossiness are assumed to be non-lossy.

    Parameters
    ----------
    from_type, to_type : str
        mtypes to find a conversion path between
    scitype : str
        scitype of ``from_type`` and ``to_type``
    convert_dict : dict with keys (str, str, str), entries converters
        conversion dictionary, see ``datatypes._convert``

    Returns
    -------
    list of str, or None
        mtypes on the conversion path, starting with ``from_type`` and ending
        with ``to_type``. None if no path exists.
    """
    for lossy_ok in [False, True]:
        adjacency = _conversion_graph(convert_dict, scitype, lossy_ok=lossy_ok)
        path = _shortest_path(adjacency, from_type, to_type)
        if path is not None:
            return path

    return None


def _get_cached_conversion_path(from_type, to_type, scitype, convert_dict):
    """Return ``get_conversion_path``, cached over calls.

    The cache is invalidated if the number of conversions in ``convert_dict``
    has changed since it was computed, e.g., if conversions were added after
    the first lookup.
    """
    global _PATH_CACHE_SIZE

    if _PATH_CACHE_SIZE != len(convert_dict):
        _PATH_CACHE.clear()
        _PATH_CACHE_SIZE = len(convert_dict)

    key = (from_type, to_type, scitype)
    if key not in _PATH_CACHE:
        _PATH_CACHE[key] = get_conversion_path(
            from_type, to_type, scitype, convert_dict
        )

    return _PATH_CACHE[key]


def get_converter_chain(from_type, to_type, scitype, convert_dict):
    """Return a chain of conversions from from_type to to_type.

    Parameters
    ----------
    from_type, to_type : str
        mtypes to obtain a conversion between
    scitype : str
        scitype of ``from_type`` and ``to_type``
    convert_dict : dict with keys (str, str, str), entries converters
        conversion dictionary, see ``datatypes._convert``

    Returns
    -------
    ConverterChain, or None
        conversion from ``from_type`` to ``to_type``, obtained by chaining
        conversions in ``convert_dict``. None if no chain exists.
    """
    path = _get_cached_conversion_path(from_type, to_type, scitype, convert_dict)

    if path is None:
        return None

    converters = [
        convert_dict[(mtype_from, mtype_to, scitype)]
        for mtype_from, mtype_to in zip(path[:-1], path[1:])
    ]

    return ConverterChain(converters=converters, mtype_from=from_type, mtype_to=to_type)

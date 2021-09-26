# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Machine type converters for scitypes.

Exports
-------
convert_to(obj, to_type: str, as_scitype: str, store=None)
    converts object "obj" to type "to_type", considerd as "as_scitype"

convert(obj, from_type: str, to_type: str, as_scitype: str, store=None)
    same as convert_to, without automatic identification of "from_type"

mtype(obj, as_scitype: str)
    returns "from_type" of obj, considered as "as_scitype"
---

Function signature of convert

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
from_type : str - the type to convert "obj" to, a valid mtype string
to_type : str - the type to convert "obj" to, a valid mtype string
as_scitype : str - name of scitype the object "obj" is considered as
store : reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

---

Function signature of convert_to

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
to_type : str - the type to convert "obj" to, a valid mtype string
as_scitype : str - name of scitype the object "obj" is considered as
store : reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

---

Function signature of mtype

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
as_scitype : str - name of scitype the object "obj" is considered as

Returns
-------
str - the type to convert "obj" to, a valid mtype string
    or None, if obj is None
"""

__author__ = ["fkiraly"]

__all__ = [
    "convert",
    "convert_to",
]

import numpy as np
import pandas as pd
from typing import Union, List

from sktime.datatypes._series import convert_dict_Series

from sktime.datatypes._panel import convert_dict_Panel

from sktime.datatypes._check import mtype as infer_mtype
from sktime.datatypes._registry import mtype_to_scitype
from sktime.datatypes.types import Mtype, SciType
from sktime.datatypes._registry import SeriesMtype, PanelMtype, Scitype
from sktime.datatypes._mtype_enum import MtypeEnum

# pool convert_dict-s and infer_mtype_dict-s
convert_dict = dict()
convert_dict.update(convert_dict_Series)
convert_dict.update(convert_dict_Panel)


def convert(
    obj, from_type: Mtype, to_type: Mtype, as_scitype: SciType = None, store=None
):
    """Convert objects between different machine representations, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    from_type: str or SeriesMtype enum or PanelMtype enum
        the type to convert "obj" to, a valid mtype string
    to_type: str or SeriesMtype enum or PanelMtype enum
        the type to convert "obj" to, a valid mtype string
    as_scitype : str or Scitype enum optional, defaults = inferred from from_type
        name of scitype the object "obj" is considered as
        default = inferred from from_type
    store : reference of storage for lossy conversions, default=None (no store)

    Returns
    -------
    converted_obj : to_type - object obj converted to to_type
                    if obj was None, returns None

    Raises
    ------
    KeyError if conversion is not implemented
    """
    if from_type is not None:
        from_type = str(from_type)
    if to_type is not None:
        to_type = str(to_type)
    if as_scitype is not None:
        as_scitype = str(as_scitype)

    if obj is None:
        return None
    if from_type == to_type:
        return obj

    if not isinstance(to_type, str):
        raise TypeError("to_type must be a str")
    if not isinstance(from_type, str):
        raise TypeError("from_type must be a str")
    if as_scitype is None:
        as_scitype = mtype_to_scitype(to_type)
    elif not isinstance(as_scitype, str):
        raise TypeError("as_scitype must be str or None")

    key = (from_type, to_type, as_scitype)

    allow_lossy = True

    if store is not None:
        allow_lossy = False

    if key not in convert_dict.keys():
        conversion_path = _find_conversion_path(from_type, to_type, allow_lossy)
        if len(conversion_path) > 0:
            converted_object = obj
            curr_type = from_type
            for next_type in conversion_path:
                converted_object = convert(
                    obj=converted_object,
                    from_type=curr_type,
                    to_type=next_type,
                    as_scitype=as_scitype,
                )
                curr_type = next_type
            return converted_object
        else:
            raise NotImplementedError(
                "no conversion defined from type "
                + str(from_type)
                + " to "
                + str(to_type)
            )
    else:
        return convert_dict[key](obj, store=store)


def convert_to(
    obj,
    to_type: Union[Mtype, List[Mtype]],
    as_scitype: SciType = None,
    store=None,
):
    """Convert object to a different machine representation, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    to_type: str or SeriesMtype enum or PanelMtype enum or List of Mtypes
        the type to convert "obj" to, a valid mtype string
    as_scitype : str or Scitype enum optional, defaults = inferred from mtype
        name of scitype the object "obj" is considered as
        default = inferred from from_type
    store : reference of storage for lossy conversions, default=None (no store)

    Returns
    -------
    converted_obj : to_type - object obj converted to to_type, if to_type is str
                     if to_type is list, converted to to_type[0],
                        unless from_type in to_type, in this case converted_obj=obj
                    if obj was None, returns None

    Raises
    ------
    TypeError if machine type of input "obj" is not recognized
    KeyError if conversion is not implemented
    """
    if isinstance(to_type, list):
        temp = []
        for val in to_type:
            temp.append(str(val))
        to_type = temp
    else:
        if to_type is not None:
            to_type = str(to_type)

    if as_scitype is not None:
        as_scitype = str(as_scitype)

    if obj is None:
        return None

    if isinstance(to_type, list):
        if not np.all(isinstance(x, str) for x in to_type):
            raise TypeError("to_type must be a str or list of str")
    elif not isinstance(to_type, str):
        raise TypeError("to_type must be a str or list of str")

    if as_scitype is None:
        if isinstance(to_type, str):
            as_scitype = mtype_to_scitype(to_type)
        else:
            as_scitype = mtype_to_scitype(to_type[0])
    elif not isinstance(as_scitype, str):
        raise TypeError("as_scitype must be a str or None")

    from_type = infer_mtype(obj=obj, as_scitype=as_scitype)

    # if to_type is a list:
    if isinstance(to_type, list):
        # no conversion of from_type is in the list
        if from_type in to_type:
            to_type = from_type
        # otherwise convert to first element
        else:
            to_type = to_type[0]

    converted_obj = convert(
        obj=obj,
        from_type=from_type,
        to_type=to_type,
        as_scitype=as_scitype,
        store=store,
    )

    return converted_obj


def _conversions_defined(scitype: SciType, allow_lossy: bool = True):
    """Return an indicator matrix which conversions are defined for scitype.

    Parameters
    ----------
    scitype: str or Scitype enum
        name of scitype for which conversions are queried
    allow_lossy: bool, defaults = True
        Boolean that when true allows path that would result in a loss of data e.g.
        column names. When False paths that result in loss of data are now allowed

    Returns
    -------
    conv_df: pd.DataFrame, columns and index is list of mtypes for scitype
            entry of row i, col j is 1 if conversion from i to j is defined,
                                     0 if conversion from i to j is not defined
    """
    if scitype is not None:
        scitype = str(scitype)

    if scitype is str(Scitype.panel):
        lookup_enum = PanelMtype
    else:
        lookup_enum = SeriesMtype

    pairs = [(x[0], x[1]) for x in list(convert_dict.keys()) if x[2] == scitype]
    cols0 = set([x[0] for x in list(convert_dict.keys()) if x[2] == scitype])
    cols1 = set([x[1] for x in list(convert_dict.keys()) if x[2] == scitype])
    cols = sorted(list(cols0.union(cols1)))

    mat = np.zeros((len(cols), len(cols)), dtype=int)
    nkeys = len(cols)
    for i in range(nkeys):
        for j in range(nkeys):
            if (cols[i], cols[j]) in pairs:
                first_col_enum = lookup_enum[cols[i]]
                second_col_enum = lookup_enum[cols[j]]
                if allow_lossy is True or lookup_enum[cols[i]].is_lossy is False:
                    mat[i, j] = 1
                elif (
                    first_col_enum.is_lossy is False
                    and second_col_enum.is_lossy is False
                ):
                    mat[i, j] = 1

    conv_df = pd.DataFrame(mat, index=cols, columns=cols)

    return conv_df


def _find_conversion_path(
    from_type: Mtype, to_type: Mtype, allow_lossy: bool = True
) -> List[str]:
    """Find a path both direct and indirect using dijsktras optimal path.

    Adapted from:
    https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/

    Parameters
    ----------
    from_type: str
        Type to convert from
    to_type: str
        Type to convert to
    allow_lossy: bool
        Boolean that defines if the conversion can be lossy

    Returns
    -------
    List[str]
        Conversion path to get the desired result. If there is no path found then
        [] is returned.
    """

    def resolve_enum_str(enum_val: Mtype, err_str: str) -> MtypeEnum:
        if isinstance(enum_val, MtypeEnum):
            return enum_val
        else:
            if enum_val in PanelMtype:
                return PanelMtype[enum_val]
            elif enum_val in SeriesMtype:
                return SeriesMtype[enum_val]
            else:
                raise ValueError(
                    f"Unable to find conversion path as the parameter"
                    f"{err_str} is not a valid mtype. Check the mtype"
                    f"passed is valid."
                )

    if from_type == to_type:
        return []
    from_type: MtypeEnum = resolve_enum_str(from_type, "from_type")
    to_type: MtypeEnum = resolve_enum_str(to_type, "to_type")

    if isinstance(from_type, PanelMtype):
        scitype = Scitype.panel
    else:
        scitype = Scitype.series

    # Checks if we're starting from a lossy value
    if allow_lossy is False:
        allow_lossy = from_type.is_lossy

    adjacency_matrix = _conversions_defined(
        scitype=str(scitype), allow_lossy=allow_lossy
    )
    current_node = str(from_type)
    visited = set()

    paths = {str(from_type): (None, 0)}

    while current_node != str(to_type):
        visited.add(current_node)
        potential_paths = [
            x for x in adjacency_matrix.columns if adjacency_matrix[current_node][x] > 0
        ]
        weight_to_current_node = paths[current_node][1]

        for node in potential_paths:
            weight = 1 + weight_to_current_node
            if node not in paths:
                paths[node] = (current_node, weight)
            else:
                current_shortest = paths[node][1]
                if current_shortest > weight:
                    paths[node] = (node, weight)

        next_node = {node: paths[node] for node in paths if node not in visited}

        if not next_node:
            return []
        current_node = min(next_node, key=lambda k: next_node[k][1])

    path = []
    while current_node is not None and current_node is not str(from_type):
        path.append(current_node)
        next_node = paths[current_node][0]
        current_node = next_node
    path = path[::-1]
    return path

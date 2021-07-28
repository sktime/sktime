# -*- coding: utf-8 -*-
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
    "mtype",
]


from sktime.datatypes._series import convert_dict_Series
from sktime.datatypes._series import infer_mtype_dict_Series

from sktime.datatypes._panel import convert_dict_Panel

# pool convert_dict-s and infer_mtype_dict-s
convert_dict = dict()
convert_dict.update(convert_dict_Series)
convert_dict.update(convert_dict_Panel)

infer_mtype_dict = dict()
infer_mtype_dict.update(infer_mtype_dict_Series)


def mtype(obj, as_scitype: str):
    """Infer the mtype of an object considered as a specific scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    as_scitype : str - name of scitype the object "obj" is considered as

    Returns
    -------
    str - the type to convert "obj" to, a valid mtype string
        or None, if obj is None

    Raises
    ------
    TypeError if no type can be identified
    """
    if obj is None:
        return None

    valid_as_scitypes = infer_mtype_dict.keys()

    if as_scitype not in valid_as_scitypes:
        raise TypeError(as_scitype + " is not a valid scitype")

    return infer_mtype_dict[as_scitype](obj=obj)


def convert(obj, from_type: str, to_type: str, as_scitype: str, store=None):
    """Convert objects between different machine representations, subject to scitype.

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
                    if obj was None, returns None

    Raises
    ------
    KeyError if conversion is not implemented
    """
    if obj is None:
        return None

    key = (from_type, to_type, as_scitype)

    if key not in convert_dict.keys():
        raise NotImplementedError(
            "no conversion defined from type " + str(from_type) + " to " + str(to_type)
        )

    converted_obj = convert_dict[key](obj, store=store)

    return converted_obj


# conversion based on queriable type to specified target
def convert_to(obj, to_type: str, as_scitype: str, store=None):
    """Convert object to a different machine representation, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    to_type : str - the type to convert "obj" to, a valid mtype string
              or list - admissible types for conversion to
    as_scitype : str - name of scitype the object "obj" is considered as
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
    if obj is None:
        return None

    from_type = mtype(obj=obj, as_scitype=as_scitype)

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

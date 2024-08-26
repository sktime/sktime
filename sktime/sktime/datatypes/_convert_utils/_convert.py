"""Conversion utilities for mtypes."""

__author__ = ["fkiraly"]


def _concat(fun1, fun2):
    """Concatenation of two converter functions, using the same store.

    Parameters
    ----------
    fun1, fun2 : functions in converter signature, see datatypes._convert

    Returns
    -------
    function in converter signature, see datatypes._convert
        concatenation fun2 o fun1, using the same store
    """

    def concat_fun(obj, store=None):
        obj1 = fun1(obj, store=store)
        obj2 = fun2(obj1, store=store)
        return obj2

    return concat_fun


def _extend_conversions(mtype, anchor_mtype, convert_dict, mtype_universe=None):
    """Obtain all conversions from and to mtype via conversion to anchor_mtype.

    Mutates convert_dict by adding all conversions from and to mtype.

    Assumes:
    convert_dict contains
    * conversion from `mtype` to `anchor_mtype`
    * conversion from `anchor_mtype` to `mtype`
    * conversions from `anchor_mtype` to all mtypes in `mtype_universe`
    * conversions from all mtypes in `mtype_universe` to `anchor_mtype`

    Guarantees:
    convert_dict contains
    * conversions from `mtype` to all mtypes in mtype_universe
    * conversions from all mtypes in mtype_universe to `mtype`

    conversions <mtype to mtype2> not in convert_dict at start are filled in as
    _concat(<from mtype to anchor_type>, <from anchor_type to mtype2>)
    conversions <mtype2 to mtype> not in convert_dict at start are filled in as
    _concat(<from mtype2 to anchor_type>, <from anchor_type to mtype>)

    Parameters
    ----------
    mtype : mtype string in convert_dict
    anchor_mtype : mtype string in convert_dict
    convert_dict : conversion dictionary with entries of converter signature
        see docstring of datatypes._convert
    mtype_universe : iterable of mtype strings in convert_dict, coercible to list or set

    Returns
    -------
    reference to convert_dict
    CAVEAT: convert_dict passed to this function gets mutated, this is a reference
    """
    keys = convert_dict.keys()
    scitype = list(keys)[0][2]

    if mtype_universe is None:
        mtype_universe = {x[1] for x in list(keys)}
        mtype_universe = mtype_universe.union([x[0] for x in list(keys)])

    for tp in set(mtype_universe).difference([mtype, anchor_mtype]):
        if (anchor_mtype, tp, scitype) in convert_dict.keys():
            if (mtype, tp, scitype) not in convert_dict.keys():
                convert_dict[(mtype, tp, scitype)] = _concat(
                    convert_dict[(mtype, anchor_mtype, scitype)],
                    convert_dict[(anchor_mtype, tp, scitype)],
                )
        if (tp, anchor_mtype, scitype) in convert_dict.keys():
            if (tp, mtype, scitype) not in convert_dict.keys():
                convert_dict[(tp, mtype, scitype)] = _concat(
                    convert_dict[(tp, anchor_mtype, scitype)],
                    convert_dict[(anchor_mtype, mtype, scitype)],
                )

    return convert_dict

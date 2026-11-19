"""Conversion utilities for mtypes."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseConverter


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


class _ConverterConcat(BaseConverter):
    """Conversion as concatenation of two converters, using the same store.

    Parameters
    ----------
    converter_first : BaseConverter, conversion from mtype_from to an anchor mtype
    converter_second : BaseConverter, conversion from the anchor mtype to mtype_to
    """

    _tags = {
        "multiple_conversions": True,
    }

    def __init__(self, converter_first, converter_second):
        self.converter_first = converter_first
        self.converter_second = converter_second
        super().__init__(
            mtype_from=converter_first.get_tag("mtype_from"),
            mtype_to=converter_second.get_tag("mtype_to"),
        )

    def _check_conversion(self, mtype_from, mtype_to):
        """Skip the check, conversions of this class are given by its components."""

    def _get_key(self):
        """Get unique dictionary key corresponding to self, from the components."""
        mtype_from = self.converter_first._get_key()[0]
        _, mtype_to, scitype = self.converter_second._get_key()
        return (mtype_from, mtype_to, scitype)

    def _convert(self, obj, store=None):
        obj = self.converter_first(obj, store=store)
        return self.converter_second(obj, store=store)


class _ConverterNoDatatypeCls(BaseConverter):
    """Base for converters from or to mtypes without a checkable datatype class.

    ``BaseConverter._get_key`` takes mtype names and scitype from the
    datatype classes of ``mtype_from`` and ``mtype_to``, as in ``skpro``.
    Some mtypes have no such class, e.g., ``pl.LazyFrame``, or have one that is
    skipped in checks, e.g., ``pd-long`` and ``pd-wide``.

    For these, the mtype name is the string tag, and the scitype is taken from
    the datatype class of the other mtype, or from the mtype register.
    Can be removed once all mtypes have a datatype class in the check registry.
    """

    def _get_key(self):
        """Get unique dictionary key corresponding to self."""
        cls_from, cls_to = self._get_cls_from_to()

        mtype_from = self.get_tag("mtype_from")
        mtype_to = self.get_tag("mtype_to")
        if cls_from is not None:
            mtype_from = cls_from.get_class_tag("name")
        if cls_to is not None:
            mtype_to = cls_to.get_class_tag("name")

        if cls_to is not None:
            scitype = cls_to.get_class_tag("scitype")
        elif cls_from is not None:
            scitype = cls_from.get_class_tag("scitype")
        else:
            from sktime.datatypes._registry import mtype_to_scitype

            scitype = mtype_to_scitype(mtype_to)
        return (mtype_from, mtype_to, scitype)


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

"""Conversion utilities for mtypes."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseConverter



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

    def _convert(self, obj, store=None):
        obj = self.converter_first(obj, store=store)
        return self.converter_second(obj, store=store)


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
    converter_first : callable in converter signature, see datatypes._convert
        conversion from mtype_from to an anchor mtype
    converter_second : callable in converter signature, see datatypes._convert
        conversion from the anchor mtype to mtype_to
    mtype_from : str, mtype converted from
    mtype_to : str, mtype converted to
    """

    _tags = {
        "multiple_conversions": True,
    }

    def __init__(self, converter_first, converter_second, mtype_from, mtype_to):
        self.converter_first = converter_first
        self.converter_second = converter_second
        super().__init__(mtype_from=mtype_from, mtype_to=mtype_to)

    def _check_conversion(self, mtype_from, mtype_to):
        """Skip the check, conversions of this class are given by its components."""

    def _convert(self, obj, store=None):
        obj = self.converter_first(obj, store=store)
        return self.converter_second(obj, store=store)

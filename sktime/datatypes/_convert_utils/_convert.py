"""Conversion utilities for mtypes."""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseConverter


class _ConverterConcat(BaseConverter):
    """Conversion as concatenation of two converters, using the same store.

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
        self.mtype_from = mtype_from
        self.mtype_to = mtype_to
        super(BaseConverter, self).__init__()

        self.set_tags(mtype_from=mtype_from, mtype_to=mtype_to)

    def _convert(self, obj, store=None):
        obj = self.converter_first(obj, store=store)
        return self.converter_second(obj, store=store)

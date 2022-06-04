# -*- coding: utf-8 -*-

__all__ = [
    "convert_dict",
]

from sktime.datatypes._hierarchical._registry import MTYPE_LIST_HIERARCHICAL

# dictionary indexed by triples of types
#  1st element = convert from - type
#  2nd element = convert to - type
#  3rd element = considered as this scitype - string
# elements are conversion functions of machine type (1st) -> 2nd

convert_dict = dict()


def convert_identity(obj, store=None):

    return obj


# assign identity function to type conversion to self
for tp in MTYPE_LIST_HIERARCHICAL:
    convert_dict[(tp, tp, "Hierarchical")] = convert_identity

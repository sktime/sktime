# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Example fixtures for mtypes/scitypes.

Exports
-------
get_examples(mtype: str, as_scitype: str, return_lossy=False)
    retrieves examples for mtype/scitype, and/or whether it is lossy

examples[i] are assumed "content-wise the same" for the same as_scitype, i
    only in different machine representations

the representation is considered "lossy" if the representation is incomplete
    e.g., metadata such as column names are missing
"""


__author__ = ["fkiraly"]

__all__ = [
    "get_examples",
]

from sktime.datatypes._alignment import example_dict_Alignment
from sktime.datatypes._panel import example_dict_lossy_Panel, example_dict_Panel
from sktime.datatypes._series import example_dict_lossy_Series, example_dict_Series

# pool example_dict-s
example_dict = dict()
example_dict.update(example_dict_Alignment)
example_dict.update(example_dict_Series)
example_dict.update(example_dict_Panel)

example_dict_lossy = dict()
example_dict_lossy.update(example_dict_lossy_Series)
example_dict_lossy.update(example_dict_lossy_Panel)


def get_examples(mtype: str, as_scitype: str, return_lossy: bool = False):
    """Retrieve a dict of examples for mtype `mtype`, scitype `as_scitype`.

    Parameters
    ----------
    mtype: str - name of the mtype for the example
    as_scitype: str - name of scitype for the example
    return_lossy: bool, optional, default=False
        whether to return second argument

    Returns
    -------
    fixtures: dict with integer keys, elements being
        fixture - example for mtype `mtype`, scitype `as_scitype`
        if return_lossy=True, elements are pairs with fixture and
        lossy: bool - whether the example is a lossy representation
    """
    # retrieve all keys that match the query
    exkeys = example_dict.keys()
    keys = [k for k in exkeys if k[0] == mtype and k[1] == as_scitype]

    # retrieve all fixtures that match the key
    fixtures = dict()

    for k in keys:
        if return_lossy:
            fixtures[k[2]] = (example_dict[k], example_dict_lossy[k])
        else:
            fixtures[k[2]] = example_dict[k]

    return fixtures

# -*- coding: utf-8 -*-
"""Example fixtures for mtypes/scitypes.

Exports
-------
get_example(mtype: str, as_scitype: str, ind=0, return_lossy=False)
    retrieves the ind-th example for mtype/scitype, and/or whether it is lossy

examples are assumed "content-wise the same" for the same as_scitype, ind
    only in different machine representations

the representation is considered "lossy" if the representation is incomplete
    e.g., metadata such as column names are missing
"""


__author__ = ["fkiraly"]

__all__ = [
    "get_example",
]


from sktime.datatypes._series import example_dict_Series
from sktime.datatypes._series import example_dict_lossy_Series
from sktime.datatypes._panel import example_dict_Panel
from sktime.datatypes._panel import example_dict_lossy_Panel

# pool example_dict-s
example_dict = dict()
example_dict.update(example_dict_Series)
example_dict.update(example_dict_Panel)

example_dict_lossy = dict()
example_dict_lossy.update(example_dict_lossy_Series)
example_dict_lossy.update(example_dict_lossy_Panel)


def get_example(mtype: str, as_scitype: str, ind: int = 0, return_lossy: bool = False):
    """Retrieve the ind-th example for mtype mtype, scitype as_scitype.

    Parameters
    ----------
    mtype: str - name of the mtype for the example
    as_scitype: str - name of scitype for the example
    ind: int - index number of the example
    return_lossy: bool, optional, default=False
        whether to return second argument

    Returns
    -------
    fixture - ind-th example for mtype mtype, scitype as_scitype
    if return_lossy=True, also returns:
    lossy: bool - whether the example is a lossy representation
    """
    key = (mtype, as_scitype, ind)

    if key not in example_dict.keys():
        raise KeyError("no example for the requested combination")

    if return_lossy:
        return example_dict[key], example_dict_lossy[key]
    else:
        return example_dict[key]

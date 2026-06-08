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

from copy import deepcopy
from functools import lru_cache

from sktime.datatypes._registry import mtype_to_scitype

__author__ = ["fkiraly"]

__all__ = [
    "get_examples",
]


@lru_cache(maxsize=1)
def generate_example_dicts(soft_deps="present"):
    """Generate example dicts using lookup."""
    from sktime.datatypes._base import BaseExample
    from sktime.utils.dependencies import _check_estimator_deps
    from sktime.utils.retrieval import _all_classes

    classes = _all_classes("sktime.datatypes")
    classes = [x[1] for x in classes]
    classes = [x for x in classes if issubclass(x, BaseExample)]
    classes = [x for x in classes if not x.__name__.startswith("Base")]

    # subset only to data types with soft dependencies present
    if soft_deps == "present":
        classes = [x for x in classes if _check_estimator_deps(x, severity="none")]

    example_dict = dict()
    example_dict_lossy = dict()
    example_dict_metadata = dict()
    for cls in classes:
        k = cls()
        key = k._get_key()
        key_meta = (key[1], key[2])
        example_dict[key] = k
        example_dict_lossy[key] = k.get_class_tags().get("lossy", False)
        example_dict_metadata[key_meta] = k.get_class_tags().get("metadata", {})

    return example_dict, example_dict_lossy, example_dict_metadata


def get_examples(
    mtype: str,
    as_scitype: str = None,
    return_lossy: bool = False,
    return_metadata: bool = False,
):
    """Retrieve a dict of examples for mtype `mtype`, scitype `as_scitype`.

    Parameters
    ----------
    mtype: str - name of the mtype for the example, a valid mtype string
        valid mtype strings, with explanation, are in datatypes.MTYPE_REGISTER
    as_scitype : str, optional - name of scitype of the example, a valid scitype string
        valid scitype strings, with explanation, are in datatypes.SCITYPE_REGISTER
        default = inferred from mtype of obj
    return_lossy: bool, optional, default=False
        whether to return second argument
    return_metadata: bool, optional, default=False
        whether to return third argument

    Returns
    -------
    fixtures: dict with integer keys, elements being
        fixture - example for mtype `mtype`, scitype `as_scitype`
        if return_lossy=True, elements are pairs with fixture and
            lossy: bool - whether the example is a lossy representation
        if return_metadata=True, elements are triples with fixture, lossy, and
            metadata: dict - metadata dict that would be returned by check_is_mtype
    """
    # if as_scitype is None, infer from mtype
    if as_scitype is None:
        as_scitype = mtype_to_scitype(mtype)

    example_dict, example_dict_lossy, example_dict_metadata = generate_example_dicts()

    # retrieve all keys that match the query
    exkeys = example_dict.keys()
    keys = [k for k in exkeys if k[0] == mtype and k[1] == as_scitype]

    # retrieve all fixtures that match the key
    fixtures = dict()

    for k in keys:
        if return_lossy:
            fixtures[k[2]] = (example_dict.get(k).build(), example_dict_lossy.get(k))
        elif return_metadata:
            fixtures[k[2]] = (
                example_dict.get(k).build(),
                example_dict_lossy.get(k),
                example_dict_metadata.get((k[1], k[2])),
            )
        else:
            fixtures[k[2]] = example_dict.get(k).build()

    # deepcopy to avoid side effects
    return deepcopy(fixtures)

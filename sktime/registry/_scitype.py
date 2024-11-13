"""Utility to determine scitype of estimator, based on base class type."""

__author__ = ["fkiraly"]

from inspect import isclass

from sktime.registry._base_classes import get_base_class_register


def is_scitype(obj, scitypes):
    """Check if obj is of desired scitype.

    Parameters
    ----------
    obj : class or object, must be an skbase BaseObject descendant
    scitypes : str or iterable of str
        scitype(s) to check, each str must be a valid scitype string

    Returns
    -------
    is_scitype : bool
        True if obj tag ``object_type``, or inferred scitype via ``registry.scitype``,
        contains at least one of the scitype strings in ``scitypes``.
    """
    obj_scitypes = scitype(
        obj, force_single_scitype=False, coerce_to_list=True, raise_on_unknown=False
    )
    if isinstance(scitypes, str):
        scitypes = [scitypes]
    scitypes = set(scitypes)
    return len(scitypes.intersection(obj_scitypes)) > 0


def scitype(
    obj, force_single_scitype=True, coerce_to_list=False, raise_on_unknown=True
):
    """Determine scitype string of obj.

    Parameters
    ----------
    obj : class or object inheriting from sktime BaseObject
    force_single_scitype : bool, optional, default = True
        whether only a single scitype is returned
        if True, only the *first* scitype found will be returned
        order is determined by the order in BASE_CLASS_REGISTER
    coerce_to_list : bool, optional, default = False
        determines the return type: if True, returns a single str,
        if False, returns a list of str
    raise_on_unknown : bool, optional, default = True
        if True, raises an error if no scitype can be determined for obj
        if False, returns "object" scitype

    Returns
    -------
    scitype : str, or list of str of sktime scitype strings from BASE_CLASS_REGISTER
        str, sktime scitype string, if exactly one scitype can be determined for obj
        or force_single_scitype is True, and if coerce_to_list is False
        list of str, of scitype strings, if more than one scitype are determined,
        or if coerce_to_list is True
        obj has scitype if it inherits from class in same row of BASE_CLASS_REGISTER

    Raises
    ------
    TypeError if no scitype can be determined for obj
    """
    # if object has tag, return tag
    if hasattr(obj, "get_tag"):
        if isclass(obj):
            tag_type = obj.get_class_tag("object_type", None)
        else:
            tag_type = obj.get_tag("object_type", None, raise_error=False)
        if tag_type is not None:
            if not isinstance(tag_type, list):
                tag_type = [tag_type]
            if force_single_scitype and len(tag_type) > 1:
                tag_type = [tag_type[0]]
            if coerce_to_list:
                return tag_type
            else:
                return tag_type[0]

    BASE_CLASS_REGISTER = get_base_class_register()

    # if the tag is not present, determine scitype from legacy base class logic
    if isclass(obj):
        scitypes = [sci[0] for sci in BASE_CLASS_REGISTER if issubclass(obj, sci[1])]
    else:
        scitypes = [sci[0] for sci in BASE_CLASS_REGISTER if isinstance(obj, sci[1])]

    if len(scitypes) == 0:
        if raise_on_unknown:
            raise TypeError("Error, no scitype could be determined for obj")
        else:
            scitypes = ["object"]

    if len(scitypes) > 1 and "object" in scitypes:
        scitypes = list(set(scitypes).difference(["object"]))

    if len(scitypes) > 1 and "estimator" in scitypes:
        scitypes = list(set(scitypes).difference(["estimator"]))

    if force_single_scitype:
        scitypes = [scitypes[0]]

    if len(scitypes) == 1 and not coerce_to_list:
        return scitypes[0]

    return scitypes

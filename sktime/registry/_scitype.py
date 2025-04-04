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
    # this function uses three passes to determine scitype:
    #
    # 1st pass: check if object has skbase tag manager, then use that
    # 2nd pass: check if object is sklearn estimator, then use sklearn scitype
    # 3rd pass: check if object is subclass of any of the base classes in
    # the BASE_CLASS_REGISTER, then use that
    # if neither: if no scitype can be determined, return "object" scitype

    def handle_output_format(detected_scitype):
        """Handle the return format of the detected scitype."""
        if not isinstance(detected_scitype, list):
            return_scitype = [detected_scitype]
        if force_single_scitype and len(detected_scitype) > 1:
            return_scitype = [detected_scitype[0]]
        if not coerce_to_list:
            return_scitype = return_scitype[0]
        return return_scitype

    # 1st pass
    # if object has scitype tag, return tag
    if hasattr(obj, "get_tag"):
        if isclass(obj):
            detected_scitype = obj.get_class_tag("object_type", None)
        else:
            detected_scitype = obj.get_tag("object_type", None, raise_error=False)
        if detected_scitype is not None:
            return handle_output_format(detected_scitype)

    # 2nd pass
    # check if object is sklearn estimator
    # if it is, return sklearn scitype
    from sktime.utils.sklearn import is_sklearn_estimator, sklearn_scitype

    if is_sklearn_estimator(obj):
        detected_scitype = f"{sklearn_scitype(obj)}_tabular"
        return handle_output_format(detected_scitype)

    # 3rd pass
    # check in the base class register
    # if the tag is not present, determine scitype from legacy base class logic
    BASE_CLASS_REGISTER = get_base_class_register()

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

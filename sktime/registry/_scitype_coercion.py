"""Utility to coerce estimators to a desired scitype, e.g., transformer."""

__author__ = ["fkiraly"]

_coerce_register = dict()


def _coerce_transformer_tabular_to_transformer(obj):
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor

    return TabularToSeriesAdaptor(obj)


_coerce_register[("transformer_tabular", "transformer")] = (
    _coerce_transformer_tabular_to_transformer
)


def coerce_scitype(
    obj,
    to_scitype,
    from_scitype=None,
    clone_obj=True,
    raise_on_unknown=False,
    raise_on_mismatch=False,
    msg="Error in object scitype check.",
):
    """Coerce obj to a given scitype.

    This function is used to coerce an object to a given scitype.
    If no source scitype or no valid coercion function is found,
    the object is returned unmodified under default settings.

    The function can also be used to raise errors if the scitype of the object
    does not match the expected scitype, or if no scitype can be determined.

    Parameters
    ----------
    obj : class or object inheriting from sktime BaseObject
    to_scitype : str
        scitype to coerce the object to
    from_scitype : str, optional, default = None
        scitype of ``obj`` that is assumed for the coercion, before coercion.
        If ``raise_on_mismatch``, non-None value will cause an exception
        if the detected scitype does not match the expected scitype.
    clone_obj : bool, optional, default = True
        if True, a clone of ``obj`` is used inside coercion composites.
        if False, the original object is passed unmodified.
    raise_on_unknown : bool, optional, default = False
        if True, raises an error if no scitype can be determined for obj.
        if False, unidentified scitypes will be treated as "object" scitype.
    raise_on_mismatch : bool, optional, default = False
        if True, raises an error if the detected scitype does not match the
        expected scitype, see ``from_scitype``.
    msg : str, optional, default = "Error in object scitype check."
        Start of error message returned with the exception, if
        an error is raised due to ``raise_on_mismatch``.

    Returns
    -------
    coerced_obj : obj, coerced to the desired scitype
        if a coercion function is found in the register, the object ``obj`` is coerced,
        otherwise the object ``obj`` is returned unmodified.
        Coercions are obtained from the ``_coerce_register`` dictionary.

    Raises
    ------
    TypeError if ``raise_on_unknown``, and no scitype can be determined for obj
    TypeError if ``raise_on_mismatch``, and ``from_scitype`` is not None, and
        the detected scitype does not match the expected scitype ``from_scitype``.
    """
    from sktime.registry._scitype import scitype
    from sktime.utils.sklearn import is_sklearn_estimator, sklearn_scitype

    if from_scitype is None or raise_on_mismatch:
        if is_sklearn_estimator(obj):
            detected_scitype = f"{sklearn_scitype(obj)}_tabular"
        else:
            detected_scitype = scitype(
                obj, force_single_scitype=True, raise_on_unknown=raise_on_unknown
            )
        if raise_on_mismatch and detected_scitype != from_scitype:
            raise TypeError(
                f"{msg} Expected object scitype {from_scitype}, "
                f"but found {detected_scitype}."
            )
        else:
            from_scitype = detected_scitype

    if clone_obj:
        if is_sklearn_estimator(obj) or not hasattr(obj, "clone"):
            from sklearn.base import clone

            obj = clone(obj)
        else:
            obj = obj.clone()

    if (from_scitype, to_scitype) not in _coerce_register:
        return obj

    # now we know that we have a coercion function in the register
    coerce_func = _coerce_register[(from_scitype, to_scitype)]

    coerced_obj = coerce_func(obj)
    return coerced_obj

"""Utility to coerce estimators to a desired scitype, e.g., transformer."""

__author__ = ["fkiraly"]

_coerce_register = dict()


def _coerce_transformer_tabular_to_transformer(obj):
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor

    return TabularToSeriesAdaptor(obj)


_coerce_register[("transformer_tabular", "transformer")] = (
    _coerce_transformer_tabular_to_transformer
)


def _coerce_series_annotator_to_transformer(obj):
    from sktime.detection.compose._as_transform import DetectorAsTransformer

    return DetectorAsTransformer(obj)


# todo 1.0.0 - remove series-annotator
_coerce_register[("series-annotator", "transformer")] = (
    _coerce_series_annotator_to_transformer
)


_coerce_register[("detector", "transformer")] = _coerce_series_annotator_to_transformer


def _coerce_clusterer_to_transformer(obj):
    from sktime.clustering.compose import ClustererAsTransformer

    return ClustererAsTransformer(obj)


_coerce_register[("clusterer", "transformer")] = _coerce_clusterer_to_transformer


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
        object to check or coerce

    to_scitype : str
        scitype to coerce the object ``obj`` to

    from_scitype : str or list of str, optional, default = None
        scitype of ``obj`` that is assumed for the coercion, before coercion.

        * If None, no scitype for ``obj`` is assumed.
        * if str, the scitype of ``obj`` is assumed to be this scitype.
        * if list of str, the scitype of ``obj`` is assumed to be one of these scitypes.

        If ``raise_on_mismatch``, non-None value will cause an exception
        if the detected scitype does not match the expected scitype(s).

    clone_obj : bool, optional, default = True
        if True, a clone of ``obj`` is used inside coercion composites.
        if False, the original object is passed unmodified.
    raise_on_unknown : bool, optional, default = False
        if True, raises an error if no scitype can be determined for obj.
        if False, unidentified scitypes will be treated as "object" scitype.
    raise_on_mismatch : bool, optional, default = False
        if True, raises an error if the detected scitype does not match the
        expected scitype, see ``from_scitype``.
        If False, the object is returned unmodified.
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
    from sktime.base._base import _safe_clone
    from sktime.registry._scitype import scitype
    from sktime.utils.sklearn import is_sklearn_estimator, sklearn_scitype

    if isinstance(from_scitype, str):
        from_scitype = [from_scitype]
    if isinstance(from_scitype, list) and len(from_scitype) == 0:
        from_scitype = None
    # from_scitype is now a list or None

    # we need to detect the scitype if it is not provided as an assumption
    need_detect = from_scitype is None or len(from_scitype) >= 2 or raise_on_mismatch
    if need_detect:
        if is_sklearn_estimator(obj):
            detected_scitype = f"{sklearn_scitype(obj)}_tabular"
        else:
            detected_scitype = scitype(
                obj, force_single_scitype=True, raise_on_unknown=raise_on_unknown
            )

    # case 1: detected scitype is not assumed scitype
    if from_scitype is not None and detected_scitype not in from_scitype:
        # if raise_on_mismatch, raise an error
        if raise_on_mismatch:
            raise TypeError(
                f"{msg} Expected object scitype {from_scitype}, "
                f"but found {detected_scitype}."
            )
        # otherwise, return the object unmodified
        else:
            return obj

    if from_scitype is None or len(from_scitype) >= 2:
        from_scitype = detected_scitype

    if clone_obj:
        obj = _safe_clone(obj)

    if (from_scitype, to_scitype) not in _coerce_register:
        return obj

    if isinstance(from_scitype, list) and detected_scitype not in from_scitype:
        return obj

    # now we know that we have a coercion function in the register
    coerce_func = _coerce_register[(from_scitype, to_scitype)]

    coerced_obj = coerce_func(obj)
    return coerced_obj

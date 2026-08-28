"""Version bridge for tags."""

# scikit-learn tag names used by ``get_sklearn_tag``, mapped to the sktime tag
# carrying the same information. Used for sktime estimators, which do not
# implement the sklearn tag interface, see ``get_sklearn_tag``.
SKLEARN_TO_SKTIME_TAG = {
    "capability:multioutput": "capability:multioutput",
    "capability:categorical": "capability:categorical_in_X",
    "fit_is_empty": "fit_is_empty",
}


def get_sklearn_tag(estimator, tagname):
    """Get the value of a scikit-learn tag.

    Parameters
    ----------
    estimator : sklearn estimator object
        The estimator from which to retrieve the tag value.

    tagname : str
        Name of the tag to retrieve.
        Supported tags:

        ``capability:multioutput : bool``
            Whether the estimator supports multi-output data.

        ``capability:categorical : bool``
            Whether the estimator can handle categorical data.

        ``fit_is_empty : bool``
            Whether the estimator's fit method does not require any data.

    Returns
    -------
    value : object
        Value of the specified tag.
    """
    from skbase.base import BaseObject
    from skbase.utils.dependencies import _check_soft_dependencies

    # sktime objects do not inherit from ``sklearn.base.BaseEstimator``, so they do
    # not implement ``__sklearn_tags__``. Up to scikit-learn 1.8, ``get_tags`` fell
    # back to defaults for such objects; from 1.9 on it raises ``AttributeError``.
    # sktime carries the same information in its own tags, so read those directly.
    if isinstance(estimator, BaseObject):
        sktime_tagname = SKLEARN_TO_SKTIME_TAG.get(tagname)
        if sktime_tagname is not None:
            return estimator.get_tag(sktime_tagname, False, raise_error=False)

    if tagname == "capability:multioutput":
        if _check_soft_dependencies("scikit-learn<1.6", severity="none"):
            return estimator._get_tags().get("multioutput", False)
        else:
            from sklearn.utils import get_tags

            return get_tags(estimator).target_tags.multi_output

    elif tagname == "capability:categorical":
        if _check_soft_dependencies("scikit-learn<1.6", severity="none"):
            if hasattr(estimator, "_get_tags"):
                categorical_list = ["categorical", "1dlabels", "2dlabels"]
                tag_values = estimator._get_tags()["X_types"]
                return any(val in tag_values for val in categorical_list)
        else:
            from sklearn.utils import get_tags

            cat1 = get_tags(estimator).input_tags.categorical
            cat2 = get_tags(estimator).target_tags.one_d_labels
            cat3 = get_tags(estimator).target_tags.two_d_labels
            return cat1 or cat2 or cat3
        return False

    elif tagname == "fit_is_empty":
        if _check_soft_dependencies("scikit-learn>=1.6", severity="none"):
            from sklearn.utils import get_tags

            return not get_tags(estimator).requires_fit
        else:
            if hasattr(estimator, "_get_tags"):
                return estimator._get_tags()["stateless"]
            else:
                return False

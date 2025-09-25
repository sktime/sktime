"""Version bridge for tags."""


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
    from sktime.utils.dependencies import _check_soft_dependencies

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

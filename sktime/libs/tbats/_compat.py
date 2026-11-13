import sklearn.utils.validation as _val

from skbase.utils.dependencies import _check_soft_dependencies


def check_array(array, **kwargs):
    """Compatibility wrapper for sklearn.utils.validation.check_array.

    Handles rename of force_all_finite to ensure_all_finite in scikit-learn 1.6+.
    """
    if "force_all_finite" in kwargs:
        val = kwargs.pop("force_all_finite")
        if _check_soft_dependencies("scikit-learn>=1.6", severity="none"):
            return _val.check_array(array, ensure_all_finite=val, **kwargs)
        else:
            return _val.check_array(array, force_all_finite=val, **kwargs)
    return _val.check_array(array, **kwargs)

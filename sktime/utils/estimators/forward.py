"""Utilities for attribute forwarding of estimators."""

__author__ = ["fkiraly"]


def _clone_fitted_params(to_obj, from_obj, overwrite=False):
    """Clone fitted parameters from one estimator to another.

    Takes all fitted parameters from ``from_obj`` and sets them on ``to_obj``.
    Fitted parameters are attributes of the estimator that end in "_".

    Mutable or pointer objects are not cloned, only the reference is copied.

    Parameters
    ----------
    to_obj : any object
        estimator to clone fitted parameters to
    from_obj : any object
        estimator to clone fitted parameters from
    overwrite : bool, optional, default = False
        whether to overwrite existing attributes in ``to_obj``

    Returns
    -------
    to_obj : reference to to_obj, with parameters set
    """
    fitted_params = _get_fitted_params_safe(from_obj)
    for key, value in fitted_params.items():
        if overwrite or not hasattr(to_obj, key):
            setattr(to_obj, f"{key}_", value)
    return to_obj


def _get_fitted_params_safe(obj):
    """Obtain fitted params of object, per sklearn convention.

    Fitted parameters are attributes of the estimator that end in "_"
    and do not start with "_". This is an ``sklearn`` and ``sktime`` API convention.

    Mutable or pointer objects are not cloned, the references are returned.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    fitted_params : dict with str keys
        fitted parameter values, keyed by names of fitted parameter.
        keys are the names of the attributes, without the trailing "_" character.
        values are the values of the attributes, with the trailing "_" character,
        in ``obj``.
    """
    # default retrieves all self attributes ending in "_"
    # and returns them with keys that have the "_" removed
    #
    # get all attributes ending in "_", exclude any that start with "_" (private)
    fitted_params = [
        attr for attr in dir(obj) if attr.endswith("_") and not attr.startswith("_")
    ]

    def hasattr_safe(obj, attr):
        try:
            if hasattr(obj, attr):
                getattr(obj, attr)
                return True
        except Exception:
            return False

    # remove the "_" at the end
    fitted_param_dict = {
        p[:-1]: getattr(obj, p) for p in fitted_params if hasattr_safe(obj, p)
    }

    return fitted_param_dict

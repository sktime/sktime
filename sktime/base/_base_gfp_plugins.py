"""A Plugin module for the base class to retrieve fitted params.

Interface specfication for plugins

---

Plugins are functions that are internally called whenver `get_fitted_params()`
method is called on a fittable object, i.e. `BaseEstimator` instances. These are
NOT included in the public API.

The standard conventions for defining new plugins are as follows:
1. The name of extending plugin follows the naming pattern `_gfp_*_plugin`
    where * is a descriptive name from which fitted parameters are taken from.
    Examples: `_gfp_nested_skbase_plugin` retrives fitted params from nested
    skbase components.
2. Generally, a plugin contains a single parameter `obj` which is an instance
    of a fitted object. It may contain additional parameters, provided
    they are available during the call to `get_fitted_params()` or any other plugin.
3. Include the plugin inside `BaseEstimator.get_fitted_params()` method in `./_base.py`
4. Add tests for the new plugin inside `./tests/test_base_sktime.py`.
"""

__author__ = ["achieveordie", "fkiraly"]
__all__ = [
    "_gfp_default",
    "_gfp_non_nested_plugin",
    "_gfp_nested_skbase_plugin",
    "_gfp_nested_sklearn_plugin",
    "_gfp_sklearn_pipeline_plugin",
]

from sklearn.base import BaseEstimator as _BaseEstimator

from sktime.base import BaseEstimator


def _gfp_default(obj, pname=""):
    """Obtain fitted params of object, per sklearn convention.

    Extracts a dict with {paramstr : paramvalue} contents,
    where paramstr are all string names of "fitted parameters".

    A "fitted attribute" of obj is one that ends in "_" but does not start with "_".
    "fitted parameters" are names of fitted attributes, minus the "_" at the end.

    Parameters
    ----------
    obj : any object, optional, default=self
    pname: str, default=''
        The name of the parent component of `obj`, used to append as
        "name__{component_name}". If parent name is not required then
        key will only be "{component_name}".

    Returns
    -------
    fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
    """
    # append '__' in case parent name is present
    pname = f"{pname}__" if pname != "" else ""

    # default retrieves all self attributes ending in "_"
    # and returns them with keys that have the "_" removed
    fitted_params = {
        f"{pname}{attr[:-1]}": getattr(obj, attr)
        for attr in dir(obj)
        if attr.endswith("_") and not attr.startswith("_") and hasattr(obj, attr)
    }
    return fitted_params


def _gfp_non_nested_plugin(obj, pname=""):
    """Plugin to get fitted non-nested parameters.

    Parameters
    ----------
    obj: any fittable object
    pname: str, default=''
        The name of the parent component of obj, if present.

    Returns
    -------
    fitted_params : dict with str keys
        fitted parameters, keyed by names of fitted parameter
    """
    return _gfp_default(obj=obj, pname=pname)


def _gfp_nested_skbase_plugin(obj):
    """Plugin to get fitted nested skbase parameters.

    Parameters
    ----------
    obj: any fittable object

    Returns
    -------
    nested_params: dict with str keys
        fitted parameters for skbase components
    """
    nested_params = {}

    # Add all skbase objects
    c_dict = obj._components()
    for c, comp in c_dict.items():
        if isinstance(comp, BaseEstimator) and comp._is_fitted:
            c_f_params = comp.get_fitted_params()
            c = c.rstrip("_")
            c_f_params = {f"{c}__{k}": v for k, v in c_f_params.items()}
            nested_params.update(c_f_params)

    return nested_params


def _gfp_nested_sklearn_plugin(obj):
    """Plugin to get fitted nested sklearn parameters.

    This doesn't include `sklearn.Pipeline` which is handled separtely in
    `_gfp_sklearn_pipeline_plugin()`.

    Parameters
    ----------
    obj: any fittable object

    Returns
    -------
    fitted_params: dict with str keys
        fitted parameters for sklearn components
    """
    fitted_params = _gfp_non_nested_plugin(obj=obj)

    # Add all nested parameters from components that are sklearn estimators
    # this is to be done recursively as we have to reach into nested sklearn
    # estimators
    n_new_params = 42
    old_new_params = fitted_params
    while n_new_params > 0:
        new_params = dict()
        for c, comp in old_new_params.items():
            if isinstance(comp, _BaseEstimator):
                c_f_params = obj._get_fitted_params_default(comp)
                c = c.rstrip("_")
                c_f_params = {f"{c}__{k}": v for k, v in c_f_params.items()}
                new_params.update(c_f_params)
        fitted_params.update(new_params)
        old_new_params = new_params.copy()
        n_new_params = len(new_params)

    return fitted_params


def _gfp_sklearn_pipeline_plugin(obj):
    """Plugin to get fitted nested `sklearn.Pipeline` parameters.

    Parameters
    ----------
    obj: any fittable object

    Returns
    -------
    fitted_params: dict with str keys
        fitted parameters for sklearn Pipeline & its components
    """
    from sklearn.pipeline import Pipeline as skPipeline

    def _get_params_for_sklearn_pipeline(component, name):
        """Get fitted params for an `sklearn.Pipeline` instance.

        This function can recurse if any child component is also an
        `sklearn.Pipeline` instance.

        Parameters
        ----------
        component: a `sklearn.Pipeline` instance
        name: str
            The name of the component. In case the Pipeline has no parent,
            then name is '' to avoid repetition of parent name for child components

        Returns
        -------
        _fitted_params: dict
            fitted parameters, keyed by names of fitted parameter
        """
        _fitted_params = {}
        for step_name, step in component.named_steps.items():
            pname = f"{name}__{step_name}" if name != "" else step_name
            if isinstance(step, skPipeline):
                _fitted_params.update(_get_params_for_sklearn_pipeline(step, pname))
            if isinstance(step, _BaseEstimator):
                # `sklearn.Pipeline` is a child of `_BaseEstimator`, so base
                # params needs to be pulled separately.
                _fitted_params.update(_gfp_non_nested_plugin(obj=step, pname=pname))

        return _fitted_params

    _fitted_params = _gfp_non_nested_plugin(obj=obj)
    fitted_params = {}

    for _, entity in _fitted_params.items():
        if isinstance(entity, skPipeline):
            fitted_params.update(_get_params_for_sklearn_pipeline(entity, name=""))

    return fitted_params

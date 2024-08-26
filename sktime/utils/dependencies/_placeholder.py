"""Decorator to create placeholder records in sktime.

Placeholder records are used for lookup and documentation generation,
and can be used to point to direct imports from external packages,
e.g., estimators with sktime compatible interfaces from 2nd and 3rd party packages.
"""


def placeholder(cls, module_name, obj_name=None, dependencies=None, condition=True):
    """Replace placeholder cls with imported object if installed, otherwise return it.

    Ensures that decorated object is replaced with
    the object of name ``obj_name``, imported from ``module_name``, if possible.

    Replacement is done if all of the following conditions are met:

    * object ``obj_name`` exists at module ``module_name``.
      If no ``obj_name`` is provided, the name of ``cls`` is used.
    * if ``condition`` is True (default)
    * if all dependencies of ``cls`` - i.e., the placeholder record - are satisfied
    * if all dependencies in ``dependencies`` are satisfied, if any are provided,
      these are checked via ``_check_soft_dependencies``.

    Parameters
    ----------
    cls : class
        Placeholder record to be replaced.
    module_name : str
        Name of the module to import from.
    obj_name : str, optional, default = name of cls
        Name of the object to import.
        If None, the name of ``cls`` is used.
    dependencies : optional, default=None
        Dependencies to check. Passed as argument to ``_check_soft_dependencies``.
    condition : bool, optional, default=True
        Condition to check before loading the object.
    """
    from sktime.utils.dependencies import _check_estimator_deps

    if obj_name is None:
        obj_name = cls.__name__

    load_condition = condition

    deps_satisfied = _check_estimator_deps(cls, severity="none")

    if dependencies is not None:
        from sktime.utils.dependencies import _check_soft_dependencies

        added_deps_satisfied = _check_soft_dependencies(dependencies, severity="none")
        deps_satisfied = deps_satisfied and added_deps_satisfied

    load_condition = load_condition and deps_satisfied

    if not load_condition:
        return cls

    try:
        # parse import_str to get the module and class name
        module = __import__(module_name, fromlist=[obj_name])
        imported_cls = getattr(module, obj_name)

        return imported_cls
    except Exception:  # noqa: S110
        pass

    # on failure, we also return the placeholder record itself
    return cls

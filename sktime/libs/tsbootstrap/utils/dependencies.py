"""Utility to check soft dependency imports, and raise warnings or errors."""

__author__ = ["fkiraly"]

from skbase.utils.dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)


def _check_estimator_deps(obj, msg=None, severity="error"):
    """Check if object/estimator's package & python requirements are met by python env.

    Convenience wrapper around `_check_python_version` and `_check_soft_dependencies`,
    checking against estimator tags `"python_version"`, `"python_dependencies"`.

    Checks whether dependency requirements of `BaseObject`-s in `obj`
    are satisfied by the current python environment.

    Parameters
    ----------
    obj : `BaseObject` descendant, instance or class, or list/tuple thereof
        object(s) that this function checks compatibility of, with the python env
    msg : str, optional, default = default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default
    severity : str, "error" (default), "warning", or "none"
        behaviour for raising errors or warnings
        "error" - raises a `ModuleNotFoundError` if environment is incompatible
        "warning" - raises a warning if environment is incompatible
            function returns False if environment is incompatible, otherwise True
        "none" - does not raise exception or warning
            function returns False if environment is incompatible, otherwise True

    Returns
    -------
    compatible : bool, whether `obj` is compatible with python environment
        False is returned only if no exception is raised by the function
        checks for python version using the python_version tag of obj
        checks for soft dependencies present using the python_dependencies tag of obj
        if `obj` contains multiple `BaseObject`-s, checks whether all are compatible

    Raises
    ------
    ModuleNotFoundError
        User friendly error if obj has python_version tag that is
        incompatible with the system python version.
        Compatible python versions are determined by the "python_version" tag of obj.
        User friendly error if obj has package dependencies that are not satisfied.
        Packages are determined based on the "python_dependencies" tag of obj.
    """
    compatible = True

    # if list or tuple, recurse & iterate over element, and return conjunction
    if isinstance(obj, (list, tuple)):  # noqa: UP038
        for x in obj:
            x_chk = _check_estimator_deps(x, msg=msg, severity=severity)
            compatible = compatible and x_chk
        return compatible

    compatible = compatible and _check_python_version(obj, severity=severity)

    pkg_deps = obj.get_class_tag("python_dependencies", None)
    pck_alias = obj.get_class_tag("python_dependencies_alias", None)
    if pkg_deps is not None and not isinstance(pkg_deps, list):
        pkg_deps = [pkg_deps]
    if pkg_deps is not None:
        pkg_deps_ok = _check_soft_dependencies(
            *pkg_deps,
            severity=severity,
            obj=obj,
            package_import_alias=pck_alias,
        )
        compatible = compatible and pkg_deps_ok

    return compatible

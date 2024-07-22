"""Utility to check soft dependency imports, and raise warnings or errors."""

__author__ = ["fkiraly", "mloning"]

import sys
import warnings
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from inspect import isclass

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version


# todo 0.32.0: remove suppress_import_stdout argument
def _check_soft_dependencies(
    *packages,
    package_import_alias=None,
    severity="error",
    obj=None,
    msg=None,
    suppress_import_stdout="deprecated",
):
    """Check if required soft dependencies are installed and raise error or warning.

    Parameters
    ----------
    packages : str or list/tuple of str, or length-1-tuple containing list/tuple of str
        str should be package names and/or package version specifications to check.
        Each str must be a PEP 440 compatible specifier string, for a single package.
        For instance, the PEP 440 compatible package name such as "pandas";
        or a package requirement specifier string such as "pandas>1.2.3".
        arg can be str, kwargs tuple, or tuple/list of str, following calls are valid:
        `_check_soft_dependencies("package1")`
        `_check_soft_dependencies("package1", "package2")`
        `_check_soft_dependencies(("package1", "package2"))`
        `_check_soft_dependencies(["package1", "package2"])`
    package_import_alias : dict with str keys and values, optional, default=empty
        key-value pairs are package name, import name
        import name is str used in python import, i.e., from import_name import ...
        should be provided if import name differs from package name
    severity : str, "error" (default), "warning", "none"
        behaviour for raising errors or warnings
        "error" - raises a `ModuleNotFoundError` if one of packages is not installed
        "warning" - raises a warning if one of packages is not installed
            function returns False if one of packages is not installed, otherwise True
        "none" - does not raise exception or warning
            function returns False if one of packages is not installed, otherwise True
    obj : python class, object, str, or None, default=None
        if self is passed here when _check_soft_dependencies is called within __init__,
        or a class is passed when it is called at the start of a single-class module,
        the error message is more informative and will refer to the class/object;
        if str is passed, will be used as name of the class/object or module
    msg : str, or None, default=None
        if str, will override the error message or warning shown with msg

    Raises
    ------
    ModuleNotFoundError
        error with informative message, asking to install required soft dependencies

    Returns
    -------
    boolean - whether all packages are installed, only if no exception is raised
    """
    # todo 0.32.0: remove this warning
    if suppress_import_stdout != "deprecated":
        warnings.warn(
            "In sktime _check_soft_dependencies, the suppress_import_stdout argument "
            "is deprecated and no longer has any effect. "
            "The argument will be removed in version 0.32.0, so users of the "
            "_check_soft_dependencies utility should not pass this argument anymore. "
            "The _check_soft_dependencies utility also no longer causes imports, "
            "hence no stdout "
            "output is created from imports, for any setting of the "
            "suppress_import_stdout argument. If you wish to import packages "
            "and make use of stdout prints, import the package directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if len(packages) == 1 and isinstance(packages[0], (tuple, list)):
        packages = packages[0]
    if not all(isinstance(x, str) for x in packages):
        raise TypeError(
            "packages argument of _check_soft_dependencies must be str or tuple of "
            f"str, but found packages argument of type {type(packages)}"
        )

    if package_import_alias is None:
        package_import_alias = {}
    msg_pkg_import_alias = (
        "package_import_alias argument of _check_soft_dependencies must "
        "be a dict with str keys and values, but found "
        f"package_import_alias of type {type(package_import_alias)}"
    )
    if not isinstance(package_import_alias, dict):
        raise TypeError(msg_pkg_import_alias)
    if not all(isinstance(x, str) for x in package_import_alias.keys()):
        raise TypeError(msg_pkg_import_alias)
    if not all(isinstance(x, str) for x in package_import_alias.values()):
        raise TypeError(msg_pkg_import_alias)

    if obj is None:
        class_name = "This functionality"
    elif not isclass(obj):
        class_name = type(obj).__name__
    elif isclass(obj):
        class_name = obj.__name__
    elif isinstance(obj, str):
        class_name = obj
    else:
        raise TypeError(
            "obj argument of _check_soft_dependencies must be a class, an object,"
            " a str, or None, but found obj of type"
            f" {type(obj)}"
        )

    if msg is not None and not isinstance(msg, str):
        raise TypeError(
            "msg argument of _check_soft_dependencies must be a str, "
            f"or None, but found msg of type {type(msg)}"
        )

    for package in packages:
        try:
            req = Requirement(package)
            req = _normalize_requirement(req)
        except InvalidRequirement:
            msg_version = (
                f"wrong format for package requirement string "
                f"passed via packages argument of _check_soft_dependencies, "
                f'must be PEP 440 compatible requirement string, e.g., "pandas"'
                f' or "pandas>1.1", but found "{package}"'
            )
            raise InvalidRequirement(msg_version)

        package_name = req.name
        package_version_req = req.specifier

        # determine the package import
        if package_name in package_import_alias.keys():
            package_import_name = package_import_alias[package_name]
        else:
            package_import_name = package_name

        # determine the package import
        if package_name in package_import_alias.keys():
            package_import_name = package_import_alias[package_name]
        else:
            package_import_name = package_name

        pkg_env_version = _get_pkg_version(package_name, package_import_name)

        # if package not present, make the user aware of installation reqs
        if pkg_env_version is None:
            if obj is None and msg is None:
                msg = (
                    f"'{package}' not found. "
                    f"'{package}' is a soft dependency and not included in the "
                    f"base sktime installation. Please run: `pip install {package}` to "
                    f"install the {package} package. "
                    f"To install all soft dependencies, run: `pip install "
                    f"sktime[all_extras]`"
                )
            elif msg is None:  # obj is not None, msg is None
                msg = (
                    f"{class_name} requires package '{package}' to be present "
                    f"in the python environment, but '{package}' was not found. "
                    f"'{package}' is a soft dependency and not included in the base "
                    f"sktime installation. Please run: `pip install {package}` to "
                    f"install the {package} package. "
                    f"To install all soft dependencies, run: `pip install "
                    f"sktime[all_extras]`"
                )
            # if msg is not None, none of the above is executed,
            # so if msg is passed it overrides the default messages

            _raise_at_severity(msg, severity, caller="_check_soft_dependencies")
            return False

        # now we check compatibility with the version specifier if non-empty
        if package_version_req != SpecifierSet(""):
            msg = (
                f"{class_name} requires package '{package}' to be present "
                f"in the python environment, with version {package_version_req}, "
                f"but incompatible version {pkg_env_version} was found. "
            )
            if obj is not None:
                msg = msg + (
                    f"This version requirement is not one by sktime, but specific "
                    f"to the module, class or object with name {obj}."
                )

            # raise error/warning or return False if version is incompatible
            if pkg_env_version not in package_version_req:
                _raise_at_severity(msg, severity, caller="_check_soft_dependencies")
                return False

    # if package can be imported and no version issue was caught for any string,
    # then obj is compatible with the requirements and we should return True
    return True


def _check_dl_dependencies(msg=None, severity="error"):
    """Check if deep learning dependencies are installed.

    Parameters
    ----------
    msg : str, optional, default= default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default
    severity : str, "error" (default), "warning", "none"
        behaviour for raising errors or warnings
        "error" - raises a ModuleNotFoundError if one of packages is not installed
        "warning" - raises a warning if one of packages is not installed
            function returns False if one of packages is not installed, otherwise True
        "none" - does not raise exception or warning
            function returns False if one of packages is not installed, otherwise True

    Raises
    ------
    ModuleNotFoundError
        User friendly error with suggested action to install deep learning dependencies

    Returns
    -------
    boolean - whether all packages are installed, only if no exception is raised
    """
    if not isinstance(msg, str):
        msg = (
            "tensorflow is required for deep learning functionality in `sktime`. "
            "To install these dependencies, run: `pip install sktime[dl]`"
        )
    if find_spec("tensorflow") is not None:
        return True
    else:
        _raise_at_severity(msg, severity, caller="_check_dl_dependencies")
        return False


def _check_mlflow_dependencies(msg=None, severity="error"):
    """Check if `mlflow` and its dependencies are installed.

    Parameters
    ----------
    msg: str, optional, default= default message (msg below)
        error message to be returned when `ModuleNotFoundError` is raised.
    severity: str, either of "error", "warning" or "none"
        behaviour for raising errors or warnings
        "error" - raises a `ModuleNotFound` if mlflow-related packages are not found.
        "warning" - raises a warning message if any mlflow-related package is not
            installed also returns False. In case all packages are present,
            returns True.
        "none" - does not raise any exception or warning and simply returns True
            if all packages are installed otherwise return False.

    Raise
    -----
    ModuleNotFoundError
        User Friendly error with a suggested action to install mlflow dependencies

    Returns
    -------
    boolean - whether all mlflow-related packages are installed.
    """
    if not isinstance(msg, str):
        msg = (
            "`mlflow` is an extra dependency and is not included "
            "in the base sktime installation. "
            "Please run `pip install mlflow` "
            "or `pip install sktime[mlflow]` to install the package."
        )

    return _check_soft_dependencies("mlflow", msg=msg, severity=severity)


@lru_cache
def _get_pkg_version(package_name, package_import_name=None):
    """Check whether package is available in environment, and return its version if yes.

    Returns ``Version`` object from ``lru_cache``, this should not be mutated.

    Parameters
    ----------
    package_name : str, optional, default=None
        name of package to check, e.g., "pandas" or "sklearn".
        This is the pypi package name, not the import name, e.g.,
        ``scikit-learn``, not ``sklearn``.
    package_import_name : str, optional, default=None
        name of package to check for import, e.g., "pandas" or "sklearn".
        Note: this is the import name, not the pypi package name, e.g.,
        ``sklearn``, not ``scikit-learn``.
        If not given, ``package_name`` is used as ``package_import_name``,
        i.e., it is assumed that the import name is the same as the package name.

    Returns
    -------
    None, if package is not found at import ``package_import_name``;
    ``importlib`` ``Version`` of package, if found at import ``package_import_name``
    """
    if package_import_name is None:
        package_import_name = package_name

    # optimized branching to check presence of import
    # and presence of package distribution
    # first we check import, then we check distribution
    # because try/except consumes more runtime
    pkg_spec = find_spec(package_import_name)
    if pkg_spec is not None:
        try:
            pkg_env_version = Version(version(package_name))
        except (InvalidVersion, PackageNotFoundError):
            pkg_env_version = None
    else:
        pkg_env_version = None

    return pkg_env_version


def _check_python_version(obj, package=None, msg=None, severity="error"):
    """Check if system python version is compatible with requirements of obj.

    Parameters
    ----------
    obj : sktime estimator, BaseObject descendant
        used to check python version
    package : str, default = None
        if given, will be used in error message as package name
    msg : str, optional, default = default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default
    severity : str, "error" (default), "warning", or "none"
        whether the check should raise an error, a warning, or nothing

    Returns
    -------
    compatible : bool, whether obj is compatible with system python version
        check is using the python_version tag of obj

    Raises
    ------
    ModuleNotFoundError
        User friendly error if obj has python_version tag that is
        incompatible with the system python version. If package is given,
        error message gives package as the reason for incompatibility.
    """
    est_specifier_tag = obj.get_class_tag("python_version", tag_value_default="None")
    if est_specifier_tag in ["None", None]:
        return True

    try:
        est_specifier = SpecifierSet(est_specifier_tag)
    except InvalidSpecifier:
        msg_version = (
            f"wrong format for python_version tag, "
            f'must be PEP 440 compatible specifier string, e.g., "<3.9, >= 3.6.3",'
            f' but found "{est_specifier_tag}"'
        )
        raise InvalidSpecifier(msg_version)

    # python sys version, e.g., "3.8.12"
    sys_version = sys.version.split(" ")[0]

    if sys_version in est_specifier:
        return True
    # now we know that est_version is not compatible with sys_version

    if isclass(obj):
        class_name = obj.__name__
    else:
        class_name = type(obj).__name__

    if not isinstance(msg, str):
        msg = (
            f"{class_name} requires python version to be {est_specifier},"
            f" but system python version is {sys.version}."
        )

        if package is not None:
            msg += (
                f" This is due to python version requirements of the {package} package."
            )

    _raise_at_severity(msg, severity, caller="_check_python_version")
    return False


def _check_env_marker(obj, package=None, msg=None, severity="error"):
    """Check if packaging marker tag is with requirements of obj.

    Parameters
    ----------
    obj : sktime estimator, BaseObject descendant
        used to check python version
    package : str, default = None
        if given, will be used in error message as package name
    msg : str, optional, default = default message (msg below)
        error message to be returned in the `ModuleNotFoundError`, overrides default
    severity : str, "error" (default), "warning", or "none"
        whether the check should raise an error, a warning, or nothing

    Returns
    -------
    compatible : bool, whether obj is compatible with system python version
        check is using the python_version tag of obj

    Raises
    ------
    InvalidMarker
        User friendly error if obj has env_marker tag that is not a
        packaging compatible marker string
    ModuleNotFoundError
        User friendly error if obj has an env_marker tag that is
        incompatible with the python environment. If package is given,
        error message gives package as the reason for incompatibility.
    """
    est_marker_tag = obj.get_class_tag("env_marker", tag_value_default="None")
    if est_marker_tag in ["None", None]:
        return True

    try:
        est_marker = Marker(est_marker_tag)
    except InvalidMarker:
        msg_version = (
            f"wrong format for env_marker tag, "
            f"must be PEP 508 compatible specifier string, e.g., "
            f'platform_system!="windows", but found "{est_marker_tag}"'
        )
        raise InvalidMarker(msg_version)

    if est_marker.evaluate():
        return True
    # now we know that est_marker is not compatible with the environment

    if isclass(obj):
        class_name = obj.__name__
    else:
        class_name = type(obj).__name__

    if not isinstance(msg, str):
        msg = (
            f"{class_name} requires an environment to satisfy "
            f"packaging marker spec {est_marker}, but environment does not satisfy it."
        )

        if package is not None:
            msg += f" This is due to requirements of the {package} package."

    _raise_at_severity(msg, severity, caller="_check_env_marker")
    return False


def _check_estimator_deps(obj, msg=None, severity="error"):
    """Check if object/estimator's package & python requirements are met by python env.

    Convenience wrapper around `_check_python_version` and `_check_soft_dependencies`,
    checking against estimator tags `"python_version"`, `"python_dependencies"`.

    Checks whether dependency requirements of `BaseObject`-s in `obj`
    are satisfied by the current python environment.

    Parameters
    ----------
    obj : `sktime` object, `BaseObject` descendant, or list/tuple thereof
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
    if isinstance(obj, (list, tuple)):
        for x in obj:
            x_chk = _check_estimator_deps(x, msg=msg, severity=severity)
            compatible = compatible and x_chk
        return compatible

    compatible = compatible and _check_python_version(obj, severity=severity)
    compatible = compatible and _check_env_marker(obj, severity=severity)

    pkg_deps = obj.get_class_tag("python_dependencies", None)
    pck_alias = obj.get_class_tag("python_dependencies_alias", None)
    if pkg_deps is not None and not isinstance(pkg_deps, list):
        pkg_deps = [pkg_deps]
    if pkg_deps is not None:
        pkg_deps_ok = _check_soft_dependencies(
            *pkg_deps, severity=severity, obj=obj, package_import_alias=pck_alias
        )
        compatible = compatible and pkg_deps_ok

    return compatible


def _normalize_requirement(req):
    """Normalize packaging Requirement by removing build metadata from versions.

    Parameters
    ----------
    req : packaging.requirements.Requirement
        requirement string to normalize, e.g., Requirement("pandas>1.2.3+foobar")

    Returns
    -------
    normalized_req : packaging.requirements.Requirement
        normalized requirement object with build metadata removed from versions,
        e.g., Requirement("pandas>1.2.3")
    """
    # Process each specifier in the requirement
    normalized_specs = []
    for spec in req.specifier:
        # Parse the version and remove the build metadata
        spec_v = Version(spec.version)
        version_wo_build_metadata = f"{spec_v.major}.{spec_v.minor}.{spec_v.micro}"

        # Create a new specifier without the build metadata
        normalized_spec = Specifier(f"{spec.operator}{version_wo_build_metadata}")
        normalized_specs.append(normalized_spec)

    # Reconstruct the specifier set
    normalized_specifier_set = SpecifierSet(",".join(str(s) for s in normalized_specs))

    # Create a new Requirement object with the normalized specifiers
    normalized_req = Requirement(f"{req.name}{normalized_specifier_set}")

    return normalized_req


def _raise_at_severity(
    msg,
    severity,
    exception_type=None,
    warning_type=None,
    stacklevel=2,
    caller="_raise_at_severity",
):
    """Raise exception or warning or take no action, based on severity.

    Parameters
    ----------
    msg : str
        message to raise or warn
    severity : str, "error", "warning", or "none"
        behaviour for raising errors or warnings
    exception_type : Exception, default=ModuleNotFoundError
        exception type to raise if severity="severity"
    warning_type : warning, default=Warning
        warning type to raise if severity="warning"
    stacklevel : int, default=2
        stacklevel for warnings, if severity="warning"
    caller : str, default="_raise_at_severity"
        caller name, used in exception if severity not in ["error", "warning", "none"]

    Returns
    -------
    None

    Raises
    ------
    exception : exception_type, if severity="error"
    warning : warning+type, if severity="warning"
    ValueError : if severity not in ["error", "warning", "none"]
    """
    if exception_type is None:
        exception_type = ModuleNotFoundError

    if severity == "error":
        raise exception_type(msg)
    elif severity == "warning":
        warnings.warn(msg, category=warning_type, stacklevel=stacklevel)
    elif severity == "none":
        return None
    else:
        raise ValueError(
            f"Error in calling {caller}, severity "
            f'argument must be "error", "warning", or "none", found {severity!r}.'
        )
    return None

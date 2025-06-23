"""Utility to check soft dependency imports, and raise warnings or errors."""

import sys
import warnings
from functools import lru_cache
from importlib.util import find_spec
from inspect import isclass

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version


def _check_soft_dependencies(
    *packages,
    severity="error",
    obj=None,
    msg=None,
    normalize_reqs=True,
):
    """Check if required soft dependencies are installed and raise error or warning.

    Parameters
    ----------
    packages : str or list/tuple of str nested up to two levels
        str should be package names and/or package version specifications to check.
        Each str must be a PEP 440 compatible specifier string, for a single package.
        For instance, the PEP 440 compatible package name such as ``"pandas"``;
        or a package requirement specifier string such as ``"pandas>1.2.3"``.
        arg can be str, kwargs tuple, or tuple/list of str, following calls are valid:

        * ``_check_soft_dependencies("package1")``
        * ``_check_soft_dependencies("package1", "package2")``
        * ``_check_soft_dependencies(("package1", "package2"))``
        * ``_check_soft_dependencies(["package1", "package2"])``
        * ``_check_soft_dependencies(("package1", "package2"), "package3")``
        * ``_check_soft_dependencies(["package1", "package2"], "package3")``
        * ``_check_soft_dependencies((["package1", "package2"], "package3"))``

        The first level is interpreted as conjunction, the second level as disjunction,
        that is, conjunction = "and", disjunction = "or".

        In case of more than a single arg, an outer level of "and" (brackets)
        is added, that is,

        ``_check_soft_dependencies("package1", "package2")``

        is the same as ``_check_soft_dependencies(("package1", "package2"))``

    severity : str, "error" (default), "warning", "none"
        behaviour for raising errors or warnings

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
          function returns False if one of packages is not installed, otherwise True

    obj : python class, object, str, or None, default=None
        if self is passed here when _check_soft_dependencies is called within __init__,
        or a class is passed when it is called at the start of a single-class module,
        the error message is more informative and will refer to the class/object;
        if str is passed, will be used as name of the class/object or module

    msg : str, or None, default=None
        if str, will override the error message or warning shown with msg

    normalize_reqs : bool, default=True
        whether to normalize the requirement strings before checking them,
        by removing build metadata from versions.
        If set True, pre, post, and dev versions are removed from all version strings.

        Example if True:
        requirement "my_pkg==2.3.4.post1" will be normalized to "my_pkg==2.3.4";
        an actual version "my_pkg==2.3.4.post1" will be considered compatible with
        "my_pkg==2.3.4". If False, the this situation would raise an error.

    Raises
    ------
    InvalidRequirement
        if package requirement strings are not PEP 440 compatible
    ModuleNotFoundError
        error with informative message, asking to install required soft dependencies
    TypeError, ValueError
        on invalid arguments

    Returns
    -------
    boolean - whether all packages are installed, only if no exception is raised
    """
    if len(packages) == 1 and isinstance(packages[0], (tuple, list)):
        packages = packages[0]

    def _is_str_or_tuple_of_strs(obj):
        """Check that obj is a str or list/tuple nesting up to 1st level of str.

        Valid examples:

        * "pandas"
        * ("pandas", "scikit-learn")
        * ["pandas", "scikit-learn"]
        """
        if isinstance(obj, (tuple, list)):
            return all(isinstance(x, str) for x in obj)

        return isinstance(obj, str)

    if not all(_is_str_or_tuple_of_strs(x) for x in packages):
        raise TypeError(
            "packages argument of _check_soft_dependencies must be str or tuple/list "
            "of str or of tuple/list of str, "
            f"but found packages argument of type {type(packages)}"
        )

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

    def _get_pkg_version_and_req(package):
        """Get package version and requirement object from package string.

        Parameters
        ----------
        package : str

        Returns
        -------
        package_version_req: SpecifierSet
            version requirement object from package string
        package_name: str
            name of package, PEP 440 compatible specifier string, e.g., "scikit-learn"
        pkg_env_version: Version
            version object of package in python environment
        """
        try:
            req = Requirement(package)
            if normalize_reqs:
                req = _normalize_requirement(req)
        except InvalidRequirement:
            msg_version = (
                f"wrong format for package requirement string, "
                f"passed via packages argument of _check_soft_dependencies, "
                f'must be PEP 440 compatible requirement string, e.g., "pandas"'
                f' or "pandas>1.1", but found {package!r}'
            )
            raise InvalidRequirement(msg_version) from None

        package_name = req.name
        package_version_req = req.specifier

        pkg_env_version = _get_pkg_version(package_name)
        if normalize_reqs:
            pkg_env_version = _normalize_version(pkg_env_version)

        return package_version_req, package_name, pkg_env_version

    # each element of the list "package" must be satisfied
    for package_req in packages:
        # for elemehts, two cases can happen:
        #
        # 1. package is a string, e.g., "pandas". Then this must be present.
        # 2. package is a tuple or list, e.g., ("pandas", "scikit-learn").
        #    Then at least one of these must be present.
        if not isinstance(package_req, (tuple, list)):
            package_req = (package_req,)
        else:
            package_req = tuple(package_req)

        def _is_version_req_satisfied(pkg_env_version, pkg_version_req):
            if pkg_env_version is None:
                return False
            if pkg_version_req != SpecifierSet(""):
                return pkg_env_version in pkg_version_req
            else:
                return True

        pkg_version_reqs = []
        pkg_env_versions = []
        pkg_names = []
        nontrivital_bound = []
        req_sat = []

        for package in package_req:
            pkg_version_req, pkg_nm, pkg_env_version = _get_pkg_version_and_req(package)
            pkg_version_reqs.append(pkg_version_req)
            pkg_env_versions.append(pkg_env_version)
            pkg_names.append(pkg_nm)
            nontrivital_bound.append(pkg_version_req != SpecifierSet(""))
            req_sat.append(_is_version_req_satisfied(pkg_env_version, pkg_version_req))

        def _quote(x):
            return f"'{x}'"

        package_req_strs = [_quote(x) for x in package_req]
        # example: ["'scipy<1.7.0'"] or ["'scipy<1.7.0'", "'numpy'"]

        package_str_q = " or ".join(package_req_strs)
        # example: "'scipy<1.7.0'"" or "'scipy<1.7.0' or 'numpy'""

        package_str = " or ".join(f"`pip install {r}`" for r in package_req)
        # example: "pip install scipy<1.7.0 or pip install numpy"

        # if package not present, make the user aware of installation reqs
        if all(pkg_env_version is None for pkg_env_version in pkg_env_versions):
            if obj is None and msg is None:
                msg = (
                    f"{class_name} requires package {package_str_q} to be present "
                    f"in the python environment, but {package_str_q} was not found. "
                )
            elif msg is None:  # obj is not None, msg is None
                msg = (
                    f"{class_name} requires package {package_str_q} to be present "
                    f"in the python environment, but {package_str_q} was not found. "
                    f"{package_str_q} is a dependency of {class_name} and required "
                    f"to construct it. "
                )
            msg = msg + (
                f"To install the requirement {package_str_q}, please run: "
                f"{package_str} "
            )
            # if msg is not None, none of the above is executed,
            # so if msg is passed it overrides the default messages

            _raise_at_severity(msg, severity, caller="_check_soft_dependencies")
            return False

        # now we check compatibility with the version specifier if non-empty
        if not any(req_sat):
            zp = zip(package_req, pkg_names, pkg_env_versions, req_sat)
            reqs_not_satisfied = [x for x in zp if x[3] is False]
            actual_vers = [f"{x[1]} {x[2]}" for x in reqs_not_satisfied]
            pkg_env_version_str = ", ".join(actual_vers)

            msg = (
                f"{class_name} requires package {package_str_q} to be present "
                f"in the python environment, with versions as specified, "
                f"but incompatible version {pkg_env_version_str} was found. "
            )
            if obj is not None:
                msg = msg + (
                    f"This version requirement is not one by sktime, but specific "
                    f"to the module, class or object with name {obj}."
                )

            # raise error/warning or return False if version is incompatible

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
        error message to be returned in the ``ModuleNotFoundError``, overrides default

    severity : str, "error" (default), "warning", "none"
        whether the check should raise an error, a warning, or nothing

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
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
        error message to be returned when ``ModuleNotFoundError`` is raised.
    severity: str, either of "error", "warning" or "none"
        behaviour for raising errors or warnings
        "error" - raises a ``ModuleNotFound`` if mlflow-related packages are not found.
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

    # we allow mlflow and mlflow-skinny, at least one must be present
    MLFLOW_DEPS = [["mlflow", "mlflow-skinny"]]

    return _check_soft_dependencies(MLFLOW_DEPS, msg=msg, severity=severity)


@lru_cache
def _get_installed_packages_private():
    """Get a dictionary of installed packages and their versions.

    Same as _get_installed_packages, but internal to avoid mutating the lru_cache
    by accident.
    """
    from importlib.metadata import distributions, version

    dists = distributions()
    package_names = {dist.metadata["Name"] for dist in dists}
    package_versions = {pkg_name: version(pkg_name) for pkg_name in package_names}
    # developer note:
    # we cannot just use distributions naively,
    # because the same top level package name may appear *twice*,
    # e.g., in a situation where a virtual env overrides a base env,
    # such as in deployment environments like databricks.
    # the "version" contract ensures we always get the version that corresponds
    # to the importable distribution, i.e., the top one in the sys.path.
    return package_versions


def _get_installed_packages():
    """Get a dictionary of installed packages and their versions.

    Returns
    -------
    dict : dictionary of installed packages and their versions
        keys are PEP 440 compatible package names, values are package versions
        MAJOR.MINOR.PATCH version format is used for versions, e.g., "1.2.3"
    """
    return _get_installed_packages_private().copy()


def _get_pkg_version(package_name):
    """Check whether package is available in environment, and return its version if yes.

    Returns ``Version`` object from ``lru_cache``, this should not be mutated.

    Parameters
    ----------
    package_name : str, optional, default=None
        name of package to check,
        PEP 440 compatibe specifier string, e.g., "pandas" or "sklearn".
        This is the pypi package name, not the import name, e.g.,
        ``scikit-learn``, not ``sklearn``.

    Returns
    -------
    None, if package is not found in python environment.
    ``importlib`` ``Version`` of package, if present in environment.
    """
    pkgs = _get_installed_packages()
    pkg_vers_str = pkgs.get(package_name, None)
    if pkg_vers_str is None:
        return None
    try:
        pkg_env_version = Version(pkg_vers_str)
    except InvalidVersion:
        pkg_env_version = None
    return pkg_env_version


def _check_python_version(
    obj, package=None, msg=None, severity="error", prereleases=True
):
    """Check if system python version is compatible with requirements of obj.

    Parameters
    ----------
    obj : sktime estimator, BaseObject descendant
        used to check python version

    package : str, default = None
        if given, will be used in error message as package name

    msg : str, optional, default = default message (msg below)
        error message to be returned in the ``ModuleNotFoundError``, overrides default

    severity : str, "error" (default), "warning", "none"
        whether the check should raise an error, a warning, or nothing

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
          function returns False if one of packages is not installed, otherwise True

    prereleases: str, default = True
        Whether prerelease versions are considered compatible.
        If True, allows prerelease versions to be considered compatible.
        If False, always considers prerelease versions as incompatible, i.e., always
        raises error, warning, or returns False, if the system python version is a
        prerelease.

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
        est_specifier = SpecifierSet(est_specifier_tag, prereleases=prereleases)
    except InvalidSpecifier:
        msg_version = (
            f"wrong format for python_version tag, "
            f'must be PEP 440 compatible specifier string, e.g., "<3.9, >= 3.6.3",'
            f" but found {est_specifier_tag!r}"
        )
        raise InvalidSpecifier(msg_version) from None

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

        if "rc" in sys_version:
            msg += " This is due to the release candidate status of your system Python."

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
        error message to be returned in the ``ModuleNotFoundError``, overrides default

    severity : str, "error" (default), "warning", "none"
        whether the check should raise an error, a warning, or nothing

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
          function returns False if one of packages is not installed, otherwise True

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
            f'platform_system!="windows", but found {est_marker_tag!r}'
        )
        raise InvalidMarker(msg_version) from None

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
    obj : ``sktime`` object, ``BaseObject`` descendant, or list/tuple thereof
        object(s) that this function checks compatibility of, with the python env

    msg : str, optional, default = default message (msg below)
        error message to be returned in the ``ModuleNotFoundError``, overrides default

    severity : str, "error" (default), "warning", "none"
        whether the check should raise an error, a warning, or nothing

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
          function returns False if one of packages is not installed, otherwise True


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
    if pkg_deps is not None and not isinstance(pkg_deps, list):
        pkg_deps = [pkg_deps]
    if pkg_deps is not None:
        pkg_deps_ok = _check_soft_dependencies(pkg_deps, severity=severity, obj=obj)
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
        # Create a new specifier without the build metadata
        normalized_version = _normalize_version(spec.version)
        normalized_spec = Specifier(f"{spec.operator}{normalized_version}")
        normalized_specs.append(normalized_spec)

    # Reconstruct the specifier set
    normalized_specifier_set = SpecifierSet(",".join(str(s) for s in normalized_specs))

    # Create a new Requirement object with the normalized specifiers
    normalized_req = Requirement(f"{req.name}{normalized_specifier_set}")

    return normalized_req


def _normalize_version(version):
    """Normalize version string by removing build metadata.

    Parameters
    ----------
    version : packaging.version.Version
        version object to normalize, e.g., Version("1.2.3+foobar")

    Returns
    -------
    normalized_version : packaging.version.Version
        normalized version object with build metadata removed, e.g., Version("1.2.3")
    """
    if version is None:
        return None
    if not isinstance(version, Version):
        version_obj = Version(version)
    else:
        version_obj = version
    normalized_version = f"{version_obj.major}.{version_obj.minor}.{version_obj.micro}"
    return normalized_version


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

    severity : str, "error" (default), "warning", "none"
        whether the check should raise an error, a warning, or nothing

        * "error" - raises a ``ModuleNotFoundError`` if one of packages is not installed
        * "warning" - raises a warning if one of packages is not installed
          function returns False if one of packages is not installed, otherwise True
        * "none" - does not raise exception or warning
          function returns False if one of packages is not installed, otherwise True

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

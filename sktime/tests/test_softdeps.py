# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests that soft dependencies are handled correctly.

sktime supports a number of soft dependencies which are necessary for using a certain
module but otherwise not necessary.

Adapted from code of mloning for the legacy Azure CI/CD build tools.
"""

__author__ = ["mloning", "fkiraly"]

import pkgutil
import re
from importlib import import_module
from unittest.mock import patch

import pytest

from sktime.registry import all_estimators
from sktime.tests._config import EXCLUDE_ESTIMATORS
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.scenarios_getter import retrieve_scenarios
from sktime.utils.dependencies import _check_python_version, _check_soft_dependencies

# list of soft dependencies used
# excludes estimators, only for soft dependencies used in non-estimator modules
SOFT_DEPENDENCIES = {
    "sktime.benchmarking.evaluation": ["matplotlib"],
    "sktime.benchmarking.experiments": ["tsfresh", "esig"],
    "sktime.classification.deep_learning": ["tensorflow"],
    "sktime.regression.deep_learning": ["tensorflow"],
    "sktime.networks": ["tensorflow"],
    "sktime.clustering.evaluation._plot_clustering": ["matplotlib"],
    "sktime.utils.numba.general": ["numba"],
    "sktime.utils.numba.stats": ["numba"],
}

MODULES_TO_IGNORE = ("sktime._contrib", "sktime.utils._testing")

# estimators excepted from checking that they raise no import exception when constructed
# for the reason that they are composites which have soft dependencies in examples
# but no soft dependencies themselves, so it's temporarily fine to raise this
# e.g., forecasting pipeline with an ARIMA estimator
# todo: long-term all example parameter settings should be soft dependency free
# strings of class names to avoid the imports
EXCEPTED_FROM_NO_DEP_CHECK = []


def _is_test(module):
    module_parts = module.split(".")
    return any(part in ("tests", "test") for part in module_parts)


def _is_ignored(module):
    return any(module_to_ignore in module for module_to_ignore in MODULES_TO_IGNORE)


def _is_private(module):
    module_parts = module.split(".")
    return any(part.startswith("_") for part in module_parts)


def _extract_dependency_from_error_msg(msg):
    # We raise an user-friendly error if a soft dependency is missing in the
    # `check_soft_dependencies` function. In the error message, the missing
    # dependency is printed in single quotation marks, so we can use that here to
    # extract and return the dependency name.
    match = re.search(r"\'(.+?)\'", msg)
    if match:
        return match.group(1)
    else:
        raise ValueError("No dependency found in error msg.")


# collect all modules
modules = pkgutil.walk_packages(path=["./sktime/"], prefix="sktime.")
modules = [x[1] for x in modules]
modules = [x for x in modules if not (_is_test(x) or _is_ignored(x) or _is_private(x))]


def is_soft_dep_missing_message(msg):
    """Check whether message is one of multiple missing softdep messages."""
    # message if PEP 440 string is specified
    missing_version_msg = "to be present in the python environment, with version"
    cond1 = missing_version_msg in msg
    # message if dependency is missing entirely
    missing_dep_entirely_msg = "requires package"
    cond2 = missing_dep_entirely_msg in msg
    # special message for deep learning dependencies
    error_msg_dl = "required for deep learning"
    cond3 = error_msg_dl in msg
    # message if environment marker not satisfied
    error_msg_marker = "packaging marker"
    cond4 = error_msg_marker in msg

    return cond1 or cond2 or cond3 or cond4


@pytest.mark.parametrize("module", modules)
def test_module_softdeps(module):
    """Test soft dependency imports in sktime modules."""
    # We try importing all modules and catch exceptions due to missing dependencies
    try:
        import_module(module)
    except ModuleNotFoundError as e:
        error_msg = str(e)

        if not is_soft_dep_missing_message(error_msg):
            raise RuntimeError(
                f"The module: {module} seems to have unsatisfied soft "
                f"dependency requirements, but does not raise an appropriate error "
                f"message when the soft dependency requirement is not satisfied. "
                f"Please use our `_check_soft_dependencies` function to "
                f"raise an appropriate error with one of the predefined messages."
            ) from e

        # If the error is raised in a module which does depend on a soft dependency,
        # we ignore and skip it
        dependencies = SOFT_DEPENDENCIES.get(module, [])
        if any(dependency in error_msg for dependency in dependencies):
            return None

        # Otherwise we raise an error
        dependency = _extract_dependency_from_error_msg(error_msg)
        raise ModuleNotFoundError(
            f"The module: {module} should not require any soft dependencies, "
            f"but tried importing: '{dependency}'. Make sure soft dependencies are "
            f"properly isolated."
        ) from e


def _has_soft_dep(est):
    """Return whether an estimator has soft dependencies."""
    softdep = est.get_class_tag("python_dependencies", None)
    return softdep is not None


def _coerce_list_of_str(obj):
    """Coerce obj to list of str."""
    if obj is None:
        return []
    elif isinstance(obj, str):
        return [obj]
    elif isinstance(obj, list):
        return obj


def _get_soft_deps(est):
    """Return soft dependencies of an estimator, as list of str."""
    softdeps = est.get_class_tag("python_dependencies", None)
    softdeps = _coerce_list_of_str(softdeps)
    if softdeps is None:
        raise RuntimeError(
            'error, "python_dependencies" tag must be None, str or list of str,'
            f" but {est.__name__} has {softdeps}"
        )
    else:
        return softdeps


def _is_in_env(modules):
    """Return whether all modules in list of str modules are installed in env."""
    modules = _coerce_list_of_str(modules)
    try:
        _check_soft_dependencies(modules)
        return True
    except ModuleNotFoundError:
        return False


def _python_compat(est):
    """Shorthand for silent python compatibility check."""
    return _check_python_version(est, severity="none")


# all estimators - exclude estimators on the global exclusion list
all_ests = all_estimators(return_names=False, exclude_estimators=EXCLUDE_ESTIMATORS)


# estimators that should fail to construct because of python version
est_python_incompatible = [est for est in all_ests if not _python_compat(est)]

# estimators that have soft dependencies
est_with_soft_dep = [est for est in all_ests if _has_soft_dep(est)]
# estimators that have soft dependencies and are python compatible
est_pyok_with_soft_dep = [est for est in est_with_soft_dep if _python_compat(est)]

# estimators that have no soft dependencies
est_without_soft_dep = [est for est in all_ests if not _has_soft_dep(est)]
# estimators that have soft dependencies and are python compatible
est_pyok_without_soft_dep = [est for est in est_without_soft_dep if _python_compat(est)]

# all estimators are now a disjoint union of the three sets:
# est_python_incompatible - python incompatible, should raise python error
# est_pyok_without_soft_dep - python compatible, has no soft dependency
# est_pyok_with_soft_dep - python compatible, has soft dependency


@pytest.mark.parametrize("estimator", est_python_incompatible)
def test_python_error(estimator):
    """Test that estimators raise error if python version is wrong."""
    try:
        estimator.create_test_instance()
    except ModuleNotFoundError as e:
        error_msg = str(e)

        # Check if appropriate exception with useful error message is raised as
        # defined in the `_check_python` function
        expected_error_msg = "requires python version to be"
        if expected_error_msg not in error_msg:
            pyspec = estimator.get_class_tag("python_version", None)
            raise RuntimeError(
                f"Estimator {estimator.__name__} has python version bound "
                f"{pyspec} according to tags, but does not raise an appropriate "
                f"error message on __init__ for incompatible python environments. "
                f"Likely reason is that __init__ does not call super().__init__."
            ) from e


@pytest.mark.parametrize("estimator", est_pyok_with_soft_dep)
def test_softdep_error(estimator):
    """Test that estimators raise error if required soft dependencies are missing."""
    softdeps = _get_soft_deps(estimator)
    if not _is_in_env(softdeps):
        try:
            estimator.create_test_instance()
        except ModuleNotFoundError as e:
            error_msg = str(e)

            if not is_soft_dep_missing_message(error_msg):
                raise RuntimeError(
                    f"Estimator {estimator.__name__} has soft dependency requirements, "
                    f"{softdeps} according to tags, but does not raise an appropriate "
                    f"error message on __init__, when those requirements are unmet. "
                    f"Likely reason is that __init__ does not call super().__init__,"
                    f" or imports super().__init__ only after an attempted import."
                ) from e


@pytest.mark.parametrize("estimator", est_pyok_with_soft_dep)
def test_est_construct_if_softdep_available(estimator):
    """Test that estimators construct if required soft dependencies are there."""
    softdeps = _get_soft_deps(estimator)
    if _is_in_env(softdeps):
        try:
            estimator.create_test_instance()
        except ModuleNotFoundError as e:
            error_msg = str(e)
            raise RuntimeError(
                f"Estimator {estimator.__name__} requires soft dependencies "
                f"{softdeps} according to tags, but raises ModuleNotFoundError "
                f"on __init__ when those dependencies are in the environment. "
                f" Likely cause is additionally needed soft dependencies, "
                f"these should be added "
                f'to the "python_dependencies" tag. Exception text: {error_msg}'
            ) from e


@pytest.mark.parametrize("estimator", all_ests)
def test_est_get_params_without_modulenotfound(estimator):
    """Test that estimator test parameters do not rely on soft dependencies."""
    try:
        estimator.get_test_params()
    except ModuleNotFoundError as e:
        error_msg = str(e)
        raise RuntimeError(
            f"Estimator {estimator.__name__} requires soft dependencies for parameters "
            f"returned by get_test_params. Test parameters should not require "
            f"soft dependencies and use only sktime internal objects. "
            f"In a case where soft dependencies are required, return a shorter list, "
            f"or an empty dict, with parameter sets that do not require soft "
            f"dependencies, gated by a dependency check, for instance using "
            f"skbase.utils.dependencies_check_soft_dependencies with severity='none'. "
            f"Exception text: {error_msg}"
        ) from e


@pytest.mark.parametrize("estimator", est_pyok_without_soft_dep)
def test_est_construct_without_modulenotfound(estimator):
    """Test that estimators that do not require soft dependencies construct properly."""
    # skip composite estimators that have no soft dependencies
    #   but which have soft dependencies in example components
    if estimator.__name__ in EXCEPTED_FROM_NO_DEP_CHECK:
        return None

    try:
        estimator.create_test_instance()
    except ModuleNotFoundError as e:
        error_msg = str(e)
        raise RuntimeError(
            f"Estimator {estimator.__name__} does not require soft dependencies "
            f"according to tags, but raises ModuleNotFoundError "
            f"on __init__ with test parameters. Any required soft dependencies should "
            f'be added to the "python_dependencies" tag, and python version bounds '
            f'should be added to the "python_version" tag. Exception text: {error_msg}'
        ) from e


@pytest.mark.parametrize("estimator", est_pyok_without_soft_dep)
def test_est_fit_without_modulenotfound(estimator):
    """Test that estimators that do not require soft dependencies fit properly."""
    # skip composite estimators that have no soft dependencies
    #   but which have soft dependencies in example components
    if estimator.__name__ in EXCEPTED_FROM_NO_DEP_CHECK:
        return None

    if not run_test_for_class(estimator):
        return None

    try:
        scenarios = retrieve_scenarios(estimator)
        if len(scenarios) == 0:
            return None
        else:
            scenario = scenarios[0]
        estimator_instance = estimator.create_test_instance()
        scenario.run(estimator_instance, method_sequence=["fit"])
    except ModuleNotFoundError as e:
        error_msg = str(e)
        raise RuntimeError(
            f"Estimator {estimator.__name__} does not require soft dependencies "
            f"according to tags, but raises ModuleNotFoundError "
            f"on fit. Any required soft dependencies should be added "
            f'to the "python_dependencies" tag, and python version bounds should be'
            f' added to the "python_version" tag. Exception text: {error_msg}'
        ) from e


@patch("sktime.utils.dependencies._dependencies.sys")
@pytest.mark.parametrize(
    "mock_release_version, prereleases, expect_exception",
    [
        (True, True, False),
        (True, False, True),
        (False, False, False),
        (False, True, False),
    ],
)
def test_check_python_version(
    mock_sys, mock_release_version, prereleases, expect_exception
):
    from sktime.base import BaseObject

    if mock_release_version:
        mock_sys.version = "3.8.1rc"
    else:
        mock_sys.version = "3.8.1"

    class DummyObjectClass(BaseObject):
        _tags = {
            "python_version": ">=3.7.1",  # PEP 440 version specifier, e.g., ">=3.7"
            "python_dependencies": None,  # PEP 440 dependency strs, e.g., "pandas>=1.0"
            "env_marker": None,  # PEP 508 environment marker, e.g., "os_name=='posix'"
        }
        """Define dummy class to test set_tags."""

    dummy_object_instance = DummyObjectClass()

    try:
        _check_python_version(dummy_object_instance, prereleases=prereleases)
    except ModuleNotFoundError as exception:
        expected_msg = (
            f"{type(dummy_object_instance).__name__} requires python version "
            f"to be {dummy_object_instance.get_tag('python_version')}, "
            f"but system python version is {mock_sys.version}. "
            "This is due to the release candidate status of your system Python."
        )

        if not expect_exception or exception.msg != expected_msg:
            # Throw Error since exception is not expected or has not the correct message
            raise AssertionError(
                "ModuleNotFoundError should be NOT raised by:",
                f"\n\t - mock_release_version: {mock_release_version},",
                f"\n\t - prereleases: {prereleases},",
                f"\nERROR MESSAGE: {exception.msg}",
            ) from exception

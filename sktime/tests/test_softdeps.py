# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests that soft dependencies are handled correctly."""

__author__ = ["mloning", "fkiraly"]

import pkgutil
import re

from importlib import import_module

import pytest

from sktime.registry import all_estimators

# list of soft dependencies used
# excludes estimators, only for soft dependencies used in non-estimator modules
SOFT_DEPENDENCIES = {
    "sktime.benchmarking.evaluation": ["matplotlib"],
    "sktime.benchmarking.experiments": ["tsfresh", "esig"],
    "sktime.classification.deep_learning": ["tensorflow"],
    "sktime.networks": ["tensorflow"],
    "sktime.clustering.evaluation._plot_clustering": ["matplotlib"],
}

MODULES_TO_IGNORE = ("sktime._contrib", "sktime.utils._testing")


def _is_test(module):
    module_parts = module.split(".")
    return any(part in ("tests", "test") for part in module_parts)


def _is_ignored(module):
    return any(module_to_ignore in module for module_to_ignore in MODULES_TO_IGNORE)


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
modules = [x for x in modules if not _is_test(x) and not _is_ignored(x)]


@pytest.mark.parametrize("module", modules)
def test_module_softdeps(module):
    """Test soft dependency imports in sktime modules."""
    # We try importing all modules and catch exceptions due to missing dependencies
    try:
        import_module(module)
    except ModuleNotFoundError as e:
        error_msg = str(e)

        # Check if appropriate exception with useful error message is raised as
        # defined in the `_check_soft_dependencies` function
        expected_error_msg = (
            "is a soft dependency and not included in the sktime installation"
        )
        if expected_error_msg not in error_msg:
            raise RuntimeError(
                f"The module: {module} seems to require a soft "
                f"dependency, but does not raise an appropriate error "
                f"message when the soft dependency is missing. Please "
                f"use our `_check_soft_dependencies` function to "
                f"raise a more appropriate error message."
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
    """Returns whether an estimator has soft dependencies."""
    softdep = est.get_class_tag("python_dependencies", None)
    return softdep is not None


est_with_soft_dep = all_estimators(return_names=False)
est_with_soft_dep = [est for est in est_with_soft_dep if _has_soft_dep(est)]


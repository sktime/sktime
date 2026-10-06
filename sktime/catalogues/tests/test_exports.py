"""Tests for the public exports of catalogue packages."""

import importlib
import pkgutil

import pytest

import sktime.catalogues
from sktime.tests.test_switch import run_test_module_changed

CATALOGUE_PACKAGES = [sktime.catalogues.__name__] + [
    modname
    for _, modname, ispkg in pkgutil.walk_packages(
        path=sktime.catalogues.__path__, prefix=sktime.catalogues.__name__ + "."
    )
    if ispkg and not modname.endswith(".tests")
]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.catalogues"),
    reason="run test only if catalogues module has changed",
)
@pytest.mark.parametrize("package_name", CATALOGUE_PACKAGES)
def test_dunder_all_contains_names(package_name):
    """Test that ``__all__`` of catalogue packages lists names of their attributes.

    Failure case of bug #11410, where ``__all__`` of
    ``sktime.catalogues.classification`` contained the classes instead of their
    names, so ``from sktime.catalogues.classification import *`` raised TypeError.
    """
    package = importlib.import_module(package_name)

    for name in package.__all__:
        assert isinstance(name, str)
        assert hasattr(package, name)

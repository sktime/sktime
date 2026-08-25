"""Tests to run without pytest, to check pytest isolation."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# test 1: check that all_estimators can crawl all modules without throwing an exception
# this is a test for soft dependency isolation, including of pytest itself
from sktime.registry import all_estimators

# all_estimators crawls all modules excepting pytest test files
# if it encounters an unisolated import, it will throw an exception
results = all_estimators()

# test 2: check that docs and examples are not installed in site-packages
# this would indicate regression on #10891
import sysconfig, os

site_packages = sysconfig.get_paths()["purelib"]
for name in ("docs", "examples"):
    path = os.path.join(site_packages, name)
    assert not os.path.isdir(path), f"{name} should not be installed in site-packages"

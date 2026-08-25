"""Tests to run without pytest, to check pytest isolation."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# test: check that all_estimators can crawl all modules without throwing an exception
# this is a test for soft dependency isolation, in particular of pytest itself
# (note that isolation of pytest cannot be tested in a pytest test,
# because pytest needs to be already imported to run pytest tests)
from sktime.registry import all_estimators

# all_estimators crawls all modules excepting pytest test files
# if it encounters an unisolated import, it will throw an exception
results = all_estimators()

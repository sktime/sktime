"""Tests to run without pytest, to check pytest isolation."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.registry import all_estimators

# all_estimators crawls all modules excepting pytest test files
# if it encounters an unisolated import, it will throw an exception
results = all_estimators()

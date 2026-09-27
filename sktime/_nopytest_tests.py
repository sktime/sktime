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

# test: check that craft can crawl all sktime and scikit-learn modules without
# throwing an exception
# this is a test for soft dependency isolation, in particular of pytest itself
from sktime.registry import craft

craft("NaiveForecaster")

# test: check that soft dependencies are isolated in the sktime.libs module
# since all_estimators does not crawl this by default,
# we use all_objects from skbase
from pathlib import Path

from skbase.lookup import all_objects as _all_objects

LIBS = str(Path(__file__).parent / "libs")
_all_objects(package_name="sktime.libs", path=LIBS)

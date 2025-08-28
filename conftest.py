"""Main configuration file for pytest.

Contents:
adds the following options to pytest
--matrixdesign : bool, default False
    allows to turn on/off the sub-sampling in the tests (for shorter runtime)
    "on" condition is partition/block design to ensure each estimator full tests are run
    on each operating system at least once, and on each python version at least once,
    but not necessarily on each operating system / python version combination
--only_changed_modules : bool, default False
    turns on/off differential testing (for shorter runtime)
    "on" condition ensures that only estimators are tested that have changed,
    more precisely, only estimators whose class is in a module
    that has changed compared to the main branch
    "off" = runs tests for all estimators

by default, all options are off, including for default local runs of pytest
if multiple options are turned on, they are combined with AND,
i.e., intersection of estimators satisfying the conditions
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]


def pytest_addoption(parser):
    """Pytest command line parser options adder."""
    parser.addoption(
        "--matrixdesign",
        default=False,
        help="sub-sample estimators in tests by os/version matrix partition design",
    )
    parser.addoption(
        "--only_changed_modules",
        default=False,
        help="test only estimators from modules that have changed compared to main",
    )


def pytest_configure(config):
    """Pytest configuration preamble."""
    from sktime.tests import _config

    if config.getoption("--matrixdesign") in [True, "True"]:
        _config.MATRIXDESIGN = True
    if config.getoption("--only_changed_modules") in [True, "True"]:
        _config.ONLY_CHANGED_MODULES = True

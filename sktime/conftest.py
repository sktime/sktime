# -*- coding: utf-8 -*-
"""Main configuration file for pytest.

Contents:
adds a --matrixdesign option to pytest
this allows to turn on/off the sub-sampling in the tests (for shorter runtime)
"on" condition is partition/block design to ensure each estimator full tests are run
    on each operating system at least once, and on each python version at least once,
    but not necessarily on each operating system / python version combination
by default, this is off, including for default local runs of pytest
"""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]


def pytest_addoption(parser):
    parser.addoption(
        "--matrixdesign",
        action="store_true",
        default=False,
        help="sub-sample estimators in tests by os/version matrix partition design",
        choice=("True", "False", True, False),
    )


def pytest_configure(config):
    if config.getoption("--matrixdesign") in [True, "True"]:
        tests.test_all_estimators.MATRIXDESIGN = True

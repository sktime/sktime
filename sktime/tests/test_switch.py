# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Switch utility for determining whether tests for a class should be run or not."""

__author__ = ["fkiraly"]


def run_test_for_class(cls):
    """Check if test should run for a class.

    This checks the following conditions:

    * if required dependencies are not present, does not run the test
    * if ONLY_CHANGED_MODULES setting is on, runs the test
      if and only if the class has changed according ot is_class_changed

    otherwise, always runs the test.

    Parameters
    ----------
    cls : class
        class for which to determine whether it should be tested

    Returns
    -------
    bool : True if class should be tested, False otherwise
    """
    from sktime.tests.test_all_estimators import ONLY_CHANGED_MODULES
    from sktime.utils.git_diff import is_class_changed
    from sktime.utils.validations._dependencies import _check_estimator_deps

    if not _check_estimator_deps(cls, severity="none"):
        return False

    if ONLY_CHANGED_MODULES:
        return is_class_changed(cls)

    # otherwise
    # i.e., dependencies are present, and differential testing is disabled
    return True

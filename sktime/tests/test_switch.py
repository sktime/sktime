# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Switch utility for determining whether tests for a class should be run or not."""

__author__ = ["fkiraly"]


def run_test_for_class(cls):
    """Check if test should run for a class.

    This checks the following conditions:

    1. whether all required soft dependencies are not present.
       If not, does not run the test.
    2. If yes:
      * if ONLY_CHANGED_MODULES setting is on, runs the test if and only
      if the module containing the class has changed according to is_class_changed
      * if ONLY_CHANGED_MODULES if off, always runs the test if all soft dependencies
      are present.

    cls can also be a list, in this case the test is run if and only if:

    * all required soft dependencies are present
    * if yes, if any of the estimators in the list should be tested by criterion 2 above

    Parameters
    ----------
    cls : class or list of class
        class for which to determine whether it should be tested

    Returns
    -------
    bool : True if class should be tested, False otherwise
        if cls was a list, is True iff True for at least one of the classes in the list
    """
    if not isinstance(cls, list):
        cls = [cls]

    from sktime.tests.test_all_estimators import ONLY_CHANGED_MODULES
    from sktime.utils.git_diff import is_class_changed
    from sktime.utils.validation._dependencies import _check_estimator_deps

    # if any of the required soft dependencies are not present, do not run the test
    if not all(_check_estimator_deps(x, severity="none") for x in cls):
        return False

    # if ONLY_CHANGED_MODULES is on, run the test if and only if
    # any of the modules containing any of the classes in the list have changed
    if ONLY_CHANGED_MODULES:
        return any(is_class_changed(x) for x in cls)

    # otherwise
    # i.e., dependencies are present, and differential testing is disabled
    return True

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Switch utility for determining whether tests for a class should be run or not."""

__author__ = ["fkiraly"]

from inspect import getmro, isclass


def run_test_for_class(cls):
    """Check if test should run for a class or function.

    This checks the following conditions:

    1. whether all required soft dependencies are present.
       If not, does not run the test.
    2. If yes:
      * if ONLY_CHANGED_MODULES setting is on, runs the test if and only
      if the module containing the class/func has changed according to is_class_changed
      * if ONLY_CHANGED_MODULES if off, always runs the test if all soft dependencies
      are present.

    cls can also be a list of classes or functions,
    in this case the test is run if and only if:

    * all required soft dependencies are present
    * if yes, if any of the estimators in the list should be tested by criterion 2 above

    Parameters
    ----------
    cls : class, function or list of classes/functions
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

    def _required_deps_present(obj):
        """Check if all required soft dependencies are present, return bool."""
        if hasattr(obj, "get_class_tag"):
            return _check_estimator_deps(obj, severity="none")
        else:
            return True

    def _is_class_changed_or_sktime_parents(cls):
        """Check if class or any of its sktime parents have changed, return bool."""
        # if cls is a function, not a class, default to is_class_changed
        if not isclass(cls):
            return is_class_changed(cls)

        # now we know cls is a class, so has an mro
        cls_and_parents = getmro(cls)
        cls_and_sktime_parents = [
            x for x in cls_and_parents if x.__module__.startswith("sktime")
        ]
        return any(is_class_changed(x) for x in cls_and_sktime_parents)

    # if any of the required soft dependencies are not present, do not run the test
    if not all(_required_deps_present(x) for x in cls):
        return False

    # if ONLY_CHANGED_MODULES is on, run the test if and only if
    # any of the modules containing any of the classes in the list have changed
    if ONLY_CHANGED_MODULES:
        return any(_is_class_changed_or_sktime_parents(x) for x in cls)

    # otherwise
    # i.e., dependencies are present, and differential testing is disabled
    return True

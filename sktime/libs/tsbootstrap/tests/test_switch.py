# copyright: tsbootstrap developers, BSD-3-Clause License (see LICENSE file)
# based on utility from sktime of the same name
"""Switch utility for determining whether tests for a class should be run or not."""

__author__ = ["fkiraly"]


def run_test_for_class(cls):
    """Check if test should run for a class or function.

    This checks the following conditions:

    1. whether all required soft dependencies are present.
       If not, does not run the test.
       If yes, runs the test

    cls can also be a list of classes or functions,
    in this case the test is run if and only if:

    * all required soft dependencies are present

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

    from tsbootstrap.utils.dependencies import _check_estimator_deps

    def _required_deps_present(obj):
        """Check if all required soft dependencies are present, return bool."""
        if hasattr(obj, "get_class_tag"):
            return _check_estimator_deps(obj, severity="none")
        else:
            return True

    # Condition 1:
    # if any of the required soft dependencies are not present, do not run the test
    if not all(_required_deps_present(x) for x in cls):
        return False
    # otherwise, continue

    return True

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Switch utility for determining whether tests for a class should be run or not.

Module does not contain tests, only test utilities.
"""

__author__ = ["fkiraly"]

from inspect import getmro, isclass


def run_test_for_class(cls):
    """Check if test should run for a class or function.

    This checks the following conditions:

    1. whether all required soft dependencies are present.
       If not, does not run the test.
       If yes, behaviour depends on ONLY_CHANGED_MODULES setting:
       if off (False), always runs the test (return True);
       if on (True), runs test if and only if
       at least one of conditions 2, 3, 4 below are met.

    2. Condition 2:

      If the module containing the class/func has changed according to is_class_changed,
      or one of the modules containing any parent classes in sktime,
      then condition 2 is met.

    3. Condition 3:

      If the object is an sktime ``BaseObject``, and one of the test classes
      covering the class have changed, then condition 3 is met.

    4. Condition 4:

      If the object is an sktime ``BaseObject``, and the package requirements
      for any of its dependencies have changed in ``pyproject.toml``,
      condition 4 is met.

    cls can also be a list of classes or functions,
    in this case the test is run if and only if both of the following are True:

    * all required soft dependencies are present
    * if ``ONLY_CHANGED_MODULES`` is True, additionally,
      if any of the estimators in the list should be tested by
      at least one of criteria 2-4 above.
      If ``ONLY_CHANGED_MODULES`` is False, this condition is always True.

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
    from sktime.utils.git_diff import get_packages_with_changed_specs, is_class_changed
    from sktime.utils.validation._dependencies import _check_estimator_deps

    PACKAGE_REQ_CHANGED = get_packages_with_changed_specs()

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

    def _tests_covering_class_changed(cls):
        """Check if any of the tests covering cls have changed, return bool."""
        from sktime.tests.test_class_register import get_test_classes_for_obj

        test_classes = get_test_classes_for_obj(cls)
        return any(is_class_changed(x) for x in test_classes)

    def _is_impacted_by_pyproject_change(cls):
        """Check if the dep specifications of cls have changed, return bool."""
        from packaging.requirements import Requirement

        if not isclass(cls) or not hasattr(cls, "get_class_tags"):
            return False

        cls_reqs = cls.get_class_tag("python_dependencies", [])
        if cls_reqs is None:
            cls_reqs = []
        if not isinstance(cls_reqs, list):
            cls_reqs = [cls_reqs]
        package_deps = [Requirement(req).name for req in cls_reqs]

        return any(x in PACKAGE_REQ_CHANGED for x in package_deps)

    # Condition 1:
    # if any of the required soft dependencies are not present, do not run the test
    if not all(_required_deps_present(x) for x in cls):
        return False
    # otherwise, continue

    # if ONLY_CHANGED_MODULES is off: always True
    # tests are always run if soft dependencies are present
    if not ONLY_CHANGED_MODULES:
        return True

    # Condition 2:
    # any of the modules containing any of the classes in the list have changed
    # or any of the modules containing any parent classes in sktime have changed
    cond2 = any(_is_class_changed_or_sktime_parents(x) for x in cls)

    # Condition 3:
    # if the object is an sktime BaseObject, and one of the test classes
    # covering the class have changed, then run the test
    cond3 = any(_tests_covering_class_changed(x) for x in cls)

    # Condition 4:
    # the package requirements for any dependency in pyproject.toml have changed
    cond4 = any(_is_impacted_by_pyproject_change(x) for x in cls)

    # run the test if and only if at least one of the conditions 2-4 are met
    return cond2 or cond3 or cond4

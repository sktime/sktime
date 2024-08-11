# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Switch utility for determining whether tests for a class should be run or not.

Module does not contain tests, only test utilities.
"""

__author__ = ["fkiraly"]

from functools import lru_cache
from inspect import getmro, isclass

from sktime.tests._config import EXCLUDE_ESTIMATORS

LOCAL_PACKAGE = "sktime"


def run_test_for_class(cls, return_reason=False):
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
      or one of the modules containing any parent classes  in the local package,
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

    Also checks whether the class or function is on the exclude override list,
    EXCLUDE_ESTIMATORS in sktime.tests._config (a list of strings, of names).
    If so, the tests are always skipped, irrespective of the other conditions.

    Parameters
    ----------
    cls : class, function or list of classes/functions
        class for which to determine whether it should be tested
    return_reason: bool, optional, default=False
        whether to return the reason for running or skipping the test

    Returns
    -------
    bool : True if class should be tested, False otherwise
        if cls was a list, is True iff True for at least one of the classes in the list
    reason: str, reason to run or skip the test, returned only if ``return_reason=True``

        * "False_exclude_list" - skip reason, class is on the exclude list
        * "False_required_deps_missing" - skip reason, required dependencies are missing
        * "False_no_change" - skip reason, no change in class or dependencies
        * "True_run_always" - run reason, run always, as ``ONLY_CHANGED_MODULES=False``
        * "True_pyproject_change" - run reason, dep(s) in ``pyproject.toml`` changed
        * "True_changed_tests" - run reason, test(s) covering class have changed
        * "True_changed_class" - run reason, module(s) containing class changed

        If multiple reasons are present, the first one in the above list is returned.

        If ``cls`` was a list, then:

        * reasons to skip - except "no change" - cause the entire list to be skipped
        * otherwise, any reasons to run cause the entire list to be run
        * otherwise, the list is not run due to "no change"
    """

    def _return(run, reason):
        if return_reason:
            return run, reason
        return run

    if isinstance(cls, (list, tuple)):
        runs = [run_test_for_class(x, return_reason=True) for x in cls]
        reasons = [x[1] for x in runs]

        # check the negative reasons that would cause the test to be skipped
        #
        # this excludes the "no change" reason, because:
        # * special negative reasons cause entire list to be skipped
        # * otherwise, positive reason causes the entire list to be run
        # * if no positive reason, then list is skipped due to lack of positive reason
        #
        # if any of the classes are on the skip list, return False
        # if any of the classes are missing dependencies, return False
        NEG_REASONS = [
            "False_exclude_list",
            "False_required_deps_missing",
        ]
        for neg_reason in NEG_REASONS:
            if any(reason == neg_reason for reason in reasons):
                return _return(False, neg_reason)

        # now check the "any of the classes should be tested" condition
        POS_REASONS = [
            "True_run_always",
            "True_pyproject_change",
            "True_changed_tests",
            "True_changed_class",
        ]
        for pos_reason in POS_REASONS:
            if any(reason == pos_reason for reason in reasons):
                return _return(True, pos_reason)

        # otherwise, we do not run, and the reason is "no change"
        return _return(False, "False_no_change")

    # if object is passed, obtain the class - objects are not hashable
    if hasattr(cls, "get_class_tag") and not isclass(cls):
        cls = cls.__class__
    # check whether estimator is on the exclude override list
    if cls.__name__ in EXCLUDE_ESTIMATORS:
        return _return(False, "False_exclude_list")

    # now we know that cls is a class or function,
    # and not on the exclude list
    run, reason = _run_test_for_class(cls)
    return _return(run, reason)


@lru_cache
def _run_test_for_class(cls):
    """Check if test should run - cached with hashable cls.

    Parameters
    ----------
    cls : class, function or list of classes/functions
        class for which to determine whether it should be tested

    Returns
    -------
    bool : True if class should be tested, False otherwise
    reason : str, reason to run or skip the test, one of:

        * "False_required_deps_missing" - skip reason, required dependencies are missing
        * "False_no_change" - skip reason, no change in class or dependencies
        * "True_run_always" - run reason, run always, as ``ONLY_CHANGED_MODULES=False``
        * "True_pyproject_change" - run reason, dep(s) in ``pyproject.toml`` changed
        * "True_changed_tests" - run reason, test(s) covering class have changed
        * "True_changed_class" - run reason, module(s) containing class changed

        If multiple reasons are present, the first one in the above list is returned.
    """
    from sktime.tests.test_all_estimators import ONLY_CHANGED_MODULES
    from sktime.utils.dependencies import _check_estimator_deps
    from sktime.utils.git_diff import get_packages_with_changed_specs, is_class_changed

    PACKAGE_REQ_CHANGED = get_packages_with_changed_specs()

    def _required_deps_present(obj):
        """Check if all required soft dependencies are present, return bool."""
        if hasattr(obj, "get_class_tag"):
            return _check_estimator_deps(obj, severity="none")
        else:
            return True

    def _is_class_changed_or_local_parents(cls):
        """Check if class or any of its local parents have changed, return bool."""
        # if cls is a function, not a class, default to is_class_changed
        if not isclass(cls):
            return is_class_changed(cls)

        # now we know cls is a class, so has an mro
        cls_and_parents = getmro(cls)
        cls_and_local_parents = [
            x for x in cls_and_parents if x.__module__.startswith(LOCAL_PACKAGE)
        ]
        return any(is_class_changed(x) for x in cls_and_local_parents)

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
    if not _required_deps_present(cls):
        return False, "False_required_deps_missing"
    # otherwise, continue

    # if ONLY_CHANGED_MODULES is off: always True
    # tests are always run if soft dependencies are present
    if not ONLY_CHANGED_MODULES:
        return True, "True_run_always"

    # run the test if and only if at least one of the conditions 2-4 are met
    # conditions are checked in order to minimize runtime due to git diff etc

    # Condition 4:
    # the package requirements for any dependency in pyproject.toml have changed
    cond4 = _is_impacted_by_pyproject_change(cls)
    if cond4:
        return True, "True_pyproject_change"

    # Condition 3:
    # if the object is an sktime BaseObject, and one of the test classes
    # covering the class have changed, then run the test
    cond3 = _tests_covering_class_changed(cls)
    if cond3:
        return True, "True_changed_tests"

    # Condition 2:
    # any of the modules containing any of the classes in the list have changed
    # or any of the modules containing any parent classes in local package have changed
    cond2 = _is_class_changed_or_local_parents(cls)
    if cond2:
        return True, "True_changed_class"

    # if none of the conditions are met, do not run the test
    # reason is that there was no change
    return False, "False_no_change"


def run_test_module_changed(module):
    """Check if test should run based on module changes

    This switch can be used to decorate tests not pertaining to a specific class.

    The function can be used to switch tests on and off
    based on whether a target module has changed.

    This checks whether the module ``module``, or any of its child modules,
    have changed.

    If ``ONLY_CHANGED_MODULES`` is False, the test is always run,
    i.e., this function always returns True.

    Parameters
    ----------
    module : string, or list of strings
        modules to check for changes, e.g., ``sktime.forecasting``

    Returns
    -------
    bool : switch to run or skip the test
        True iff: at least one of the modules or its submodules have changed,
        or if ``ONLY_CHANGED_MODULES`` is False
    """
    from sktime.tests.test_all_estimators import ONLY_CHANGED_MODULES
    from sktime.utils.git_diff import is_module_changed

    # if ONLY_CHANGED_MODULES is off: always True
    # tests are always run if soft dependencies are present
    if not ONLY_CHANGED_MODULES:
        return True

    if not isinstance(module, (list, tuple)):
        module = [module]

    return any(is_module_changed(mod) for mod in module)

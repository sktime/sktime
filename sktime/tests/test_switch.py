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
       at least one of conditions 2, 3, 4, 5, 6 below are met.

    2. Condition 2:

      If the module containing the class/func has changed according to is_class_changed,
      then condition 2 is met.
      If the class is a core object, then in addition, condition 2 is also met,
      if one of the modules containing any parent classes in the local package
      have changed.

    3. Condition 3 (only checked for core objects):

      If the object is an sktime ``BaseObject``, and one of the test classes
      covering the class have changed, then condition 3 is met.

    4. Condition 4:

      If the object is an sktime ``BaseObject``, and the package requirements
      for any of its dependencies have changed in ``pyproject.toml``,
      condition 4 is met.

    5. Condition 5 (only checked for core objects):

      If the object is an sktime ``BaseObject``,
      and one of the core framework modules ``datatypes``, ``tests``, ``utils``
      have changed, then condition 5 is met.

    6. Condition 6:

      If the object is an sktime ``BaseObject``, and any of the modules
      in the class tag ``tests:libs`` hvae changed, condition 6 is met.

    cls can also be a list of classes or functions,
    in this case the test is run if and only if both of the following are True:

    * all required soft dependencies are present
    * if ``ONLY_CHANGED_MODULES`` is True, additionally,
      if any of the estimators in the list should be tested by
      at least one of criteria 2-5 above.
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
        * "False_requires_vm" - skip reason, class requires its own VM.
        * "False_no_change" - skip reason, no change in class or dependencies
        * "True_run_always" - run reason, run always, as ``ONLY_CHANGED_MODULES=False``
        * "True_pyproject_change" - run reason, dep(s) in ``pyproject.toml`` changed
        * "True_changed_tests" - run reason, test(s) covering class have changed
        * "True_changed_class" - run reason, module(s) containing class changed
        * "True_changed_framework" - run reason, core framework modules changed
        * "True_changed_libs" - run reason, library dependencies have changed

        If multiple reasons are present, the first one in the above list is returned.

        If ``cls`` was a list, then:

        * reasons to skip - except "no change" - cause the entire list to be skipped
        * otherwise, any reasons to run cause the entire list to be run
        * otherwise, the list is not run due to "no change"
    """
    from sktime.tests._config import ONLY_CHANGED_MODULES

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
            "True_changed_framework",
            "True_changed_libs",
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
    run, reason = _run_test_for_class(cls, only_changed_modules=ONLY_CHANGED_MODULES)
    return _return(run, reason)


def _flatten_list(nested_list):
    """Recursively flattens a nested list or tuple of arbitrary depth."""
    flat_list = []

    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(_flatten_list(item))  # Recursively flatten
        else:
            flat_list.append(item)

    return flat_list


@lru_cache
def _run_test_for_class(
    cls,
    ignore_deps=False,
    only_changed_modules=True,
    only_vm_required=False,
):
    """Check if test should run - cached with hashable cls.

    Parameters
    ----------
    cls : class, function or list of classes/functions
        class for which to determine whether it should be tested
    ignore_deps : boolean, default=False
        whether to ignore the soft dependencies check.
        If True, will not skip due to False_required_deps_missing, see below.
    only_changed_modules : boolean, default=True
        whether to run tests only for classes impacted by changed modules.
        If False, will only check active "False" conditions to skip.
    only_vm_required : boolean, default=False
        whether th return only classes that require their own VM.
        If True, will only return classes with tag "tests:vm"=True.
        If False, will only return classes with tag "tests:vm"=False.

    Returns
    -------
    bool : True if class should be tested, False otherwise
    reason : str, reason to run or skip the test, one of:

        * "False_required_deps_missing" - skip reason, required dependencies are missing
        * "False_requires_vm" - skip reason, class requires its own VM.
        * "False_no_change" - skip reason, no change in class or dependencies.
          Only active if ``ignore_deps=False``.
        * "True_run_always" - run reason, run always, as ``ONLY_CHANGED_MODULES=False``
        * "True_pyproject_change" - run reason, dep(s) in ``pyproject.toml`` changed
        * "True_changed_tests" - run reason, test(s) covering class have changed
        * "True_changed_class" - run reason, module(s) containing class changed
        * "True_changed_framework" - run reason, core framework modules changed
        * "True_changed_libs" - run reason, library dependencies changed

        If multiple reasons are present, the first one in the above list is returned.
    """
    from sktime.utils.dependencies import _check_estimator_deps
    from sktime.utils.git_diff import (
        get_packages_with_changed_specs,
        is_class_changed,
        is_module_changed,
    )

    PACKAGE_REQ_CHANGED = get_packages_with_changed_specs()

    def _required_deps_present(obj):
        """Check if all required soft dependencies are present, return bool."""
        if hasattr(obj, "get_class_tag"):
            return _check_estimator_deps(obj, severity="none")
        else:
            return True

    def _is_core_object(cls):
        """Check if the class is a core object, for condition 3 and 5."""
        if hasattr(cls, "get_class_tag"):
            return cls.get_class_tag("tests:core", False)
        return False

    def _is_class_changed_or_local_parents(cls):
        """Check if class or any of its local parents have changed, return bool."""
        # if cls is a function, not a class, default to is_class_changed
        if not isclass(cls) or not _is_core_object(cls):
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

    def _requires_vm(cls):
        """Check if the class requires a test VM, return bool."""
        if not isclass(cls) or not hasattr(cls, "get_class_tags"):
            return False
        return cls.get_class_tag("tests:vm", False)

    def _is_impacted_by_pyproject_change(cls, include_core_deps=False):
        """Check if the dep specifications of cls have changed, return bool."""
        from packaging.requirements import Requirement

        if not isclass(cls) or not hasattr(cls, "get_class_tags"):
            return False

        cls_reqs = cls.get_class_tag("python_dependencies", [])
        if cls_reqs is None:
            cls_reqs = []
        if not isinstance(cls_reqs, list):
            cls_reqs = [cls_reqs]
        cls_reqs = _flatten_list(cls_reqs)
        package_deps = [Requirement(req).name for req in cls_reqs]

        if include_core_deps:
            CORE_DEPENDENCIES = [
                "scikit-base",
                "scikit-learn",
                "scipy",
                "numpy",
                "pandas",
                "scikit-base",
            ]
            package_deps += CORE_DEPENDENCIES

        return any(x in PACKAGE_REQ_CHANGED for x in package_deps)

    def _is_impacted_by_lib_dep_change(cls, only_changed_modules):
        """Check if library dependencies have changed, return bool."""
        if not isclass(cls) or not hasattr(cls, "get_class_tags"):
            return False

        libs = cls.get_class_tag("tests:libs", [])
        if libs is None or libs == []:
            return False

        return run_test_module_changed(libs, only_changed_modules=only_changed_modules)

    # Condition 1:
    # if any of the required soft dependencies are not present, do not run the test
    if not ignore_deps and not _required_deps_present(cls):
        return False, "False_required_deps_missing"
    # otherwise, continue

    # if only_vm_required=False, and the class requires a test vm, skip
    if not only_vm_required and _requires_vm(cls):
        return False, "False_requires_vm"
    # if only_vm_required=True, and the class does not require a test vm, skip
    if only_vm_required and not _requires_vm(cls):
        return False, "False_requires_vm"

    # if ONLY_CHANGED_MODULES is off: always True
    # tests are always run if soft dependencies are present
    if not only_changed_modules:
        return True, "True_run_always"

    # run the test if and only if at least one of the conditions 2-4 are met
    # conditions are checked in order to minimize runtime due to git diff etc

    # variable: is cls a core object?
    cls_is_core = _is_core_object(cls)

    # Condition 4:
    # the package requirements for any dependency in pyproject.toml have changed
    cond4 = _is_impacted_by_pyproject_change(cls, include_core_deps=cls_is_core)
    if cond4:
        return True, "True_pyproject_change"

    # Condition 3:
    # if the object is an sktime BaseObject, and one of the test classes
    # covering the class have changed, then run the test
    if cls_is_core and _tests_covering_class_changed(cls):
        return True, "True_changed_tests"

    # Condition 2:
    # any of the modules containing any of the classes in the list have changed
    # additionally, for core objects:
    # any of the modules containing any parent classes in local package have changed
    cond2 = _is_class_changed_or_local_parents(cls)
    if cond2:
        return True, "True_changed_class"

    # Condition 5 (only for core objects):
    # if the object is an sktime BaseObject, and one of the core framework modules
    # datatypes, tests, utils have changed, then run the test
    if cls_is_core:
        FRAMEWORK_MODULES = [
            "sktime.datatypes",
            "sktime.tests._config",
            "sktime.tests.test_all_estimators",
            "sktime.tests.test_class_register",
            "sktime.tests.test_doctest",
            "sktime.tests.test_softdeps",
            "sktime.tests.test_switch",
            "sktime.utils",
        ]
        if any([is_module_changed(x) for x in FRAMEWORK_MODULES]):
            return True, "True_changed_framework"

    # Condition 6:
    # any of the specified library dependencies within sktime have changed
    if _is_impacted_by_lib_dep_change(cls, only_changed_modules=only_changed_modules):
        return True, "True_changed_libs"

    # if none of the conditions are met, do not run the test
    # reason is that there was no change
    return False, "False_no_change"


def run_test_module_changed(module, only_changed_modules=None):
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
    only_changed_modules : boolean or None, default=_config.ONLY_CHANGED_MODULES
        whether to run tests only for classes impacted by changed modules.
        If False, will only check active "False" conditions to skip.
        If True, always returns True.
        if None, uses the global setting from
        sktime.tests._config.ONLY_CHANGED_MODULES

    Returns
    -------
    bool : switch to run or skip the test
        True iff: at least one of the modules or its submodules have changed,
        or if ``ONLY_CHANGED_MODULES`` is False
    """
    # default value for only_changed_modules
    if only_changed_modules is None:
        from sktime.tests._config import ONLY_CHANGED_MODULES

        only_changed_modules = ONLY_CHANGED_MODULES

    # if only_changed_modules is off: always True
    # tests are always run if soft dependencies are present
    if not only_changed_modules:
        return True

    from sktime.utils.git_diff import is_module_changed

    if not isinstance(module, (list, tuple)):
        module = [module]

    return any(is_module_changed(mod) for mod in module)


@lru_cache
def _get_all_changed_classes(vm=False):
    """Get all sktime object classes that have changed compared to the main branch.

    Returns a tuple of string class names of object classes that have changed.

    Parameters
    ----------
    vm : bool, optional, default=False
        whether to run estimator in its own virtual machine.
        Queries the tag ``"tests:vm"`` in the class tags.
        If ``vm`` is True, only classes with tag ``"tests:vm"=True`` are returned.

    Returns
    -------
    tuple of strings of class names : object classes that have changed
    """
    from sktime.registry import all_estimators

    def _changed_class(cls):
        """Check if a class has changed compared to the main branch."""
        run, _ = _run_test_for_class(cls, ignore_deps=True, only_vm_required=vm)
        return run

    names = [name for name, est in all_estimators() if _changed_class(est)]
    return names

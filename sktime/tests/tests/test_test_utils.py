"""Tests for the test utilities."""

from skbase.utils.dependencies import _check_estimator_deps

from sktime.registry import all_estimators
from sktime.tests._config import (
    EXCLUDE_SOFT_DEPS,
    EXCLUDED_TESTS_BY_TEST,
)
from sktime.tests.test_switch import run_test_for_class


def test_excluded_tests_by_test():
    """Test that EXCLUDED_TESTS_BY_TEST contains estimators with <2 test params."""
    all_ests = all_estimators()
    filtered_estimators = [
        x[0]
        for x in all_ests
        if (
            (
                len(x[1].get_test_params()) < 2
                or isinstance(x[1].get_test_params(), dict)
            )
            and (
                len(x[1].get_param_names())
                - len(x[1].get_class_tags().get("reserved_params", []))
                > 0
            )
        )
    ]
    excluded_estimators = EXCLUDED_TESTS_BY_TEST["test_get_test_params_coverage"]
    assert set(excluded_estimators) - set(EXCLUDE_SOFT_DEPS) <= set(
        filtered_estimators
    ) - set(EXCLUDE_SOFT_DEPS), (
        "If this PR adds test parameters to an estimator's get_test_params: "
        "Please ensure to remove this estimator "
        "from EXCLUDED_TESTS_BY_TEST and EXCLUDE_SOFT_DEPS "
        "in sktime.tests._config, if it is present there."
    )


def test_run_test_for_class():
    """Test that run_test_for_class runs tests for various cases."""
    # estimator on the exception list
    from sktime.classification.hybrid import HIVECOTEV2

    # estimator with soft deps
    from sktime.forecasting.fbprophet import Prophet

    # estimator without soft deps
    from sktime.forecasting.naive import NaiveForecaster

    # boolean flag for whether to run tests for all estimators
    from sktime.tests._config import ONLY_CHANGED_MODULES

    # test that assumptions on being on exception list are correct
    assert HIVECOTEV2.get_class_tag("tests:skip_all", False)
    # if this fails, switch the example
    assert not NaiveForecaster.get_class_tag("tests:skip_all", False)  # same here
    assert not Prophet.get_class_tag("tests:skip_all", False)  # same here

    f_on_excl_list = HIVECOTEV2
    f_no_deps = NaiveForecaster
    f_with_deps = Prophet

    # check result for skipped estimator
    run = run_test_for_class(f_on_excl_list)
    # run should be False, as the estimator is on the exception list
    assert isinstance(run, bool)
    assert not run
    # same with reason returned
    res = run_test_for_class(f_on_excl_list, return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run, reason = res
    assert isinstance(run, bool)
    assert not run
    assert isinstance(reason, str)
    assert reason == "False_exclude_list"

    # check result for estimator without soft deps
    run = run_test_for_class(f_no_deps)
    assert isinstance(run, bool)
    if not ONLY_CHANGED_MODULES:  # if we run all tests, we should run this one
        assert run

    # result depends now on whether there is a change in the classes
    res = run_test_for_class(f_no_deps, return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run_nodep, reason_nodep = res
    assert isinstance(run_nodep, bool)
    assert isinstance(reason_nodep, str)

    POS_REASONS = [
        "True_pyproject_change",
        "True_changed_class",
        "True_changed_tests",
        "True_changed_framework",
        "True_changed_libs",
    ]

    if not ONLY_CHANGED_MODULES:
        assert run_nodep
        assert reason_nodep == "True_run_always"
    elif run_nodep:
        # otherwise, if we run, it must be due to changes in class or pyproject
        assert reason_nodep in POS_REASONS
    else:  # not run and only changed modules
        assert reason_nodep in ["False_no_change", "False_requires_vm"]

    # now check estimator with soft deps
    run_wdep = run_test_for_class(f_with_deps)
    assert isinstance(run, bool)

    dep_present = _check_estimator_deps(f_with_deps, severity="none")
    if not dep_present:
        assert not run_wdep

    res = run_test_for_class(f_with_deps, return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run_wdep, reason_wdep = res

    if not dep_present:
        assert not run_wdep
        assert reason_wdep in ["False_required_deps_missing", "False_requires_vm"]
    elif not ONLY_CHANGED_MODULES:
        assert run_wdep
        assert reason_wdep == "True_run_always"
    elif run_wdep:
        assert reason_wdep in POS_REASONS
    else:  # not run and only changed modules
        assert reason_wdep in ["False_no_change", "False_requires_vm"]

    # now a list of estimator with exception plus one estimator
    run = run_test_for_class([f_on_excl_list, f_no_deps])
    assert isinstance(run, bool)
    assert not run

    res = run_test_for_class([f_on_excl_list, f_no_deps], return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run, reason = res
    assert isinstance(run, bool)
    assert not run
    assert reason == "False_exclude_list"

    # now a list of the estimator with and without soft deps
    run = run_test_for_class([f_no_deps, f_with_deps])
    assert isinstance(run, bool)

    # if deps are not present, we do not run the test
    # otherwise we run the test iff we run one of the two
    if not dep_present:
        assert not run
    else:
        assert run == run_nodep or run_wdep

    res = run_test_for_class([f_no_deps, f_with_deps], return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run, reason = res

    if not dep_present:
        assert not run
        assert reason in ["False_required_deps_missing", "False_requires_vm"]
    elif not ONLY_CHANGED_MODULES:
        assert run
        assert reason == "True_run_always"
    elif run:
        assert reason in POS_REASONS
        assert reason_wdep == reason or reason_nodep == reason
    else:
        assert reason == "False_no_change"
        assert reason_wdep in ["False_no_change", "False_requires_vm"]
        assert reason_nodep == "False_no_change"


def _skip_all_vm_class():
    """Return a class with tests:skip_all and tests:vm both set, or None."""
    ests = all_estimators(filter_tags={"tests:vm": True}, return_names=False)
    for est in ests:
        if est.get_class_tag("tests:skip_all", False):
            return est
    return None


def test_skip_all_not_in_vm_class_list():
    """Test that tests:skip_all classes are not scheduled for VM testing.

    The VM job matrix is built from ``_get_all_vm_classes`` and
    ``_get_all_changed_classes``, neither of which goes through the public
    ``run_test_for_class``. Both must still honour the ``tests:skip_all`` tag.
    """
    from sktime.tests.test_switch import _get_all_vm_classes, _run_test_for_class

    ests = dict(all_estimators())
    on_exclude_list = [
        name
        for name in _get_all_vm_classes()
        if ests[name].get_class_tag("tests:skip_all", False)
    ]
    assert on_exclude_list == [], (
        "classes with the tests:skip_all tag must not be scheduled for VM testing, "
        f"but these were: {on_exclude_list}"
    )

    cls = _skip_all_vm_class()
    if cls is None:
        return
    # the private helper is what the changed-class list is built from,
    # so it must return the skip reason on its own, not only via the public wrapper
    assert _run_test_for_class(cls, ignore_deps=True, only_vm_required=True) == (
        False,
        "False_exclude_list",
    )


def test_run_test_vm_skips_skip_all(monkeypatch):
    """Test that run_test_vm does not test a class with the tests:skip_all tag.

    ``run_test_vm`` passes the class to ``check_estimator`` directly, which does not
    go through the fixture generation where ``is_excluded`` would apply the tag.
    """
    import skbase.utils.dependencies

    import sktime.utils
    from sktime.tests._test_vm import run_test_vm

    cls = _skip_all_vm_class()
    if cls is None:
        return

    called = []

    def _spy(*args, **kwargs):
        called.append((args, kwargs))

    # both are imported inside run_test_vm, so patch them at their source
    monkeypatch.setattr(sktime.utils, "check_estimator", _spy)
    # ensure the early return for missing soft dependencies cannot mask the tag check
    monkeypatch.setattr(
        skbase.utils.dependencies, "_check_estimator_deps", lambda *a, **k: True
    )

    run_test_vm(cls.__name__)

    assert called == [], (
        f"run_test_vm ran tests for {cls.__name__}, which has the tests:skip_all tag"
    )

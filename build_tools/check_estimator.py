"""Console accessible pytest script to check a single estimator.

Used in test-est job for single-estimator-VM, and for manual testing via pytest.
"""

import pytest

from sktime.utils.estimator_checks import _get_test_names_from_class


def pytest_addoption(parser):
    parser.addoption(
        "--estimator",
        action="store",
        default="__none__",
        help="Custom string parameter for this file",
    )


def pytest_generate_tests(metafunc):
    estimator = metafunc.config.getoption("estimator")

    if estimator == "__none__":
        pytest.skip("No estimator specified, skipping test.")
    elif "estimator" in metafunc.fixturenames:
        metafunc.parametrize("estimator", [estimator], ids=[estimator])

    if "test_name" in metafunc.fixturenames:
        from sktime.utils.estimator_checks import _get_test_names_from_class

        test_names = _get_test_names_from_class(estimator)
        metafunc.parametrize("test_name", test_names, ids=test_names)


def test_sktime_compatible_estimator(estimator, test_name):
    from sktime.registry import craft
    from sktime.utils.dependencies import _check_estimator_deps
    from sktime.utils.estimator_checks import check_estimator

    cls = craft(estimator)
    if not _check_estimator_deps(cls, severity="none"):
        print(
            f"Skipping estimator: {cls} due to incompatibility "
            "with python or OS version."
        )
        return None

    skips = cls.get_class_tag("tests:skip_by_name", None)
    check_estimator(
        cls,
        raise_exceptions=True,
        tests_to_run=test_name,
        tests_to_exclude=skips,
    )

"""Tests for check_n_jobs."""

import os

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.validation import check_n_jobs


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.validation"),
    reason="run test only if the validation module has changed",
)
def test_check_n_jobs_positive_and_none():
    """Test that None means one job and positive values are returned as is."""
    assert check_n_jobs(None) == 1
    assert check_n_jobs(1) == 1
    assert check_n_jobs(4) == 4


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.validation"),
    reason="run test only if the validation module has changed",
)
def test_check_n_jobs_negative():
    """Test the joblib convention for negative n_jobs, see #5990.

    -1 means all CPUs, -2 all CPUs but one, and so on, with a minimum of 1,
    the same as ``joblib.parallel.effective_n_jobs``. Negative values used to
    be mapped to more jobs than there are CPUs.
    """
    from joblib.parallel import effective_n_jobs

    n_cpus = os.cpu_count()
    assert check_n_jobs(-1) == n_cpus
    assert check_n_jobs(-2) == max(n_cpus - 1, 1)
    assert check_n_jobs(-n_cpus - 10) == 1

    for n_jobs in [-1, -2, -3, -n_cpus, -n_cpus - 1]:
        assert check_n_jobs(n_jobs) == effective_n_jobs(n_jobs)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.validation"),
    reason="run test only if the validation module has changed",
)
@pytest.mark.parametrize("n_jobs", [0, 1.5, "2"])
def test_check_n_jobs_invalid(n_jobs):
    """Test that zero and non-integer values raise an informative error."""
    with pytest.raises(ValueError, match="`n_jobs` must"):
        check_n_jobs(n_jobs)

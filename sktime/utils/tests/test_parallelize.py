import copy
import warnings

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.parallel import _get_parallel_test_fixtures, parallelize

_should_run = run_test_module_changed("sktime.utils.parallel")


def square(x, **kwargs):
    return x**2


@pytest.mark.skipif(
    not _should_run,
    reason="sktime.utils.parallel unchanged, skipping parallelize tests",
)
@pytest.mark.parametrize("fixture", _get_parallel_test_fixtures())
def test_parallelize_simple_loop(fixture):
    backend = fixture["backend"]
    backend_params = copy.deepcopy(fixture["backend_params"])
    params_before = copy.deepcopy(fixture["backend_params"])

    nums = range(8)
    expected = [x**2 for x in nums]

    result = parallelize(
        square,
        nums,
        backend=backend,
        backend_params=backend_params,
    )

    assert list(result) == expected
    assert backend_params == params_before


def _warn_for_one_input(x, **kwargs):
    """Return x, but raise a UserWarning when x == 2.

    Module-level by design: process-based joblib backends ("loky",
    "multiprocessing") pickle the target function to send it to worker
    processes, so a closure or local function would fail there.
    """
    if x == 2:
        warnings.warn(
            "expected test warning from _warn_for_one_input",
            UserWarning,
            stacklevel=2,
        )
    return x


@pytest.mark.skipif(
    not _should_run,
    reason="sktime.utils.parallel unchanged, skipping parallelize tests",
)
@pytest.mark.parametrize("fixture", _get_parallel_test_fixtures())
def test_parallelize_warnings_reach_caller(fixture):
    """Warnings raised inside a parallelized call must reach the caller.

    Covers all backends returned by ``_get_parallel_test_fixtures``. The
    "loky" and "multiprocessing" joblib backends run each job in a separate
    process, so only the return value of a job is passed back to the
    caller; a warning raised inside the job does not otherwise propagate.
    "threading" and sequential execution share the caller's process and are
    unaffected. This matters for ``evaluate``, which raises
    ``FitFailedWarning`` from inside a parallelized fold.
    """
    backend = fixture["backend"]
    backend_params = copy.deepcopy(fixture["backend_params"])

    with pytest.warns(UserWarning, match="expected test warning"):
        result = parallelize(
            _warn_for_one_input,
            range(4),
            backend=backend,
            backend_params=backend_params,
        )

    assert list(result) == [0, 1, 2, 3]

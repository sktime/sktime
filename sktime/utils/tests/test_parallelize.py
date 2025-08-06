import copy

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

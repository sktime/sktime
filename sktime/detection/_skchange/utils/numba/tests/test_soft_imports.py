import os
import sys
from contextlib import contextmanager

import pytest


def remove_modules_with_prefix(prefix):
    to_remove = [mod for mod in sys.modules if mod.startswith(prefix)]
    for mod in to_remove:
        del sys.modules[mod]


@contextmanager
def temp_env_and_modules(remove_module_prefix: str, env_vars: dict = None):
    original_modules = sys.modules.copy()
    original_environ = os.environ.copy()

    remove_modules_with_prefix(remove_module_prefix)
    if env_vars is not None:
        os.environ.update(env_vars)
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
        os.environ.clear()
        os.environ.update(original_environ)


def test_setting_wrong_env_variable_raises():
    with (
        temp_env_and_modules(
            remove_module_prefix="skchange", env_vars={"NUMBA_CACHE": "invalid_value"}
        ),
        pytest.raises(ValueError),
    ):
        import sktime.detection._skchange.utils.numba  # noqa: F401, I001


def test_setting_truthy_env_variable_does_not_raise():
    with temp_env_and_modules(
        remove_module_prefix="skchange", env_vars={"NUMBA_FASTMATH": "1"}
    ):
        import sktime.detection._skchange.utils.numba  # noqa: F401, I001

    assert True


def test_setting_falsy_env_variable_does_not_raise():
    with temp_env_and_modules(
        remove_module_prefix="skchange", env_vars={"NUMBA_FASTMATH": "0"}
    ):
        import sktime.detection._skchange.utils.numba  # noqa: F401, I001

    assert True


def test_njit_function():
    from sktime.detection._skchange.utils.numba import njit

    @njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_jit_function():
    from sktime.detection._skchange.utils.numba import jit

    @jit
    def multiply(a, b):
        return a * b

    assert multiply(2, 3) == 6


def test_jit_function_with_args():
    from sktime.detection._skchange.utils.numba import jit

    @jit(cache=True)
    def multiply(a, b):
        return a * b

    assert multiply(2, 3) == 6


def test_njit_function_with_args():
    from sktime.detection._skchange.utils.numba import njit

    @njit(cache=True)
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_prange_function():
    from sktime.detection._skchange.utils.numba import njit, prange

    @njit(parallel=True)
    def sum_prange(n):
        total = 0
        for i in prange(n):
            total += i
        return total

    assert sum_prange(10) == sum(range(10))

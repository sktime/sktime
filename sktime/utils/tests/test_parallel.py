import os
import warnings

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.parallel import parallelize


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.parallel")
    or not _check_soft_dependencies("ray", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
def test_ray_leaves_params_invariant():
    def trial_function(params, meta):
        return params

    backend = "ray"
    backend_params = {
        "mute_warnings": True,
        "ray_remote_args": {"num_cpus": os.cpu_count() - 1},
    }
    # copy for later comparison
    backup = backend_params.copy()

    params = [1, 2, 3]
    meta = {}

    parallelize(trial_function, params, meta, backend, backend_params)

    assert backup == backend_params


def _warn_for_one_input_ray(x, meta=None):
    """Return x, but raise a UserWarning when x == 2.

    Module-level, since ray pickles the target function to send it to a
    remote worker process, where a closure or local function would fail.
    """
    if x == 2:
        warnings.warn(
            "expected test warning from _warn_for_one_input_ray",
            UserWarning,
            stacklevel=2,
        )
    return x


@pytest.mark.skipif(
    not run_test_module_changed("sktime.utils.parallel")
    or not _check_soft_dependencies("ray", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
def test_ray_warnings_reach_caller():
    """Warnings raised inside a ray remote call must reach the caller.

    Same mechanism as the joblib backends in
    ``test_parallelize_warnings_reach_caller``; see
    ``_run_and_capture_warnings`` for why. ``mute_warnings=True`` filters
    warnings before they are raised in the worker, so it continues to
    suppress them rather than forward them.
    """
    # leave at least 1 cpu for ray even on a single-cpu runner
    num_cpus = max(os.cpu_count() - 1, 1)

    backend_params = {
        "mute_warnings": False,
        "ray_remote_args": {"num_cpus": num_cpus},
    }
    with pytest.warns(UserWarning, match="expected test warning"):
        result = parallelize(
            _warn_for_one_input_ray, range(4), {}, "ray", backend_params
        )
    assert list(result) == [0, 1, 2, 3]

    backend_params = {
        "mute_warnings": True,
        "ray_remote_args": {"num_cpus": num_cpus},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = parallelize(
            _warn_for_one_input_ray, range(4), {}, "ray", backend_params
        )
    assert list(result) == [0, 1, 2, 3]

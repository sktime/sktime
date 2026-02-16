import os

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies
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

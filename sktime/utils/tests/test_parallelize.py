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


def _make_recording_parallel():
    """Return a stand-in for ``joblib.Parallel`` and the list it records into.

    The stand-in records the keyword arguments it is constructed with, and runs
    the delayed calls sequentially, so tests do not depend on spawning processes.
    """
    calls = []

    class _RecordingParallel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __call__(self, iterable):
            return [fun(*args, **kwargs) for fun, args, kwargs in iterable]

    return _RecordingParallel, calls


# joblib-family backends, with backend_params that do not contain n_jobs
JOBLIB_NO_N_JOBS = [
    ("loky", None),
    ("loky", {}),
    ("multiprocessing", {}),
    ("threading", {}),
    ("joblib", {"backend": "loky"}),
]


@pytest.mark.skipif(
    not _should_run,
    reason="sktime.utils.parallel unchanged, skipping parallelize tests",
)
@pytest.mark.parametrize("backend, backend_params", JOBLIB_NO_N_JOBS)
def test_parallelize_joblib_default_n_jobs(backend, backend_params, monkeypatch):
    """Test that joblib backends use all cores if n_jobs is not passed.

    Regression test for #5293, where parallelization did not happen
    unless ``n_jobs`` was passed explicitly.
    """
    recording_parallel, calls = _make_recording_parallel()
    # works because _parallelize_joblib imports joblib.Parallel at call time
    monkeypatch.setattr("joblib.Parallel", recording_parallel)
    params_before = copy.deepcopy(backend_params)

    result = parallelize(
        square, range(4), backend=backend, backend_params=backend_params
    )

    assert list(result) == [0, 1, 4, 9]
    assert len(calls) == 1
    assert calls[0]["n_jobs"] == -1
    if backend == "joblib":
        assert calls[0]["backend"] == backend_params["backend"]
    else:
        assert calls[0]["backend"] == backend
    # the default must not be written back into the caller's dict
    assert backend_params == params_before


@pytest.mark.skipif(
    not _should_run,
    reason="sktime.utils.parallel unchanged, skipping parallelize tests",
)
@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_parallelize_joblib_explicit_n_jobs(backend, monkeypatch):
    """Test that an explicitly passed n_jobs is not overridden by the default."""
    recording_parallel, calls = _make_recording_parallel()
    monkeypatch.setattr("joblib.Parallel", recording_parallel)

    result = parallelize(
        square, range(4), backend=backend, backend_params={"n_jobs": 2}
    )

    assert list(result) == [0, 1, 4, 9]
    assert len(calls) == 1
    assert calls[0]["n_jobs"] == 2

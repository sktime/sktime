# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common abstraction utilities for parallelization backends.

New parallelization or iteration backends can be added easily as follows:

* Add a new backend name to ``backend_dict``, syntax is
  backend_name: backend_type, where backend_type collects backend options,
  e.g., multiple options for a single parallelization backend.
* Add a new function to ``para_dict``, should have name
  ``_parallelize_<backend_name>`` and take the same arguments as
  ``_parallelize_none``. Ensure that ``backend`` and ``backend_params`` are arguments,
  even if there is only one backend option, or no additional parameters.
* add the backend string in the docstring of parallelize, and any downstream
  functions that use ``parallelize`` and expose the backend parameter an argument
"""

__author__ = ["fkiraly"]


def parallelize(fun, iter, meta=None, backend=None, backend_params=None):
    """Parallelize loop over iter via backend.

    Executes ``fun(x, meta)`` in parallel for ``x`` in ``iter``,
    and returns the results as a list in the same order as ``iter``.

    Uses the iteration or parallelization backend specified by ``backend``.

    Parameters
    ----------
    fun : callable
        function to be executed in parallel
    iter : iterable
        iterable over which to parallelize
    meta : dict, optional
        variables to be passed to fun
    backend : str, optional
        backend to use for parallelization, one of

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib`` ``Parallel`` loops
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as ``"dask"``, but returns delayed object instead of list

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading":
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``
    """
    if meta is None:
        meta = {}
    if backend is None:
        backend = "None"
    if backend_params is None:
        backend_params = {}

    backend_name = backend_dict[backend]
    para_fun = para_dict[backend_name]

    ret = para_fun(
        fun=fun, iter=iter, meta=meta, backend=backend, backend_params=backend_params
    )
    return ret


backend_dict = {
    "None": "none",
    "loky": "joblib",
    "multiprocessing": "joblib",
    "threading": "joblib",
    "dask": "dask",
    "dask_lazy": "dask",
}
para_dict = {}


def _parallelize_none(fun, iter, meta, backend, backend_params):
    """Execute loop via simple sequential list comprehension."""
    ret = [fun(x, meta=meta) for x in iter]
    return ret


para_dict["none"] = _parallelize_none


def _parallelize_joblib(fun, iter, meta, backend, backend_params):
    """Parallelize loop via joblib Parallel."""
    from joblib import Parallel, delayed

    par_params = backend_params.copy()
    par_params["backend"] = backend

    if "n_jobs" not in par_params:
        par_params["n_jobs"] = -1

    ret = Parallel(**par_params)(delayed(fun)(x, meta=meta) for x in iter)
    return ret


para_dict["joblib"] = _parallelize_joblib


def _parallelize_dask(fun, iter, meta, backend, backend_params):
    """Parallelize loop via dask."""
    from dask import compute, delayed

    lazy = [delayed(fun)(x, meta=meta) for x in iter]
    if backend == "dask":
        return compute(*lazy, **backend_params)
    else:
        return lazy


para_dict["dask"] = _parallelize_dask


def _get_parallel_test_fixtures():
    """Return fixtures for parallelization tests.

    Returns a list of parameter fixtures, where each fixture
    is a dict with keys "backend" and "backend_params".
    """
    from sktime.utils.validation._dependencies import _check_soft_dependencies

    fixtures = []

    # test no parallelization
    fixtures.append({"backend": "None", "backend_params": {}})

    # test joblib backends
    for backend in ["loky", "multiprocessing", "threading"]:
        fixtures.append({"backend": backend, "backend_params": {}})
        fixtures.append({"backend": backend, "backend_params": {"n_jobs": 2}})
        fixtures.append({"backend": backend, "backend_params": {"n_jobs": -1}})

    # test dask backends
    if _check_soft_dependencies("dask", severity="none"):
        fixtures.append({"backend": "dask", "backend_params": {}})
        fixtures.append({"backend": "dask", "backend_params": {"scheduler": "sync"}})

    return fixtures

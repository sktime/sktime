# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common abstraction utilities for parallelization backends.

New parallelization or iteration backends can be added easily as follows:

* Add a new backend name to ``backend_dict``, syntax is
  backend_name: backend_type, where backend_type collects backend options,
  e.g., multiple options for a single parallelization backend.
* Add a new function to ``para_dict``, should have name
  ``_parallelize_<backend_name>`` and take the same arguments as
  ``_parallelize_none``. Ensure that ``backend`` is an argument,
  even if there is only one backend option for the backend_type
* add the backend string in the docstring of parallelize, and any downstream
  functions that use ``parallelize`` and expose the backend parameter an argument
"""

__author__ = ["fkiraly"]


def parallelize(self, fun, iter, meta=None, backend=None):
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
    """
    if meta is None:
        meta = {}
    if backend is None:
        backend = "None"

    backend_name = backend_dict[backend]
    para_fun = para_dict[backend_name]

    ret = para_fun(self, fun=fun, iter=iter, meta=meta, backend=backend)
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


def _parallelize_none(self, fun, iter, meta, backend):
    """Execute loop via simple sequential list comprehension."""
    ret = [fun(x, meta=meta) for x in iter]
    return ret


para_dict["none"] = _parallelize_none


def _parallelize_joblib(self, fun, iter, meta, backend):
    """Parallelize loop via joblib Parallel."""
    from joblib import Parallel, delayed

    ret = Parallel(n_jobs=-1, backend=backend)(delayed(fun)(x, meta=meta) for x in iter)
    return ret


para_dict["joblib"] = _parallelize_joblib


def _parallelize_dask(self, fun, iter, meta, backend):
    """Parallelize loop via dask."""
    from dask import compute, delayed

    lazy = [delayed(fun)(x, meta=meta) for x in iter]
    if backend == "dask":
        return compute(*lazy)
    else:
        return lazy


para_dict["dask"] = _parallelize_dask

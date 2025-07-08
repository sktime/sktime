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

    Executes ``fun(x, meta=meta)`` in parallel for ``x`` in ``iter``,
    and returns the results as a list in the same order as ``iter``.

    Uses the iteration or parallelization backend specified by ``backend``.

    Parameters
    ----------
    fun : callable, must have exactly two arguments, second argument of name "meta"
        function to be executed in parallel

    iter : iterable
        iterable over which to parallelize, elements are passed to fun in order,
        to the first argument

    meta : dict, optional
        variables to be passed to fun, as the second argument, under the key ``meta``

    backend : str, optional
        backend to use for parallelization, one of

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib`` ``Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as ``"dask"``, but returns delayed object instead of list
        - "ray": uses a ray remote to execute jobs in parallel

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

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
    "joblib": "joblib",
    "dask": "dask",
    "dask_lazy": "dask",
    "ray": "ray",
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
    if "backend" not in par_params:
        # if user selects custom joblib backend but does not specify backend explicitly,
        # raise a ValueError
        if backend == "joblib":
            raise ValueError(
                '"joblib" was selected as first layer parallelization backend, '
                "but no backend string was "
                'passed in the backend parameters dict, e.g., "spark". '
                "Please specify a backend to joblib as a key-value pair "
                "in the backend_params arg or the backend:parallel:params config "
                'when using "joblib". '
                'For clarity, "joblib" should only be used for two-layer '
                "backend dispatch, where the first layer is joblib, "
                "and the second layer is a custom backend of joblib, e.g., spark. "
                "For first-party joblib backends, please use the backend string "
                'of sktime directly, e.g., by specifying "multiprocessing" or "loky".'
            )
        # in all other cases, we ensure the backend parameter is one of
        # "loky", "multiprocessing" or "threading", as passed via backend
        else:
            par_params["backend"] = backend
    elif backend != "joblib":
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


def _parallelize_ray(fun, iter, meta, backend, backend_params):
    """Parallelize loop via ray."""
    import logging
    import warnings

    import ray

    # remove the possible excess keys
    logger = logging.getLogger(backend_params.pop("logger_name", None))
    mute_warnings = backend_params.pop("mute_warnings", False)
    shutdown_ray = backend_params.pop("shutdown_ray", True)

    if "ray_remote_args" not in backend_params.keys():
        backend_params["ray_remote_args"] = {}

    @ray.remote  # pragma: no cover
    def _ray_execute_function(
        fun, params: dict, meta: dict, mute_warnings: bool = False
    ):
        if mute_warnings:
            warnings.filterwarnings("ignore")  # silence sktime warnings
        assert ray.is_initialized()
        result = fun(params, meta)
        return result

    if not ray.is_initialized():
        logger.info("Starting Ray Parallel")
        context = ray.init(**backend_params["ray_remote_args"])
        logger.info(
            f"Ray initialized. Open dashboard at http://{context.dashboard_url}"
        )

    # this is to keep the order of results while still using wait to optimize runtime
    refs = [
        _ray_execute_function.remote(fun, x, meta, mute_warnings=mute_warnings)
        for x in iter
    ]
    res_dict = dict.fromkeys(refs)

    unfinished = refs
    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=1)
        res_dict[finished[0]] = ray.get(finished[0])

    if shutdown_ray:
        ray.shutdown()

    res = [res_dict[ref] for ref in refs]
    return res


para_dict["ray"] = _parallelize_ray


# list of backends where we skip tests during CI
SKIP_FIXTURES = [
    "ray",  # unstable, sporadic crashes in CI, see bug 8149
]


def _get_parallel_test_fixtures(naming="estimator"):
    """Return fixtures for parallelization tests.

    Returns a list of parameter fixtures, where each fixture
    is a dict with keys "backend" and "backend_params".

    Parameters
    ----------
    naming : str, optional
        naming convention for the parameters, one of

        "estimator": for use in estimator constructors,
        ``backend`` and ``backend_params``
        "config": for use in ``set_config``,
        ``backend:parallel`` and ``backend:parallel:params``

    Returns
    -------
    fixtures : list of dict
        list of backend parameter fixtures
        keys depend on ``naming`` parameter, see above
        either ``backend`` and ``backend_params`` (``naming="estimator"``),
        or ``backend:parallel`` and ``backend:parallel:params`` (``naming="config"``)
        values are backend strings and backend parameter dicts
        only backends that are available in the environment are included
    """
    from sktime.utils.dependencies import _check_soft_dependencies

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

    # test ray backend
    if _check_soft_dependencies("ray", severity="none"):
        import os

        fixtures.append(
            {
                "backend": "ray",
                "backend_params": {
                    "mute_warnings": True,
                    "ray_remote_args": {"num_cpus": os.cpu_count() - 1},
                },
            }
        )

    fixtures = [x for x in fixtures if x["backend"] not in SKIP_FIXTURES]
    # remove backends in SKIP_FIXTURES from fixtures

    return fixtures

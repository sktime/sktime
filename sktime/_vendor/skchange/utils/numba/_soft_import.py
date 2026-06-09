"""Dispatch njit decorator used to isolate numba.

We wrap the import of `@jit` and `@njit` from `numba` through a check
that sees if `numba` is installed. If not, we provide identity
decorators instead, which return the unjitted functions.

To configure the default arguments to the `@jit` and `@njit` decorators,
we read specific environment variables. If the environment variables are
not set, we use predefined default values.

The default values are:
- `cache=True`
- `fastmath=False`
- `parallel=False`

They are configured by the corresponding environment variables:
- `NUMBA_CACHE`
- `NUMBA_FASTMATH`
- `NUMBA_PARALLEL`

To enable or disable these features, set the environment variables to
`truthy` or `falsy` values.

The `truthy` values are:
- `["", "1", "true", "True", "TRUE"]`

The `falsy` values are:
- `["0", "false", "False", "FALSE"]`

If you're running `skchange` from VS Code, you can set these environment
variables in the `.env` file in the root of the project directory.

Additionally, we provide a `prange` function that dispatches to `numba.prange`.
If `numba` is not installed, it dispatches to the regular Python `range`.

For additional numba configurations, including how to disable numba, see
`https://numba.readthedocs.io/en/stable/reference/envvars.html`_.

The functionality to check for whether or not `numba` is installed
is copied from `sktime.utils.numba.njit`.
"""

from functools import wraps
from os import environ

from sktime.utils.dependencies import _check_soft_dependencies

numba_available = _check_soft_dependencies("numba", severity="none")

if numba_available:
    # The TBB threading layer is not easily available, degrade its priority:
    from numba import config

    config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]


def read_boolean_env_var(name, default_value):
    """Read a boolean environment variable."""
    truthy_strings = ["", "1", "true", "True", "TRUE"]
    falsy_strings = ["0", "false", "False", "FALSE"]

    env_value = environ.get(name)
    if env_value is None:
        return default_value

    if env_value in truthy_strings:
        return True
    elif env_value in falsy_strings:
        return False
    else:
        raise ValueError(
            f"Invalid value for boolean environment variable '{name}': {env_value}"
        )


def define_prange(_):
    """Dispatch prange based on environment variables."""
    if numba_available:
        from numba import prange as numba_prange

        return numba_prange
    else:
        return range


@define_prange
def prange(*args, **kwargs):
    """Dispatch prange based on numba dependency."""
    ...  # pragma: no cover


def configure_jit(jit_default_kwargs):
    """Decorate jit with default kwargs from environment variables."""

    def decorator(_):
        if numba_available:
            from numba import jit as numba_jit

            @wraps(numba_jit)
            def jit(maybe_func=None, **kwargs):
                """Dispatch jit decorator based on environment variables."""
                # This syntax overwrites the default kwargs
                # with the provided kwargs if they overlap.
                kwargs = {**jit_default_kwargs, **kwargs}
                return numba_jit(maybe_func, **kwargs)

        else:

            def jit(maybe_func=None, **kwargs):
                """Identity decorator for replacing jit by passthrough."""
                if callable(maybe_func):
                    # Called with the 'direct' syntax:
                    # @jit
                    # def func(*args, **kwargs):
                    #     ...
                    return maybe_func
                else:
                    # Called with arguments to the decorator:
                    # @jit(cache=True)
                    # def func(*args, **kwargs):
                    #     ...
                    def decorator(func):
                        return func

                    return decorator

        return jit

    return decorator


def configure_njit(njit_default_kwargs):
    """Configure njit with default kwargs from environment variables."""

    def decorator(_):
        if numba_available:
            from numba import njit as numba_njit

            @wraps(numba_njit)
            def njit(maybe_func=None, **kwargs):
                """Dispatch njit decorator based on environment variables."""
                # This syntax overwrites the default kwargs
                # with the provided kwargs if they overlap.
                kwargs = {**njit_default_kwargs, **kwargs}
                return numba_njit(maybe_func, **kwargs)

        else:

            def njit(maybe_func=None, **kwargs):
                """Identity decorator for replacing njit by passthrough."""
                if callable(maybe_func):
                    # Called with the 'direct' syntax:
                    # @njit
                    # def func(*args, **kwargs):
                    #     ...
                    return maybe_func
                else:
                    # Called with arguments to the decorator:
                    # @jit(cache=True)
                    # def func(*args, **kwargs):
                    #     ...
                    def decorator(func):
                        return func

                    return decorator

        return njit

    return decorator


@configure_jit(
    jit_default_kwargs={
        "cache": read_boolean_env_var("NUMBA_CACHE", default_value=True),
        "fastmath": read_boolean_env_var("NUMBA_FASTMATH", default_value=False),
        "parallel": read_boolean_env_var("NUMBA_PARALLEL", default_value=False),
    },
)
def jit():
    """Dispatch jit decorator based on environment variables."""
    ...  # pragma: no cover


@configure_njit(
    njit_default_kwargs={
        "cache": read_boolean_env_var("NUMBA_CACHE", default_value=True),
        "fastmath": read_boolean_env_var("NUMBA_FASTMATH", default_value=False),
        "parallel": read_boolean_env_var("NUMBA_PARALLEL", default_value=False),
    }
)
def njit():
    """Dispatch njit decorator based on environment variables."""
    ...  # pragma: no cover

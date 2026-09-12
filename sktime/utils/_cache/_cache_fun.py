"""Persistent cache for zero-argument functions."""
import functools
import hashlib
import importlib
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar



class PersistentCache:
    """Persistent cache for zero-argument functions.

    This class provides a decorator and utility methods to persistently cache
    the results of zero-argument functions to the filesystem.

    This class has two modes:

    * cache generation via ``PersistentCache.create``. This writes a pickled
      cache file to the filesystem, in the ``.cache`` directory located inside
      the ``module`` directory.
    * cache retrieval via the decorated function. For this, a function
      should be decorated with this cache instance. This reads the pickled cache
      file from the filesystem. If the cache does not exist, an error is raised.

    Parameters
    ----------
    module : str
        The name of the module whose filesystem location will be used to store
        the cache files. Module strings should be importable Python modules,
        in Python module notation (e.g., ``"mypackage.utils.cache"``).

    Examples
    --------
    To use the persistent cache for a zero-argument function ``my_function``:
    >>> cache = PersistentCache("mypackage.utils.cache")
    >>> @cache
    ... def my_function():
    ...     return 42

    The cache can be used only after it has been populated, via
    call ``cache.create(my_function)``. This can be called on the decorated function,
    or on the undecorated function ``my_function``.
    >>> cache.create(my_function)

    The cache needs to be populated only once per function.
    For packaging, ensure that the cache files are included in the distribution.
    """
    def __init__(self, module: str):
        self.module = importlib.import_module(module)

        module_file = getattr(self.module, "__file__", None)
        if module_file is None:
            raise ValueError(f"Module {module!r} has no filesystem location")

        self.cache_dir = Path(module_file).resolve().parent / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

    def __call__(self, func):
        """Decorate a zero-argument function.

        Parameters
        ----------
        func : Callable[[], T]
            The zero-argument function to be decorated.

        Returns
        -------
        Callable[[], T]
            The decorated function that retrieves its result from the persistent cache.
        """
        self._validate(func)
        cache_file = self._cache_file(func)

        @functools.wraps(func)
        def wrapper():
            return self._load(cache_file, func)

        wrapper._persistent_cache_function = func
        wrapper._persistent_cache_file = cache_file

        wrapper.cache_clear = functools.partial(self._clear, cache_file)

        return wrapper

    def create(self, func):
        """
        Explicitly create or refresh the persistent cache for `func`.

        Accepts both the original function and a function decorated with
        this cache instance.
        """
        original = getattr(func, "_persistent_cache_function", func)

        self._validate(original)

        value = original()
        self._write(self._cache_file(original), value)

        return value

    def _load(self, path, func):
        """Load the cached value for `func` from `path`.

        Parameters
        ----------
        path : Path
            The path to the cache file.
        func : Callable[..., Any]
            The zero-argument function whose cached value is to be loaded.
        """
        if not path.exists():
            raise RuntimeError(
                f"No persistent cache exists for "
                f"{func.__module__}.{func.__qualname__}. "
                f"Create it with cache.create({func.__name__})."
            )

        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.PickleError) as exc:
            raise RuntimeError(
                f"Unable to read persistent cache for "
                f"{func.__module__}.{func.__qualname__}"
            ) from exc

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        """Atomically write a pickle file.

        Writes the value ``value`` to the cache file at ``path`` atomically.
        The resulting file is a valid pickle file containing the serialized value.

        Parameters
        ----------
        path : Path
            The path to the cache file.
        value : Any
            The value to be pickled and written to the cache file.
        """

        temporary = path.with_suffix(".tmp")

        pkl_prot = pickle.HIGHEST_PROTOCOL

        try:
            with temporary.open("wb") as f:
                pickle.dump(value, f, protocol=pkl_prot)

            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _clear(path: Path) -> None:
        path.unlink(missing_ok=True)

    def _cache_file(self, func: Callable[..., Any]) -> Path:
        identity = f"{func.__module__}.{func.__qualname__}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]

        return self.cache_dir / f"{func.__name__}-{digest}.pickle"

    @staticmethod
    def _validate(func: Callable[..., Any]) -> None:
        code = func.__code__

        if (
            code.co_argcount
            or code.co_kwonlyargcount
            or code.co_flags & 0x04  # *args
            or code.co_flags & 0x08  # **kwargs
        ):
            raise TypeError(
                f"{func.__qualname__} must be a zero-argument function"
            )

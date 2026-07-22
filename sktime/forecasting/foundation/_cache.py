"""Process-local cache for expensive foundation-model runtime handles.

The cache lets equal estimator instances reuse loaded weights and supporting
objects. It is intentionally small: there is no persistence, eviction policy, or
explicit resource finalization.
"""

from collections.abc import Callable

from sktime.forecasting.foundation._result import ModelHandle


def _make_hashable(obj):
    """Convert common containers to stable hashable equivalents.

    Dictionaries, lists, tuples, and sets are normalized recursively. Other
    unhashable objects fall back to ``repr``; such representations are not
    guaranteed to be stable when they contain object identity. Adapters can
    override ``_get_unique_model_key`` to provide a better representation for
    custom configuration objects.
    """
    if isinstance(obj, dict):
        return tuple((key, _make_hashable(value)) for key, value in sorted(obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(value) for value in obj)
    if isinstance(obj, set):
        return tuple(sorted(_make_hashable(value) for value in obj))
    try:
        hash(obj)
    except TypeError:
        return repr(obj)
    return obj


class _FoundationModelCache:
    """Process-local cache for loaded foundation-model state.

    Handles are shared by identity across estimator instances. The cache has no
    eviction policy and does not synchronize concurrent cache misses, so loaders
    must not rely on exactly-once behavior across threads. Cached objects should
    be reusable and independent of fitted series context.
    """

    def __init__(self):
        self._entries: dict[tuple, ModelHandle] = {}

    def get_or_load(
        self,
        key: tuple,
        loader: Callable[[], ModelHandle],
    ) -> ModelHandle:
        """Return the cached handle for ``key`` or call and store ``loader``.

        Parameters
        ----------
        key : tuple
            Model-loading identity. Nested common containers are accepted and
            converted to hashable equivalents.
        loader : callable returning ModelHandle
            Zero-argument function that constructs reusable backend state.

        Returns
        -------
        ModelHandle
            Existing or newly loaded handle. Callers must treat shared contents
            as model-level state rather than request-specific state.
        """
        key = _make_hashable(key)
        handle = self._entries.get(key)
        if handle is not None:
            return handle

        handle = loader()
        self._entries[key] = handle
        return handle

    def clear(self) -> None:
        """Remove all cache references.

        Fitted estimators may still hold handles, and clearing does not explicitly
        close backend objects or guarantee immediate accelerator-memory release.
        """
        self._entries.clear()


FOUNDATION_MODEL_CACHE = _FoundationModelCache()


def clear_foundation_model_cache() -> None:
    """Clear process-local shared handles, primarily for tests and resource reset."""
    FOUNDATION_MODEL_CACHE.clear()

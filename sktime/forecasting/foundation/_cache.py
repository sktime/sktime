"""Process-local cache for foundation-model runtime handles."""

from collections.abc import Callable

from sktime.forecasting.foundation._result import ModelHandle


def _make_hashable(obj):
    """Convert common containers to hashable, deterministic equivalents."""
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
    """Process-local cache for loaded foundation-model state."""

    def __init__(self):
        self._entries: dict[tuple, ModelHandle] = {}

    def get_or_load(
        self,
        key: tuple,
        loader: Callable[[], ModelHandle],
    ) -> ModelHandle:
        """Return cached handle for ``key`` or load and store it."""
        key = _make_hashable(key)
        handle = self._entries.get(key)
        if handle is not None:
            return handle

        handle = loader()
        self._entries[key] = handle
        return handle

    def clear(self) -> None:
        """Clear all cached handles."""
        self._entries.clear()


FOUNDATION_MODEL_CACHE = _FoundationModelCache()


def clear_foundation_model_cache() -> None:
    """Clear the process-local foundation-model cache."""
    FOUNDATION_MODEL_CACHE.clear()

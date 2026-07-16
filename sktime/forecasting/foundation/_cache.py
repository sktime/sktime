"""Process-local cache for foundation-model runtime handles."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def _stable_repr(obj):
    """Return a stable hashable representation for cache-key components."""
    return _make_hashable(obj)


@dataclass
class _CacheEntry:
    """Internal cache entry."""

    key: tuple
    handle: ModelHandle
    hits: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        """Return diagnostics without exposing model objects."""
        is_named_key = all(
            isinstance(item, tuple) and len(item) == 2 for item in self.key
        )
        if is_named_key:
            key_items = dict(self.key)
            family = key_items.get("class")
            model_path = key_items.get("model_path")
        else:
            family = self.key[0] if self.key else None
            model_path = self.key[1] if len(self.key) > 1 else None

        return {
            "key": self.key,
            "family": family,
            "model_path": model_path,
            "hits": self.hits,
            "shareable": self.handle.shareable,
            "mutable": self.handle.mutable,
        }


class FoundationModelCache:
    """Process-local cache for loaded foundation-model state."""

    def __init__(self):
        self._entries: dict[tuple, _CacheEntry] = {}

    def get_or_load(
        self,
        key: tuple,
        loader: Callable[[], ModelHandle],
        *,
        shareable: bool = True,
    ) -> ModelHandle:
        """Return cached handle for ``key`` or load and store it."""
        if not shareable:
            return loader()

        key = _make_hashable(key)
        entry = self._entries.get(key)
        if entry is not None:
            entry.hits += 1
            return entry.handle

        handle = loader()
        if handle.shareable and not handle.mutable:
            self._entries[key] = _CacheEntry(key=key, handle=handle)
        return handle

    def release(self, key: tuple) -> None:
        """Release one cache entry."""
        key = _make_hashable(key)
        self._entries.pop(key, None)

    def clear(self) -> None:
        """Clear all cached handles."""
        self._entries.clear()

    def info(self) -> list[dict[str, Any]]:
        """Return public cache diagnostics."""
        return [entry.to_public_dict() for entry in self._entries.values()]


FOUNDATION_MODEL_CACHE = FoundationModelCache()


def clear_foundation_model_cache() -> None:
    """Clear the process-local foundation-model cache."""
    FOUNDATION_MODEL_CACHE.clear()


def foundation_model_cache_info() -> list[dict[str, Any]]:
    """Return process-local foundation-model cache diagnostics."""
    return FOUNDATION_MODEL_CACHE.info()

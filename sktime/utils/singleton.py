"""Utilities for implementing singleton and multiton oop pattern."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import threading
from collections import OrderedDict


def _singleton(cls):
    """Turn a class into a singleton."""
    instance = None

    def get_instance(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return get_instance


class _MultitonRegistry:
    """Tracks multiton instances across all ``_multiton``-decorated classes.

    A single instance of this registry is shared by every class created with
    the ``_multiton`` decorator. It records, for each cached instance, which
    per-class cache dict it lives in and under which key, ordered by recency
    of use. When ``maxsize`` is set and the number of tracked instances
    exceeds it, the least recently used instance is evicted (removed from its
    owning cache dict), so it becomes eligible for garbage collection.

    By default ``maxsize`` is ``None``, which preserves the previous
    unbounded caching behaviour.
    """

    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self._entries = OrderedDict()  # token -> (owning instances dict, key)
        self._lock = threading.Lock()

    def touch_or_register(self, token, instances, key):
        """Record use of ``instances[key]``, registering it if new."""
        with self._lock:
            if token in self._entries:
                self._entries.move_to_end(token)
            else:
                self._entries[token] = (instances, key)
                self._evict_if_needed()

    def _evict_if_needed(self):
        # caller already holds self._lock
        if self.maxsize is None:
            return
        while len(self._entries) > self.maxsize:
            _, (instances, key) = self._entries.popitem(last=False)
            instances.pop(key, None)

    def set_maxsize(self, maxsize):
        """Set the global cache size limit, evicting if now over it."""
        with self._lock:
            self.maxsize = maxsize
            self._evict_if_needed()

    def clear(self):
        """Evict every tracked multiton instance from its owning cache."""
        with self._lock:
            for instances, key in self._entries.values():
                instances.pop(key, None)
            self._entries.clear()

    def __len__(self):
        return len(self._entries)


_registry = _MultitonRegistry()


def set_global_multiton_limit(maxsize):
    """Bound the total number of multiton instances kept alive at once.

    Applies across every class created with the ``_multiton`` decorator, not
    just one of them. When the limit is exceeded, the least recently used
    instance is evicted first. Pass ``None`` to remove the limit again.

    Parameters
    ----------
    maxsize : int or None
        Maximum number of multiton instances to keep cached across all
        multiton classes. ``None`` means unbounded (the default).
    """
    _registry.set_maxsize(maxsize)


def clear_all_multitons():
    """Evict every currently cached multiton instance, regardless of class."""
    _registry.clear()


def _multiton(cls):
    """Turn a class into a multiton.

    Instances are cached per ``key`` and registered with a shared global
    registry (see ``set_global_multiton_limit`` and ``clear_all_multitons``)
    so callers can bound or clear the combined cache across all multiton
    classes, not just this one.
    """
    instances = {}

    def get_instance(key, *args, **kwargs):
        if key not in instances:
            instances[key] = cls(key, *args, **kwargs)
        _registry.touch_or_register((cls, key), instances, key)
        return instances[key]

    return get_instance

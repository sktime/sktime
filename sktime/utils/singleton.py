"""Utilities for implementing singleton and multiton oop pattern."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import uuid
from copy import deepcopy


def _singleton(cls):
    """Turn a class into a singleton."""
    instance = None

    def get_instance(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return get_instance


def _multiton(cls):
    """Turn a class into a multiton."""
    instances = {}

    def get_instance(key, *args, **kwargs):
        if key not in instances:
            instances[key] = cls(key, *args, **kwargs)
        return instances[key]

    return get_instance


def _new_unique_key(base_key: str) -> str:
    """Return a detached multiton key derived from a loading-param key."""
    return f"{base_key}|detached:{uuid.uuid4().hex}"


class _MultitonPretrainMixin:
    """Reusable detachment behavior for cache-backed pretrained forecasters.

    Forecasters use ``_get_multiton_cache_class``, ``_get_loading_param_key``,
    and ``_get_multiton_cache_kwargs`` to describe their cache entry. The
    optional ``_get_multiton_model_attr`` hook selects the forecaster model
    attribute; it defaults to ``"model_"``.
    """

    def _get_multiton_model_attr(self):
        """Return the forecaster attribute holding the cached model."""
        return "model_"

    def _get_multiton_cache_model_attr(self):
        """Return the cache-entry attribute holding the loaded model."""
        return "model_"

    def _get_pretrained_reference_attrs(self):
        """Return pretrained attributes that clones/reset preserve by reference."""
        return (self._get_multiton_model_attr(), "_multiton_key")

    def _get_unique_key(self):
        """Return the detached key if set, otherwise the loading-param key."""
        if getattr(self, "_multiton_key", None) is not None:
            return self._multiton_key
        return self._get_loading_param_key()

    def _detach_pretrained_multiton(self):
        """Copy the current model into a uniquely keyed cache entry."""
        model = self._load_model()
        model_attr = self._get_multiton_model_attr()

        new_key = _new_unique_key(self._get_loading_param_key())
        cache_class = self._get_multiton_cache_class()
        cache_entry = cache_class(
            key=new_key,
            **self._get_multiton_cache_kwargs(),
        )
        cache_model_attr = self._get_multiton_cache_model_attr()
        setattr(cache_entry, cache_model_attr, deepcopy(model))

        self._multiton_key = new_key
        setattr(self, model_attr, getattr(cache_entry, cache_model_attr))

        if not hasattr(self, "_pretrained_attrs"):
            self._pretrained_attrs = []
        for attr in self._get_pretrained_reference_attrs():
            if attr not in self._pretrained_attrs:
                self._pretrained_attrs.append(attr)

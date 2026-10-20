"""Utilities for implementing singleton and multiton oop pattern."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

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


def _detach_multiton_model(model):
    """Copy a module loaded from a multiton before in-place training.

    Foundation-model forecasters cache one module per load key. ``pretrain``
    fine-tunes that module in place, which would change every other instance
    that still points at the cache, including zero-shot forecasters. Training
    uses the returned copy; the cached module stays shared and unchanged.

    Parameters
    ----------
    model : model instance
        Module returned by a ``_Cached*`` loader.

    Returns
    -------
    model : private copy of ``model``
    """
    return deepcopy(model)

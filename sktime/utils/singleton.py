"""Utilities for implementing singleton and multiton oop pattern."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)


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


def _detach_from_multiton(obj):
    """Return a private copy of an object held by a multiton.

    All callers that request the same key from a multiton receive the same
    object. An operation that modifies the object in place, such as
    fine-tuning the weights of a model, must therefore act on a private copy,
    otherwise it also changes the object seen by every other instance with the
    same key.

    Parameters
    ----------
    obj : object
        Object returned by a multiton, e.g., a loaded model.

    Returns
    -------
    copy : object
        Deep copy of ``obj``, sharing no state with the multiton.
    """
    from copy import deepcopy

    return deepcopy(obj)

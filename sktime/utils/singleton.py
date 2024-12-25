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

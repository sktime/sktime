# -*- coding: utf-8 -*-
"""Dummy njit decorator used to isolate numba."""


def njit(*args, **kwargs):
    """Identity decorator for replacing njit by passthrough."""

    def decorator(func):
        return func

    return decorator

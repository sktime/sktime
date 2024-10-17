"""Implements base class for customer sample weights of performance metric in sktime."""

from __future__ import annotations

from inspect import signature
from typing import Protocol, runtime_checkable

__author__ = ["markussagen"]
__all__ = ["SampleWeightGenerator", "check_sample_weight_generator"]


@runtime_checkable
class SampleWeightGenerator(Protocol):
    def __call__(self, y_true, y_pred=None, **kwargs):
        ...


def check_sample_weight_generator(obj):
    """Check if obj is a valid SampleWeightGenerator."""
    if obj is None or not callable(obj):
        return False

    sig = signature(obj)
    params = sig.parameters

    # Check if the function has at least one parameter (y_true)
    if len(params) < 1:
        raise ValueError("Sample weight generator must have at least one parameter (y_true)")

    param_names = list(params.keys())

    # Check if the first parameter is y_true
    if param_names[0] != "y_true":
        raise ValueError("First parameter of sample weight generator must be 'y_true'")

    # Check if the function accepts **kwargs
    if not any(param.kind == param.VAR_KEYWORD for param in params.values()):
        raise ValueError("Sample weight generator must accept **kwargs")

    # NOTE: this does not check the return type!

    return True

"""Implements base class for customer sample weights of performance metric in sktime."""

__author__ = ["markussagen"]
__all__ = ["SampleWeightGenerator", "check_sample_weight_generator"]

from inspect import signature
from typing import Protocol, runtime_checkable


@runtime_checkable
class SampleWeightGenerator(Protocol):
    """Protocol for sample weight generators.

    This protocol defines the interface for sample weight generator functions used in
    performance metrics calculations. Sample weight generators are used to assign
    weights to individual time points in time series data.

    The generator must have at least one parameter (y_true) and accept **kwargs.
    It must return a 1D numpy array of sample weights.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
        Estimated target values.
    **kwargs : dict
        Additional keyword arguments that may be used in weight calculation.

    Returns
    -------
    sample_weights : np.ndarray of shape (n_samples,)
        1D array of sample weights.

    Raises
    ------
    ValueError
        If the sample weight generator does not have at least one parameter (y_true).
        If the first parameter is not y_true.
        If the sample weight generator does not accept **kwargs.

    Notes
    -----
    Implementations of this protocol should ensure that the returned weights are
    non-negative and sum to a value greater than zero.

    Examples
    --------
    >>> import numpy as np
    >>> class ExampleWeightGenerator:
    ...     def __call__(self, y_true, y_pred=None, **kwargs):
    ...         return np.ones_like(y_true)
    >>> weight_gen = ExampleWeightGenerator()
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> weights = weight_gen(y_true)
    >>> print(weights)
    [1. 1. 1. 1. 1.]
    """

    def __call__(self, y_true, y_pred=None, **kwargs):
        """Generate sample weights for time series data.

        This method should be implemented by concrete sample weight generators.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            Estimated target values. Default is None.
        **kwargs : dict
            Additional keyword arguments that may be used in weight calculation.

        Returns
        -------
        sample_weights : np.ndarray of shape (n_samples,)
            1D array of sample weights.

        Notes
        -----
        - The method should always return a 1D numpy array of weights.
        - The length of the returned array should match the number of samples in y_true.
        - If y_pred is not used in the weight calculation, it should be ignored.
        """
        ...


def check_sample_weight_generator(obj):
    """Check if obj is a valid SampleWeightGenerator."""
    if obj is None or not callable(obj):
        return False

    sig = signature(obj)
    params = sig.parameters

    # Check if the function has at least one parameter (y_true)
    if len(params) < 1:
        msg = "Sample weight generator must have at least one parameter (y_true)"
        raise ValueError(msg)

    param_names = list(params.keys())

    # Check if the first parameter is y_true
    if param_names[0] != "y_true":
        raise ValueError("First parameter of sample weight generator must be 'y_true'")

    # Check if the function accepts **kwargs
    if not any(param.kind == param.VAR_KEYWORD for param in params.values()):
        raise ValueError("Sample weight generator must accept **kwargs")

    # NOTE: this does not check the return type!

    return True

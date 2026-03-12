"""Utility functions for handling neural network activation functions.

This module provides a centralized way to instantiate and manage PyTorch activation
functions, supporting both string names and direct torch.nn.Module instances.
"""

__all__ = ["instantiate_activation"]

from sktime.utils.dependencies import _safe_import

# Soft dependency for Torch
NNModule = _safe_import("torch.nn.Module")


def instantiate_activation(activation, activation_kwargs={}):
    """Instantiate an activation function from a string or return module as-is.

    This utility function supports backward compatibility across all network
    implementations by providing a single point of activation instantiation.

    Parameters
    ----------
    activation : str, torch.nn.Module, or None
        The activation function to instantiate.
        - If str: activation name (case-insensitive)
        - If torch.nn.Module: returned as-is
        - If None: returns None
    activation_kwargs : dict, optional
        Additional keyword arguments to pass when instantiating the activation function.
        This is only applicable if `activation` is a string. Ignored otherwise.

    Returns
    -------
    activation_function : torch.nn.Module or None
        An instantiated PyTorch activation module, or None if activation is None.

    Raises
    ------
    TypeError
        If activation is not a string, torch.nn.Module, or None.
    ValueError
        If activation is a string but not recognized in the supported list.

    Notes
    -----
    Supported activation functions include:
    - Output layer activations: sigmoid, softmax, logsoftmax, logsigmoid, relu
    - Hidden layer activations: relu, leakyrelu, elu, prelu, gelu, selu, rrelu,
      celu, tanh, hardtanh (plus all output layer activations)
    """
    if activation is None:
        return None

    if isinstance(activation, NNModule):
        return activation

    if not isinstance(activation, str):
        raise TypeError(
            f"`activation` should be of type str, torch.nn.Module, or None. "
            f"But found the type to be: {type(activation)}"
        )

    act_lower = activation.lower()

    activation_map = {
        "relu": ("torch.nn.ReLU", {}),  # activation_name: (module_path, default_kwargs)
        "sigmoid": ("torch.nn.Sigmoid", {}),
        "tanh": ("torch.nn.Tanh", {}),
        "softmax": ("torch.nn.Softmax", {"dim": 1}),
        "logsoftmax": ("torch.nn.LogSoftmax", {"dim": 1}),
        "logsigmoid": ("torch.nn.LogSigmoid", {}),
        "leakyrelu": ("torch.nn.LeakyReLU", {}),
        "leaky_relu": ("torch.nn.LeakyReLU", {}),
        "elu": ("torch.nn.ELU", {}),
        "prelu": ("torch.nn.PReLU", {}),
        "gelu": ("torch.nn.GELU", {}),
        "selu": ("torch.nn.SELU", {}),
        "rrelu": ("torch.nn.RReLU", {}),
        "celu": ("torch.nn.CELU", {}),
        "hardtanh": ("torch.nn.Hardtanh", {}),
    }

    if act_lower not in activation_map:
        supported = sorted(set(activation_map.keys()))
        raise ValueError(
            f"Activation '{activation}' is not supported. "
            f"Supported activations are: {supported}"
        )

    module_path, kwargs = activation_map[act_lower]
    activation_class = _safe_import(module_path)
    if not activation_kwargs:
        activation_kwargs = kwargs
    return activation_class(**activation_kwargs)

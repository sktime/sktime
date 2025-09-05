"""Abstract base class for deep learning networks."""

__author__ = ["RecreationalMath"]

from abc import abstractmethod

from sktime.base import BaseObject


class BasePytorchDeepNetwork(BaseObject):
    """Abstract base class for PyTorch deep learning networks."""

    # do we need to have a separate tag for TensorFlow networks and PyTorch networks?
    # Need to ask maintainers
    _tags = {"object_type": "network"}

    @abstractmethod
    def forward(self, input, **kwargs):
        """Define the forward pass in the network.

        Parameters
        ----------
        input : tensor
            The input data fed into the input layer

        Returns
        -------
        output : tensor
            The output data produced by the network,
            it could represent predictions, features, 
            or any transformed data depending on the network's purpose 
            (e.g., logits for classification, 
            reconstructed data for autoencoders, etc.). 
            The exact shape and meaning of the output tensor 
            depend on the network architecture and the task.
        """
        ...

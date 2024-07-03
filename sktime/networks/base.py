"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall"]

from sktime.base import BaseObject


class BaseDeepNetwork(BaseObject):
    """Abstract base class for deep learning networks."""

    _tags = {"object_type": "network"}

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        raise RuntimeError("abstract method")

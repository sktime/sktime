"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall"]

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sktime.base import BaseObject
from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch


class BaseDeepNetwork(BaseObject, ABC):
    """Abstract base class for deep learning networks."""

    _tags = {"object_type": "network"}

    @abstractmethod
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
        ...

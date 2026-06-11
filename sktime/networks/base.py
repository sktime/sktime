"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall", "fkiraly"]

from sktime.base import BaseObject
from sktime.utils.dependencies import _check_estimator_deps


class BaseDeepNetwork(BaseObject):
    """Abstract base class for deep learning networks."""

    _tags = {
        "object_type": "network",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(self):
        super().__init__()

        # this block has a double purpose:
        # - emit a warning if dependencies are not met, but allow instantiation
        # - if dependencies are met, call __post_init__ used by inheriting classes
        if _check_estimator_deps(self, severity="warning"):
            self.__post_init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * dynamic tag setting
        * any soft dependency imports in the constructor
        """
        pass

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
        raise NotImplementedError("abstract method")

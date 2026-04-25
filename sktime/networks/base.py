"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall"]

from abc import abstractmethod

from sktime.base import BaseObject


class BaseDeepNetwork(BaseObject):
    """Abstract base class for deep learning networks."""

    _tags = {
        "object_type": "network",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(self):
        super().__init__()

        from sktime.utils.dependencies import (
            _check_estimator_deps,
            _check_soft_dependencies,
        )

        _check_estimator_deps(self)
        recommended = self.get_class_tag(
            "python_dependencies_recommended", tag_value_default=None
        )
        if recommended is not None:
            _check_soft_dependencies(recommended, severity="warning", obj=self)

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

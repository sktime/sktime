"""Base class for bootstrap algorithms."""

from sktime.base import BaseObject


class BaseBootstrap(BaseObject):
    """Base class for bootstrap resampling algorithms."""

    _tags = {
        "object_type": "bootstrap",
        "capability:bootstrap_index": True,
    }

    def clone(self):
        """Return a clone of self."""
        return self.__class__(**self.get_params())

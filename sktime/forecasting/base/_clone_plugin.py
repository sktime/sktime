# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Clone plugin for preserving pretrained state in forecasters."""

from copy import deepcopy

from skbase.base._clone_plugins import BaseCloner, _default_clone


class _PretrainedCloner(BaseCloner):
    """Clone plugin that preserves pretrained state.

    Inherits from skbase's ``BaseCloner`` and only implements
    the two required methods: ``_check`` and ``_clone``.

    This plugin checks if an object has pretrained attributes
    (tracked in ``_pretrained_attrs``) and copies them to the clone,
    preserving the pretrained state across clone operations.

    Used in cross-validation and other scenarios where forecasters
    are cloned but pretrained state should be preserved.
    """

    def _check(self, obj):
        """Check if obj has pretrained attributes that should be preserved.

        This method checks both:
        1. Class capability: The object's class declares ``capability:pretrain=True``
        2. Instance state: The object was actually pretrained (has pretrained attrs)

        Both conditions are required because:
        - Tag alone would cause false positives for new (unfitted) estimators
        - Runtime checks alone could trigger on non-pretrain estimators if
          ``_pretrained_attrs`` somehow got set incorrectly
        """
        has_pretrain_capability = obj.get_tag(
            "capability:pretrain", tag_value_default=False, raise_error=False
        )
        has_pretrained_state = (
            hasattr(obj, "_pretrained_attrs")
            and obj._pretrained_attrs
            and hasattr(obj, "_state")
            and obj._state in ("pretrained", "fitted")
        )
        return bool(has_pretrain_capability and has_pretrained_state)

    def _clone(self, obj):
        """Clone obj and preserve pretrained attributes."""
        # First, do the standard clone (copies hyperparameters)
        new_object = _default_clone(estimator=obj, recursive_clone=self.recursive_clone)
        if obj.get_config()["clone_config"]:
            new_object.set_config(**obj.get_config())
        new_object._pretrained_attrs = list(obj._pretrained_attrs)
        for attr in obj._pretrained_attrs:
            if hasattr(obj, attr):
                setattr(new_object, attr, deepcopy(getattr(obj, attr)))

        new_object._state = "pretrained"

        return new_object

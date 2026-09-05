# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Clone plugin for preserving pretrained state in forecasters."""

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
        from copy import deepcopy

        from sktime.utils.torch_utils import (
            clone_state_dict,
            is_torch_module,
            load_state_dict_into,
        )

        # First, do the standard clone (copies hyperparameters)
        new_object = _default_clone(estimator=obj, recursive_clone=self.recursive_clone)
        if obj.get_config()["clone_config"]:
            new_object.set_config(**obj.get_config())

        new_object._pretrained_attrs = list(obj._pretrained_attrs)
        for attr in obj._pretrained_attrs:
            if hasattr(obj, attr):
                obj_attr = getattr(obj, attr)
            # Use state_dict cloning for nn.Module, deepcopy for others
            if is_torch_module(obj_attr):
                try:
                    state_dict = clone_state_dict(obj_attr)
                    cloned_obj = obj_attr.__class__()  # Create empty instance
                    load_state_dict_into(cloned_obj, state_dict)
                    setattr(new_object, attr, cloned_obj)
                except (RuntimeError, AttributeError, TypeError):
                    setattr(new_object, attr, deepcopy(obj_attr))
            else:
                setattr(new_object, attr, deepcopy(obj_attr))

        new_object._state = "pretrained"
        return new_object

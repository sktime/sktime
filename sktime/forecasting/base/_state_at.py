# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixin with state-aware private variants of reset, set_params, and clone."""

__author__ = ["SimonBlanke"]

from collections import defaultdict

from skbase.base import BaseObject
from skbase.base._clone_base import _check_clone, _clone

_VALID_TARGET_STATES = ("new", "pretrained")


class _StateAtMixin:
    """State-aware private variants of reset, set_params and clone.

    Mixin for ``BaseForecaster``, later also ``BaseTransformer``. Provides
    ``_reset_at(state)``, ``_set_params_at(state, params)`` and
    ``_clone_at(state)``. ``state`` is an upper bound on the retained state
    tier, not a guarantee: estimators without pretrain capability or without
    pretrained state degrade gracefully to the behavior of the public
    methods.
    """

    @staticmethod
    def _check_target_state(state):
        """Raise ValueError if state is not a valid target state."""
        if state not in _VALID_TARGET_STATES:
            raise ValueError(
                f"target state must be one of {_VALID_TARGET_STATES}, "
                f"but found {state!r}"
            )

    def _has_pretrain_capability(self):
        """Return whether the estimator declares capability:pretrain."""
        return self.get_tag(
            "capability:pretrain", tag_value_default=False, raise_error=False
        )

    def _get_protected_pretrained_attrs(self):
        """Return names of attributes protected by state-aware operations.

        Source of truth is the ``pretrain:fitted_params`` tag. If the tag
        is empty or not set, falls back to the runtime list
        ``_pretrained_attrs`` auto-registered by ``pretrain``.
        """
        tag_attrs = self.get_tag(
            "pretrain:fitted_params", tag_value_default=None, raise_error=False
        )
        if tag_attrs:
            return list(tag_attrs)
        return list(getattr(self, "_pretrained_attrs", None) or [])

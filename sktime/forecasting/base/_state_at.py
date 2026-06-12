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

    def _save_pretrained_state(self):
        """Save protected pretrained attributes before an in-place reset."""

        if not getattr(self, "_pretrained_attrs", None):
            return {}
        attr_names = self._get_protected_pretrained_attrs()
        saved = {a: getattr(self, a) for a in attr_names if hasattr(self, a)}
        if saved:
            # restore a runtime list consistent with the surviving attrs,
            # so get_pretrained_params and the clone plugin keep working
            saved["_pretrained_attrs"] = [a for a in attr_names if a in saved]
        return saved

    def _restore_pretrained_state(self, saved):
        """Restore saved pretrained attributes after an in-place reset."""
        for attr, value in saved.items():
            setattr(self, attr, value)
        if saved:
            # reset removed task-fitted state; with pretrained attrs
            # restored, the estimator is in the "pretrained" state tier
            self._state = "pretrained"

    def _reset_at(self, state):
        """Reset the estimator, retaining state up to ``state``.

        State-aware variant of ``reset``. ``reset`` itself is unchanged
        and always resets to the post-init state.

        Parameters
        ----------
        state : str, "pretrained" or "new"
            Upper bound on the retained state tier.

            * ``"pretrained"``: attributes named by the
              ``pretrain:fitted_params`` tag (fallback: runtime
              ``_pretrained_attrs``) are restored after the reset, ending
              in state ``"pretrained"``. Estimators without pretrain
              capability or without pretrained state degrade to a full
              reset, ending in state ``"new"``.
            * ``"new"``: identical to ``reset``.

        Returns
        -------
        self : reference to self, reset in-place.

        Raises
        ------
        ValueError
            If ``state`` is not one of ``"pretrained"``, ``"new"``.
        """
        self._check_target_state(state)
        if not self._has_pretrain_capability() or state == "new":
            return self.reset()
        saved = self._save_pretrained_state()
        self.reset()
        self._restore_pretrained_state(saved)
        return self

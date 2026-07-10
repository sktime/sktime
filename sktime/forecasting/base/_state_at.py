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

    def _clone_at(self, state):
        """Clone the estimator, retaining state up to ``state``.

        State-aware variant of ``clone``.

        Parameters
        ----------
        state : str, "pretrained" or "new"
            Upper bound on the retained state tier.

            * ``"pretrained"``: equivalent to ``clone()``. The
              ``_PretrainedCloner`` clone plugin copies pretrained
              attributes when present; without pretrained state this
              degrades to a blank clone.
            * ``"new"``: blank clone in post-init state, bypassing the
              pretrained-state clone plugin. Only this estimator's own
              plugin is bypassed; pretrained state held by component
              estimators is governed by their own clone behavior.

        Returns
        -------
        clone : new object of same type, without shared references to self.

        Raises
        ------
        ValueError
            If ``state`` is not one of ``"pretrained"``, ``"new"``.
        """
        self._check_target_state(state)
        if not self._has_pretrain_capability() or state == "pretrained":
            return self.clone()
        self_clone = _clone(self, base_cls=BaseObject)
        if self.get_config()["check_clone"]:
            _check_clone(original=self, clone=self_clone)
        return self_clone

    def _set_params_at(self, state, params):
        """Set parameters, resetting only state above ``state``.

        State-aware variant of ``set_params``. Mostly a copy of skbase
        ``BaseObject.set_params``, with two differences: it calls
        ``self._reset_at(state)`` instead of ``self.reset()``, and it
        passes ``state`` on to components that implement
        ``_set_params_at``. Components without the method receive a plain
        ``set_params`` call; the state chain ends there by design.

        Deliberately no early fallback on the capability tag here:
        composites are usually not pretrain capable themselves, but must
        still pass the state on to capable components. For estimator
        trees without any pretrain-capable node, behavior is identical to
        ``set_params``, because ``_reset_at`` falls back to ``reset``.

        Parameters
        ----------
        state : str, "pretrained" or "new"
            Upper bound on the retained state tier, see ``_reset_at``.
        params : dict
            Parameters as in ``set_params``, passed as an explicit dict.
            ``<component>__<parameter>`` keys address components.

        Returns
        -------
        self : reference to self, with parameters set.

        Raises
        ------
        ValueError
            If ``state`` is invalid, or parameter keys do not resolve.
        """
        self._check_target_state(state)
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        unmatched_keys = []

        nested_params = defaultdict(dict)
        for full_key, value in params.items():
            key, delim, sub_key = full_key.partition("__")
            if key not in valid_params:
                unmatched_keys += [key]
            elif delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        self._reset_at(state)

        for key, sub_params in nested_params.items():
            component = valid_params[key]
            if hasattr(component, "_set_params_at"):
                component._set_params_at(state, sub_params)
            else:
                component.set_params(**sub_params)

        if len(unmatched_keys) > 0:
            valid_params = self.get_params(deep=True)
            unmatched_params = {key: params[key] for key in unmatched_keys}
            aliased_params = self._alias_params(unmatched_params, valid_params)

            if set(aliased_params) == set(unmatched_params):
                raise ValueError(
                    "Invalid parameter keys provided to _set_params_at of "
                    f"object {self}. Check the list of available parameters "
                    "with `object.get_params().keys()`. "
                    f"Invalid keys provided: {unmatched_keys}"
                )

            self._set_params_at(state, aliased_params)

        return self

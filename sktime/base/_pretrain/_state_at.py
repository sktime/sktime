# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixin with state-aware private variants of reset, set_params, and clone."""

__author__ = ["SimonBlanke"]

from collections import defaultdict

from skbase.base import BaseObject
from skbase.base._clone_base import _check_clone, _clone

from sktime.base._base import _safe_clone
from sktime.base._pretrain._clone_plugin import _PretrainedCloner

_VALID_TARGET_STATES = ("new", "pretrained")


def _clone_at(obj, state):
    """Clone any object, retaining state up to ``state`` where supported.

    For use by composites on their components. Calls ``obj._clone_at(state)``
    if ``obj`` implements it, otherwise falls back to ``_safe_clone(obj)``,
    i.e., ``obj.clone()``, or sklearn ``clone`` for non-``sktime`` objects.

    Parameters
    ----------
    obj : object to clone, e.g., an estimator held by a composite
    state : str, "pretrained" or "new"
        Upper bound on the retained state tier, see ``_StateAtMixin._clone_at``.

    Returns
    -------
    clone : new object of same type, without shared references to ``obj``.

    Raises
    ------
    ValueError
        If ``state`` is not one of ``"pretrained"``, ``"new"``.
    """
    _StateAtMixin._check_target_state(state)
    if hasattr(obj, "_clone_at"):
        return obj._clone_at(state)
    return _safe_clone(obj)


def _set_params_at(obj, state, params):
    """Set parameters of any object, retaining state up to ``state`` if supported.

    For use by composites on their components. Calls
    ``obj._set_params_at(state, params)`` if ``obj`` implements it, otherwise
    falls back to ``obj.set_params(**params)``.

    Parameters
    ----------
    obj : object to set parameters of, e.g., an estimator held by a composite
    state : str, "pretrained" or "new"
        Upper bound on the retained state tier, see ``_StateAtMixin._reset_at``.
    params : dict
        Parameters as in ``set_params``, passed as an explicit dict.

    Returns
    -------
    obj : reference to ``obj``, with parameters set.

    Raises
    ------
    ValueError
        If ``state`` is not one of ``"pretrained"``, ``"new"``.
    """
    _StateAtMixin._check_target_state(state)
    if hasattr(obj, "_set_params_at"):
        return obj._set_params_at(state, params)
    return obj.set_params(**params)


class _StateAtMixin:
    """State-aware private variants of reset, set_params and clone.

    Used via ``_PretrainMixin`` by estimator base classes. Provides
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
              pretrained-state clone plugin ``_PretrainedCloner``. All other
              clone plugins from ``_get_clone_plugins`` still apply. As in
              ``clone``, the plugins are also used for components, so
              pretrained state held by component estimators is dropped as
              well, and the target state applies to the whole estimator tree.

        Returns
        -------
        clone : new object of same type, without shared references to self.

        Raises
        ------
        ValueError
            If ``state`` is not one of ``"pretrained"``, ``"new"``.
        """
        self._check_target_state(state)
        if state == "pretrained":
            return self.clone()
        # Deliberately no early fallback on the capability tag here:
        # composites are usually not pretrain capable themselves, but must
        # still pass the target state on to capable components.
        # For estimator trees without any pretrain-capable node, the result is
        # identical to ``clone``, since ``_PretrainedCloner`` only applies to
        # pretrain-capable estimators.
        clone_plugins = [
            plugin
            for plugin in (self._get_clone_plugins() or [])
            if plugin is not _PretrainedCloner
        ]
        self_clone = _clone(self, base_cls=BaseObject, clone_plugins=clone_plugins)
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

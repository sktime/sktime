# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixin with pretrain state logic shared by estimator base classes."""

__author__ = ["SimonBlanke", "srupat"]

from collections import defaultdict

from skbase.base import BaseObject
from skbase.base._clone_base import _check_clone, _clone

from sktime.base._pretrain._clone_plugin import _PretrainedCloner
from sktime.base._pretrain._state_at import _check_target_state


class _PretrainMixin:
    """Pretrain state logic shared by estimator base classes.

    Provides the ``state`` property, the ``_is_fitted`` compatibility
    property, ``get_pretrained_params``, the ``_PretrainedCloner`` clone
    plugin, ``_pretrain_and_track``, which runs ``_pretrain`` or
    ``_pretrain_update`` and records the attributes these set, and the
    state-aware private variants of reset, set_params and clone:
    ``_reset_at(state)``, ``_set_params_at(state, params)`` and
    ``_clone_at(state)``. For these, ``state`` is an upper bound on the
    retained state tier, not a guarantee: estimators without pretrain
    capability or without pretrained state degrade gracefully to the
    behavior of the public methods.

    The public ``pretrain`` method, including input checks and conversion,
    is left to the inheriting base class, as its signature is scitype specific.
    Inheriting classes must set ``self._state = "new"`` in ``__init__``.
    """

    @classmethod
    def _get_clone_plugins(cls):
        """Get clone plugins for the estimator.

        Overrides the default skbase clone behavior to preserve
        pretrained attributes when cloning estimators.

        The ``_PretrainedCloner`` plugin ensures that when an estimator
        with pretrained state is cloned (e.g., during cross-validation),
        the pretrained attributes are copied to the clone. Estimators without
        the ``capability:pretrain`` tag are cloned as usual.

        Returns
        -------
        list
            List containing ``_PretrainedCloner`` plugin class.
        """
        return [_PretrainedCloner]

    @property
    def _is_fitted(self):
        """Internal fitted state for backward compatibility with skbase.

        Same semantics as ``is_fitted``: returns True only after ``fit()``.
        For pretrain state checks, use ``self.state`` directly.

        Returns
        -------
        bool
            True if the estimator has been fitted, False otherwise.
        """
        return self._state == "fitted"

    @_is_fitted.setter
    def _is_fitted(self, value):
        """Setter for backward compatibility.

        Parameters
        ----------
        value : bool
            If True, sets state to "fitted". If False, sets state to "new".
        """
        if value:
            self._state = "fitted"
        else:
            self._state = "new"

    @property
    def state(self):
        """State of the estimator.

        Possible states are:

        * "new": post-init state
        * "pretrained": after ``pretrain`` has been called, until ``fit`` is called
        * "fitted": after ``fit`` has been called

        Returns
        -------
        str, one of {"new", "pretrained", "fitted"}
            State of the estimator.
        """
        return self._state

    def _pretrain_and_track(self, **kwargs):
        """Run ``_pretrain`` or ``_pretrain_update``, track new attributes.

        Clears fitted state while preserving pretrained attributes, then calls
        ``_pretrain`` if the state is ``"new"``, otherwise ``_pretrain_update``.
        Public attributes ending in ``"_"`` that are
        created by the call are appended to ``self._pretrained_attrs``, and the
        state is set to ``"pretrained"``.

        Parameters
        ----------
        kwargs : dict
            Inner data arguments, passed on to ``_pretrain``
            or ``_pretrain_update``.

        Returns
        -------
        self : reference to self
        """
        if self._state == "fitted":
            self._reset_at("pretrained")
        prior_attrs = {
            a for a in dir(self) if a.endswith("_") and not a.startswith("_")
        }
        if self._state == "new":
            self._pretrain(**kwargs)
        else:
            self._pretrain_update(**kwargs)

        if not hasattr(self, "_pretrained_attrs"):
            self._pretrained_attrs = []

        # Track new pretrained attributes (extend, not append, to avoid nested lists)
        new_attrs = [
            a
            for a in dir(self)
            if a.endswith("_")
            and not a.startswith("_")
            and a not in self._pretrained_attrs
            and a not in prior_attrs
        ]
        self._pretrained_attrs.extend(new_attrs)

        self._state = "pretrained"
        return self

    def get_pretrained_params(self, deep=True):
        """Get pretrained parameters of this estimator.

        Returns a dictionary of attributes that were set during ``pretrain``.
        These are attributes ending in ``"_"`` that appeared on the estimator
        after ``pretrain`` was called, and are tracked separately from the
        fitted parameters set by ``fit``.

        State required:
            Requires state to be ``"pretrained"`` or ``"fitted"``
            (after pretraining). Returns an empty dict if the estimator
            has not been pretrained, or does not have the
            ``capability:pretrain`` tag.

        Parameters
        ----------
        deep : bool, default=True
            Whether to return pretrained parameters of nested estimators.

            * If True, will return a dict of parameter name : value for this
              object, including pretrained parameters of nested estimators.
            * If False, will return a dict of parameter name : value for this
              object, but not include pretrained parameters of nested
              estimators.

        Returns
        -------
        params : dict
            Dictionary of pretrained parameter names mapped to their values.
            Keys are attribute names ending in ``"_"`` that were set during
            pretraining. Returns empty dict if estimator has not been
            pretrained.

            If ``deep=True``, also contains keys/value pairs of nested
            estimators' pretrained parameters, indexed as
            ``[attrname]__[paramname]``.

        See Also
        --------
        pretrain : Pretrain estimator on a collection of time series.
        get_fitted_params : Get parameters set during ``fit``.

        Examples
        --------
        >>> from sktime.forecasting.dummy_global import DummyGlobalForecaster
        >>> from sktime.utils._testing.hierarchical import _make_hierarchical
        >>> y_panel = _make_hierarchical(
        ...     hierarchy_levels=(3,), min_timepoints=12, max_timepoints=12
        ... )
        >>> forecaster = DummyGlobalForecaster()
        >>> forecaster.pretrain(y_panel)
        DummyGlobalForecaster()
        >>> params = forecaster.get_pretrained_params()
        >>> sorted(params.keys())  # doctest: +ELLIPSIS
        ['global_mean_', ..., 'n_pretrain_timepoints_']
        """
        if not hasattr(self, "_pretrained_attrs"):
            return {}

        params = {}
        for attr in self._pretrained_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                params[attr] = value

                # Handle nesting: if value is an estimator with pretrained params
                if deep and hasattr(value, "get_pretrained_params"):
                    nested = value.get_pretrained_params(deep=True)
                    for nested_key, nested_val in nested.items():
                        params[f"{attr}__{nested_key}"] = nested_val

        return params

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
        _check_target_state(state)
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
        _check_target_state(state)
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
        _check_target_state(state)
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

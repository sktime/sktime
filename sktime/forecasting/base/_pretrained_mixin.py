# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixin for preserving pretrained state across reset and set_params.

This module exists as a staging bridge (sktime-only). It overrides
``reset`` and ``set_params`` from skbase's ``BaseObject`` because skbase does
not yet provide hooks for non-resettable attributes. When skbase gains native
support, this mixin can be removed from BaseForecaster's inheritance
list and the file deleted.

See https://github.com/sktime/skbase/issues/554 for the upstream issue.
"""

__author__ = ["simonblanke"]

from collections import defaultdict


class _PretrainedStateMixin:
    """Preserve pretrained state across ``reset`` and ``set_params``.

    Temporary mixin for ``BaseForecaster``. Overrides ``reset`` and
    ``set_params`` inherited from skbase to save and restore attributes
    tracked in ``_pretrained_attrs``. Intended to be removed once skbase
    provides equivalent hooks natively.
    """

    def _has_pretrained_state(self):
        """Return whether this instance has pretrained state to preserve."""
        has_pretrain_capability = self.get_tag(
            "capability:pretrain", tag_value_default=False, raise_error=False
        )
        has_pretrained_state = (
            hasattr(self, "_pretrained_attrs")
            and self._pretrained_attrs
            and hasattr(self, "_state")
            and self._state in ("pretrained", "fitted")
        )
        return bool(has_pretrain_capability and has_pretrained_state)

    def _save_pretrained_state(self):
        """Save pretrained attributes before an in-place reset."""
        if not self._has_pretrained_state():
            return {}

        attr_names = list(self._pretrained_attrs)
        saved = {
            attr: getattr(self, attr) for attr in attr_names if hasattr(self, attr)
        }
        saved["_pretrained_attrs"] = attr_names
        return saved

    def _restore_pretrained_state(self, attrs):
        """Restore pretrained attributes after an in-place reset."""
        for attr, value in attrs.items():
            setattr(self, attr, value)
        if attrs.get("_pretrained_attrs"):
            # reset removes fitted/task-specific state, so a fitted pretrained
            # forecaster must come back as pretrained rather than fitted.
            self._state = "pretrained"

    def reset(self, keep_pretrained=True):
        """Reset object while preserving pretrained state by default.

        Wraps skbase's ``BaseObject.reset`` to save pretrained attributes
        before the reset wipes all instance state, and restores them
        afterwards.

        Parameters
        ----------
        keep_pretrained : bool, default=True
            If True, pretrained attributes tracked in ``_pretrained_attrs``
            are restored after reset. If False, pretrained state is
            discarded and reset behaves like the generic skbase reset.

        Returns
        -------
        self : reference to self
            Reset instance.
        """
        pretrained_state = self._save_pretrained_state() if keep_pretrained else {}
        super().reset()
        if pretrained_state:
            self._restore_pretrained_state(pretrained_state)
        return self

    def set_params(self, **params):
        """Set parameters, optionally controlling pretrained state reset.

        Reimplements skbase's ``BaseObject.set_params`` to support the
        ``_reset`` control flag. When ``_reset=False``, parameter values
        are written via ``setattr`` without calling ``reset``, preserving
        all current state including fitted attributes. The flag propagates
        to nested forecasters.

        When ``_reset=True`` (the default), behaviour is identical to the
        inherited ``set_params`` except that ``self.reset()`` calls this
        mixin's override which preserves pretrained state.

        Passing ``_keep_pretrained=False`` discards pretrained state when
        ``_reset=True``.

        Parameters
        ----------
        **params : dict
            Object parameters. If ``_reset=False`` is passed, parameter
            values are set without calling ``reset`` on this object or
            nested components. If ``_keep_pretrained=False`` is passed,
            pretrained attributes are discarded during reset.

        Returns
        -------
        self : reference to self
            Instance with updated parameters.
        """
        reset = params.pop("_reset", True)
        keep_pretrained = params.pop("_keep_pretrained", True)

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

        if reset:
            self.reset(keep_pretrained=keep_pretrained)

        for key, sub_params in nested_params.items():
            component = valid_params[key]
            if hasattr(component, "set_params"):
                # propagate _reset only to objects likely to understand it
                if isinstance(component, _PretrainedStateMixin):
                    component.set_params(
                        **sub_params,
                        _reset=reset,
                        _keep_pretrained=keep_pretrained,
                    )
                else:
                    component.set_params(**sub_params)

        if len(unmatched_keys) > 0:
            valid_params = self.get_params(deep=True)
            unmatched_params = {key: params[key] for key in unmatched_keys}
            aliased_params = self._alias_params(unmatched_params, valid_params)

            if set(aliased_params) == set(unmatched_params):
                raise ValueError(
                    "Invalid parameter keys provided to set_params of "
                    f"object {self}. Check the list of available parameters "
                    "with `object.get_params().keys()`. "
                    f"Invalid keys provided: {unmatched_keys}"
                )

            self.set_params(
                **aliased_params,
                _reset=reset,
                _keep_pretrained=keep_pretrained,
            )

        return self

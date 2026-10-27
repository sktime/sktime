# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixin with pretrain state logic shared by estimator base classes."""

from sktime.base._pretrain._clone_plugin import _PretrainedCloner
from sktime.base._pretrain._state_at import _StateAtMixin


class _PretrainMixin(_StateAtMixin):
    """Pretrain state logic shared by estimator base classes.

    Provides the ``state`` property, the ``_is_fitted`` compatibility
    property, ``get_pretrained_params``, the ``_PretrainedCloner`` clone
    plugin, the state-aware methods of ``_StateAtMixin``, and
    ``_pretrain_and_track``, which runs ``_pretrain`` or ``_pretrain_update``
    and records the attributes these set.

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

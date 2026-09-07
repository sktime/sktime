"""Frozen forecaster which behaves like the fitted forecaster passed in init."""

__all__ = ["FrozenForecaster"]

from sktime.forecasting.base._delegate import _DelegatedForecaster


class FrozenForecaster(_DelegatedForecaster):
    """Frozen forecaster which behaves like an already fitted forecaster passed in init.

    Composing with FrozenForecaster instructs the wrapped forecaster to behave like
    the fitted forecaster passed in init. This is useful for testing the impact of
    updating the forecaster on forecasts, or for using a pre-fitted forecaster.

    In ``fit``, does nothing.

    Delegates the following methods to ``forecaster``:

    * ``predict``
    * ``predict_interval``
    * ``predict_quantiles``
    * ``predict_proba``

    If ``update_forecaster`` is ``True``, then the following are also delegated:

    * ``update``
    * ``update_predict``

    Parameters
    ----------
    forecaster : sktime forecaster - fitted
        The fitted sktime forecaster which is used in ``predict`` and other methods.

    deepcopy_forecaster : bool, default=False
        Whether internally a deepcopy of the passed forecaster is used.

        * True: a deepcopy of the passed forecaster is used internally
        * False: the passed forecaster is used internally as reference

    update_forecaster : bool, default=False
        Whether the passed forecaster is updated when ``update`` is called.

        * True: the passed forecaster is updated when ``update`` is called.
          In this case, ``deepcopy_forecaster`` must be ``True``.
          If ``deepcopy_forecaster`` is ``False``, raises ValueError at construction.

        * False: the passed forecaster is not updated when ``update`` is called.

    Attributes
    ----------
    forecaster_ : sktime forecaster, deepcopy of ``forecaster``
        if ``deepcopy_forecaster`` is ``True``, a ``deepcopy`` of ``forecaster``;
        otherwise a reference to the original ``forecaster`` is stored here
    """

    _tags = {
        "authors": "fkiraly",
        "maintainers": "fkiraly",
        "fit_is_empty": True,
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    def __init__(self, forecaster, deepcopy_forecaster=False, update_forecaster=False):
        self.forecaster = forecaster
        self.deepcopy_forecaster = deepcopy_forecaster
        self.update_forecaster = update_forecaster

        super().__init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * dynamic tag setting
        * any soft dependency imports in the constructor
        """
        if self.update_forecaster and not self.deepcopy_forecaster:
            raise ValueError(
                "If update_forecaster is True, deepcopy_forecaster must be True."
            )

        if self.deepcopy_forecaster:
            from copy import deepcopy

            self.forecaster_ = deepcopy(self.forecaster)
        else:
            self.forecaster_ = self.forecaster

        self._set_delegated_tags(self.forecaster_)

        self.set_tags(**{"fit_is_empty": True})

    def _fit(self, y, X, fh):
        """Fit forecaster to training data."""
        # empty because the forecaster is already fitted.
        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions apply,
              the method should handle uni- and multivariate y appropriately

        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        if self.update_forecaster:
            self.forecaster_.update(y=y, X=X, update_params=update_params)
        else:
            self.forecaster_.update(y=y, X=X, update_params=False)
        return self

    @classmethod
    def _get_clone_plugins(cls):
        """Get clone plugins for BaseObject.

        See scikit-base documentation for details on clone plugins.

        We need to override the cloning functionality for this estimator,
        since the ``forecaster`` attribute is already fitted when passed.
        We do not want to reset ``forecaster`` on ``clone``, and this needs to be
        overridden.

        Returns
        -------
        list of BaseCloner descendants, or None
            List of clone plugins for descendants.
            Each plugin must inherit from ``BaseCloner``
            in ``skbase.base._clone_plugins``, and implement
            the methods ``_check`` and ``_clone``.
        """
        from skbase.base._clone_plugins import _CloneSkbase

        class ModelCloner(_CloneSkbase):
            """Clone plugin to preserve model attribute of self."""

            def _check(self, obj):
                """Check if the plugin should be applied to the given object."""
                return isinstance(obj, cls)

            def _clone(self, obj):
                """Clone the ``model`` attribute of the given object."""
                # we do not want to reset the model on clone, so we return it as is
                temp = obj.forecaster
                temp2 = temp.clone()
                obj.forecaster = temp2
                clone = super()._clone(obj)
                obj.forecaster = temp
                clone.forecaster = temp
                return clone

        return [ModelCloner]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator/test.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.datasets import load_airline
        from sktime.forecasting.naive import NaiveForecaster

        est = NaiveForecaster(strategy="last")
        est.fit(load_airline()[:20])

        params0 = {"forecaster": est}
        params1 = {
            "forecaster": est,
            "deepcopy_forecaster": True,
            "update_forecaster": True,
        }
        params2 = {
            "forecaster": est,
            "deepcopy_forecaster": True,
            "update_forecaster": False,
        }
        return [params0, params1, params2]

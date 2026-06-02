"""Deprecation mixin and base class for global forecasting."""

from contextlib import contextmanager

from sktime.datatypes import convert_to
from sktime.forecasting.base._base import BaseForecaster
from sktime.utils.validation.forecasting import check_alpha

__all__ = ["_BaseGlobalForecaster", "_GlobalForecastingDeprecationMixin"]


class _GlobalForecastingDeprecationMixin:
    """Mixin for backward-compatible y parameter in predict().

    This mixin provides a deprecation path for forecasters that previously
    inherited from ``_BaseGlobalForecaster`` and supported a ``y`` parameter
    in ``predict()`` for global forecasting.

    .. deprecated:: 0.41.0

        The ``y`` parameter in ``predict()`` is deprecated and will be removed
        in version 1.1.0. Global forecasting now uses the data from ``fit()``.

    Usage
    -----
    Add this mixin BEFORE ``BaseForecaster`` in the inheritance list::

        class MyForecaster(_GlobalForecastingDeprecationMixin, BaseForecaster):
            ...

    The mixin must come before ``BaseForecaster`` to ensure correct MRO.
    """

    # TODO 1.1.0: Remove this mixin class entirely

    @contextmanager
    def _temporary_y_swap(self, X, y):
        """Temporarily replace self._y and cutoff with passed y data.

        Preserves old global forecasting behavior during the deprecation period,
        so that _predict sees the passed y instead of the fit-time y.
        """
        old_y = self._y
        old_cutoff = self._cutoff
        _, y_inner = self._check_X_y(X=X, y=y)
        self._y = y_inner
        self._set_cutoff_from_y(y_inner)
        try:
            yield
        finally:
            self._y = old_y
            self._cutoff = old_cutoff

    def _warn_y_deprecated(self, method_name):
        from sktime.utils.warnings import warn

        warn(
            f"In {self.__class__.__name__}.{method_name}(), the 'y' parameter "
            "is deprecated and will be removed in sktime version 1.1.0. "
            "Global forecasters now pass global pretraining data via pretrain, and"
            "historical  data via fit. "
            "To retain current behavior, pass pretraining data to pretrain(y) - "
            "previously fit(y) -"
            "and historical data as y to fit(y) - previously predict(y).",
            category=FutureWarning,
            obj=self,
        )

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

            If fh is not None and not of type ForecastingHorizon it is coerced to
            ForecastingHorizon via a call to _check_fh. In particular,
            if fh is of type pd.Index it is coerced via
            ForecastingHorizon(fh, is_relative=False)

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        Returns
        -------
        y_pred : time series in sktime compatible data container format
            Point forecasts at ``fh``, with same index as ``fh``.
            ``y_pred`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)
        """
        if y is not None:
            self._warn_y_deprecated("predict")
            with self._temporary_y_swap(X, y):
                return super().predict(fh=fh, X=X)
        return super().predict(fh=fh, X=X)

    def predict_interval(self, fh=None, X=None, coverage=0.90, y=None):
        """Compute/return prediction interval forecasts.

        If ``coverage`` is iterable, multiple intervals will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

            If ``fh`` is not None and not of type ``ForecastingHorizon``,
            it is coerced to ``ForecastingHorizon`` internally (via ``_check_fh``).

            * if ``fh`` is ``int`` or array-like of ``int``, it is interpreted as
              relative horizon, and coerced to a
              relative ``ForecastingHorizon(fh, is_relative=True)``.
            * if ``fh`` is of type ``pd.Index``, it is interpreted
              as an absolute horizon, and coerced
              to an absolute ``ForecastingHorizon(fh, is_relative=False)``.

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        coverage : float or list of float of unique values, optional (default=0.90)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Prediction interval forecasts, with columns and rows as follows:

            * Column has multi-index: first level is variable name from y in fit,
              second level coverage fractions for which intervals were computed.
              in the same order as in input ``coverage``.
              Third level is string "lower" or "upper", for lower/upper interval end.
            * Row index is fh, with additional (upper) levels equal to instance levels,
              from y seen in fit, if y seen in fit was Panel or Hierarchical.
            * Entries are forecasts of lower/upper interval end,
              for var in col index, at nominal coverage in second col index,
              lower/upper depending on third col index, for the row index.
              Upper/lower interval end forecasts are equivalent to
              quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        if y is not None:
            self._warn_y_deprecated("predict_interval")
            with self._temporary_y_swap(X, y):
                return super().predict_interval(fh=fh, X=X, coverage=coverage)
        return super().predict_interval(fh=fh, X=X, coverage=coverage)

    def predict_quantiles(self, fh=None, X=None, alpha=None, y=None):
        """Compute/return quantile forecasts.

        If ``alpha`` is iterable, multiple quantiles will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional


            If ``fh`` is not None and not of type ``ForecastingHorizon``,
            it is coerced to ``ForecastingHorizon`` internally (via ``_check_fh``).

            * if ``fh`` is ``int`` or array-like of ``int``, it is interpreted as
              relative horizon, and coerced to a
              relative ``ForecastingHorizon(fh, is_relative=True)``.
            * if ``fh`` is of type ``pd.Index``, it is interpreted
              as an absolute horizon, and coerced
              to an absolute ``ForecastingHorizon(fh, is_relative=False)``.


        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        alpha : float or list of float of unique values, optional (default=[0.05, 0.95])
            A probability or list of, at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts, with columns and rows as follows:

            * Column has multi-index: first level is variable name from y in fit,
              second level being the values of alpha passed to the function.
            * Row index is fh, with additional (upper) levels equal to instance levels,
              from y seen in fit, if y seen in fit was Panel or Hierarchical.
            * Entries are quantile forecasts, for var in col index,
              at quantile probability in second col index, for the row index.
        """
        if y is not None:
            self._warn_y_deprecated("predict_quantiles")
            with self._temporary_y_swap(X, y):
                return super().predict_quantiles(fh=fh, X=X, alpha=alpha)
        return super().predict_quantiles(fh=fh, X=X, alpha=alpha)

    def predict_var(self, fh=None, X=None, cov=False, y=None):
        """Compute/return variance forecasts.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional


            If ``fh`` is not None and not of type ``ForecastingHorizon``,
            it is coerced to ``ForecastingHorizon`` internally (via ``_check_fh``).

            * if ``fh`` is ``int`` or array-like of ``int``, it is interpreted as
              relative horizon, and coerced to a
              relative ``ForecastingHorizon(fh, is_relative=True)``.
            * if ``fh`` is of type ``pd.Index``, it is interpreted
              as an absolute horizon, and coerced
              to an absolute ``ForecastingHorizon(fh, is_relative=False)``.


        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on ``cov`` variable
            Variance forecasts, with columns and rows as follows:

            * Column names are exactly those of ``y`` passed in ``fit``/``update``.
              For nameless formats, column index will be a RangeIndex.
            * Row index is fh, with additional levels equal to instance levels,
              from y seen in fit, if y seen in fit was Panel or Hierarchical.
            * Entries are variance forecasts, for var in col index.

            A variance forecast for given variable and fh index is a predicted
            marginal variance for that variable and index, given observed data.
        """
        if y is not None:
            self._warn_y_deprecated("predict_var")
            with self._temporary_y_swap(X, y):
                return super().predict_var(fh=fh, X=X, cov=cov)
        return super().predict_var(fh=fh, X=X, cov=cov)


class _BaseGlobalForecaster(BaseForecaster):
    """Base global forecaster template class.

    .. deprecated:: 0.41.0

        ``_BaseGlobalForecaster`` is deprecated and will be removed in
        version 1.1.0. Inherit from ``BaseForecaster`` directly instead.

        For backward compatibility with the ``y`` parameter in ``predict()``,
        also inherit from ``_GlobalForecastingDeprecationMixin``.

        Key migration steps:

        * Change base class from ``_BaseGlobalForecaster`` to ``BaseForecaster``
        * Optionally add ``_GlobalForecastingDeprecationMixin`` for backward
          compatibility with the ``y`` parameter in ``predict()``
        * Update ``_fit`` signature to ``_fit(self, y, X=None, fh=None)``
        * Update ``_predict`` signature to ``_predict(self, fh, X=None)``
          (remove ``y`` parameter, use ``self._y`` instead)
        * Remove usage of ``self._global_forecasting`` flag

    The base forecaster specifies the methods and method signatures that all
    global forecasters have to implement.

    Specific implementations of these methods is deferred to concrete forecasters.
    """

    _tags = {"object_type": ["global_forecaster", "forecaster"]}

    # TODO 1.1.0: remove _BaseGlobalForecaster class entirely
    def __init_subclass__(cls, **kwargs):
        """Warn when _BaseGlobalForecaster is subclassed."""
        super().__init_subclass__(**kwargs)

        # Only warn for classes not in the sktime package
        module = cls.__module__
        if not module.startswith("sktime."):
            from sktime.utils.warnings import warn

            warn(
                f"Class '{cls.__name__}' inherits from _BaseGlobalForecaster, "
                "which is deprecated and will be removed in sktime version 1.1.0. "
                "Please inherit from BaseForecaster instead. "
                "For backward compatibility with the 'y' parameter in predict(), "
                "also inherit from _GlobalForecastingDeprecationMixin. "
                "See the _BaseGlobalForecaster docstring for migration steps.",
                category=FutureWarning,
                obj=cls,
            )

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is not passed (not performing global forecasting), ``X`` should
            only contain the time points to be predicted.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        y_pred : time series in sktime compatible data container format
            Point forecasts at ``fh``, with same index as ``fh``.
            ``y_pred`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        # handle inputs
        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True
        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # we call the ordinary _predict if no looping/vectorization needed
        if not self._is_vectorized:
            y_pred = self._predict(fh=fh, X=X_inner, y=y_inner)
        else:
            # otherwise we call the vectorized version of predict
            y_pred = self._vectorize("predict", y=y_inner, X=X_inner, fh=fh)

        # convert to output mtype, identical with last y mtype seen
        y_out = convert_to(
            y_pred,
            self._y_metadata["mtype"],
            store=self._converter_store_y,
            store_behaviour="freeze",
        )

        return y_out

    def _predict(self, fh, X, y):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        raise NotImplementedError("abstract method")

    def predict_quantiles(self, fh=None, X=None, alpha=None, y=None):
        """Compute/return quantile forecasts.

        If ``alpha`` is iterable, multiple quantiles will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        alpha : float or list of float of unique values, optional (default=[0.05, 0.95])
            A probability or list of, at which quantile forecasts are computed.

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        quantiles : pd.DataFrame

            * Column has multi-index: first level is variable name from y in fit,
              second level being the values of ``alpha`` passed to the function.
            * Row index is ``fh``.
              If ``y`` seen in fit was Panel or Hierarchical, has additional
              (upper) levels equal to instance levels, from ``y`` seen in ``fit``.
            * Entries are quantile forecasts, for variable in column index,
              at quantile probability in second column index, for the row index.

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "quantile predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True
        # default alpha
        if alpha is None:
            alpha = [0.05, 0.95]
        # check alpha and coerce to list
        alpha = check_alpha(alpha, name="alpha")

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # we call the ordinary _predict_quantiles if no looping/vectorization needed
        if not self._is_vectorized:
            quantiles = self._predict_quantiles(fh=fh, X=X_inner, alpha=alpha, y=y)
        else:
            # otherwise we call the vectorized version of predict_quantiles
            quantiles = self._vectorize(
                "predict_quantiles",
                fh=fh,
                X=X_inner,
                alpha=alpha,
                y=y,
            )

        return quantiles

    def predict_interval(self, fh=None, X=None, coverage=0.90, y=None):
        """Compute/return prediction interval forecasts.

        If ``coverage`` is iterable, multiple intervals will be calculated.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        coverage : float or list of float of unique values, optional (default=0.90)
           nominal coverage(s) of predictive interval(s)

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        pred_int : pd.DataFrame

            * Column has multi-index: first level is variable name from y in fit,
              second level are the coverage fractions for which intervals were computed,
              in the same order as in input ``coverage``.
              Third level is string "lower" or "upper", for lower/upper interval end.
            * Row index is ``fh``.
              If ``y`` seen in fit was Panel or Hierarchical, has additional
              (upper) levels equal to instance levels, from ``y`` seen in ``fit``.
            * Entries are forecasts of lower/upper interval end,
              for variable in column index, at nominal coverage in second column index,
              lower/upper depending on third col index, for the row index.
              Upper/lower interval end forecasts are equivalent to
              quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "prediction intervals. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # check alpha and coerce to list
        coverage = check_alpha(coverage, name="coverage")

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # we call the ordinary _predict_interval if no looping/vectorization needed
        if not self._is_vectorized:
            pred_int = self._predict_interval(
                fh=fh, X=X_inner, coverage=coverage, y=y_inner
            )
        else:
            # otherwise we call the vectorized version of predict_interval
            pred_int = self._vectorize(
                "predict_interval",
                fh=fh,
                X=X_inner,
                coverage=coverage,
                y=y_inner,
            )

        return pred_int

    def predict_var(self, fh=None, X=None, cov=False, y=None):
        """Compute/return variance forecasts.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.

        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on ``cov`` variable

            If cov=False:

            * Column names are exactly those of ``y`` passed in ``fit``/``update``.
              For nameless formats, column index will be a ``RangeIndex``.
            * Row index is ``fh``. If ``y`` seen in ``fit`` was Panel or Hierarchical,
              has additional levels equal to instance levels, from ``y`` in ``fit``.
            * Entries are variance forecasts, for var in col index.
              A variance forecast for given variable and fh index is a predicted
              variance for that variable and index, given observed data.

            If cov=True:

            * Column index is a multiindex: 1st level is variable names (as above)
              2nd level is fh.
            * Row index is ``fh``. If ``y`` seen in ``fit`` was Panel or Hierarchical,
              has additional levels equal to instance levels, from ``y`` in ``fit``.
            * Entries are (co-)variance forecasts, for var in col index, and
              covariance between time index in row and col.
              Note: no covariance forecasts are returned between different variables.

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "variance predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")
        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # we call the ordinary _predict_interval if no looping/vectorization needed
        if not self._is_vectorized:
            pred_var = self._predict_var(fh=fh, X=X_inner, cov=cov, y=y)
        else:
            # otherwise we call the vectorized version of predict_interval
            pred_var = self._vectorize("predict_var", fh=fh, X=X_inner, cov=cov, y=y)

        return pred_var

    def predict_proba(self, fh=None, X=None, marginal=True, y=None):
        """Compute/return fully probabilistic forecasts.

        Note: currently only implemented for Series (non-panel, non-hierarchical) y.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, np.array or ``ForecastingHorizon``, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "fully probabilistic predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )

        if hasattr(self, "_is_vectorized") and self._is_vectorized:
            raise NotImplementedError(
                "automated vectorization for predict_proba is not implemented"
            )

        # check global forecasting tag
        gf = self.get_tag(
            "capability:global_forecasting", tag_value_default=False, raise_error=False
        )
        if not gf and y is not None:
            ValueError("no global forecasting support!")

        self.check_is_fitted()
        if y is None:
            self._global_forecasting = False
        else:
            self._global_forecasting = True

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # this also updates cutoff from y
        # be cautious, in fit self._X and self._y is also updated but not here!
        if y_inner is not None:
            self._set_cutoff_from_y(y_inner)

        # pass to inner _predict_proba
        pred_dist = self._predict_proba(fh=fh, X=X_inner, marginal=marginal, y=y)

        return pred_dist

    # @classmethod
    # def _implementation_counts(cls) -> dict:
    #     """Functions need at least n overrides to be counted as implemented.

    #     A function needs to be specified only if n!=1.

    #     Returns
    #     -------
    #     dict
    #         key is function name, and the value is n.
    #     """
    #     return {
    #         "_predict_proba": 2,
    #         "_predict_var": 2,
    #         "_predict_interval": 2,
    #         "_predict_quantiles": 2,
    #     }

    # @classmethod
    # def _has_implementation_of(cls, method):
    #     """Check if method has a concrete implementation in this class.

    #     This assumes that having an implementation is equivalent to
    #         at least n overrides of `method` in the method resolution order.

    #     Parameters
    #     ----------
    #     method : str
    #         name of method to check implementation of

    #     Returns
    #     -------
    #     bool, whether method has implementation in cls
    #         True if cls.method has been overridden at least n times in
    #         the inheritance tree (according to method resolution order)
    #         n is different for each function. If a function has been overridden
    #         in _BaseGlobalForecaster and is going to be overridden in
    #         specific forecaster again, n should be 2.
    #         n should be specified in return of self._implementation_counts if n!=1.
    #     """
    #     # walk through method resolution order and inspect methods
    #     #   of classes and direct parents, "adjacent" classes in mro
    #     mro = inspect.getmro(cls)
    #     # collect all methods that are not none
    #     methods = [getattr(c, method, None) for c in mro]
    #     methods = [m for m in methods if m is not None]
    #     implementation_counts = cls._implementation_counts()
    #     if method in implementation_counts.keys():
    #         n = implementation_counts[method]
    #     else:
    #         n = 1
    #     _n = 0
    #     for i in range(len(methods) - 1):
    #         # the method has been overridden once iff
    #         #  at least two of the methods collected are not equal
    #         #  equivalently: some two adjacent methods are not equal
    #         overridden = methods[i] != methods[i + 1]
    #         if overridden:
    #             _n += 1
    #         if _n >= n:
    #             return True

    #     return False

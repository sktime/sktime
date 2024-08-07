# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements pipelines for forecasting."""

__all__ = ["TransformedTargetForecaster", "ForecastingPipeline", "ForecastX"]

import typing

import pandas as pd

from sktime.base import _HeterogenousMetaEstimator
from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.registry import scitype
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.validation.series import check_series
from sktime.utils.warnings import warn


class _Pipeline(_HeterogenousMetaEstimator, BaseForecaster):
    """Abstract class for forecasting pipelines."""

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_steps"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "steps_"

    def _get_pipeline_scitypes(self, estimators):
        """Get list of scityes (str) from names/estimator list."""
        return [scitype(x[1], raise_on_unknown=False) for x in estimators]

    def _get_forecaster_index(self, estimators):
        """Get the index of the first forecaster in the list."""
        return self._get_pipeline_scitypes(estimators).index("forecaster")

    def _check_steps(self, estimators, allow_postproc=False):
        """Check Steps.

        Parameters
        ----------
        estimators : list of estimators, or list of (name, estimator) pairs
        allow_postproc : bool, optional, default=False
            whether transformers after the forecaster are allowed

        Returns
        -------
        step : list of (name, estimator) pairs, estimators are cloned (not references)
            if estimators was a list of (str, estimator) tuples, then just cloned
            if was a list of estimators, then str are generated via _get_estimator_names

        Raises
        ------
        TypeError if names in ``estimators`` are not unique
        TypeError if estimators in ``estimators`` are not all forecaster or transformer
        TypeError if there is not exactly one forecaster in ``estimators``
        TypeError if not allow_postproc and forecaster is not last estimator
        """
        self_name = type(self).__name__
        if not isinstance(estimators, list):
            msg = (
                f"steps in {self_name} must be list of estimators, "
                f"or (string, estimator) pairs, "
                f"the two can be mixed; but, found steps of type {type(estimators)}"
            )
            raise TypeError(msg)

        # if len(estimators) == 1:
        #     msg = (
        #         f"in {self_name}, found steps of length 1, "
        #         f"this will result in the same behaviour "
        #         f"as not wrapping the single step in a pipeline. "
        #         f"Consider not wrapping steps in {self_name} as it is redundant."
        #     )
        #     warn(msg, obj=self)

        estimator_tuples = self._get_estimator_tuples(estimators, clone_ests=True)
        names, estimators = zip(*estimator_tuples)

        # validate names
        self._check_names(names)

        scitypes = self._get_pipeline_scitypes(estimator_tuples)
        if not set(scitypes).issubset(["forecaster", "transformer"]):
            raise TypeError(
                f"estimators passed to {self_name} "
                f"must be either transformer or forecaster"
            )
        if scitypes.count("forecaster") != 1:
            raise TypeError(
                f"exactly one forecaster must be contained in the chain, "
                f"but found {scitypes.count('forecaster')}"
            )

        forecaster_ind = self._get_forecaster_index(estimator_tuples)

        if not allow_postproc and forecaster_ind != len(estimators) - 1:
            TypeError(
                f"in {self_name}, last estimator must be a forecaster, "
                f"but found a transformer"
            )

        # Shallow copy
        return estimator_tuples

    def _iter_transformers(self, reverse=False, fc_idx=-1):
        # exclude final forecaster
        steps = self.steps_[:fc_idx]

        if reverse:
            steps = reversed(steps)

        for idx, (name, transformer) in enumerate(steps):
            yield idx, name, transformer

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self.steps)

    def _get_inverse_transform(self, transformers, y, X=None, mode=None):
        """Iterate over transformers.

        Inverse transform y (used for y_pred and pred_int)

        Parameters
        ----------
        transformers : list of (str, transformer) to apply
        y : pd.Series, pd.DataFrame
            Target series
        X : pd.Series, pd.DataFrame
            Exogenous series.
        mode : None or "proba"
            if proba, uses logic for probabilistic returns

        Returns
        -------
        y : pd.Series, pd.DataFrame
            Inverse transformed y
        """
        for _, transformer in reversed(transformers):
            # skip sktime transformers where inverse transform
            # is not wanted ur meaningful (e.g. Imputer, HampelFilter)
            skip_trafo = transformer.get_tag("skip-inverse-transform", False)
            if not skip_trafo:
                if mode is None:
                    y = transformer.inverse_transform(y, X)
                # if proba, we slice by quantile/coverage combination
                #   and collect the same quantile/coverage by variable
                #   then inverse transform, then concatenate
                elif mode == "proba":
                    idx = y.columns
                    n = idx.nlevels
                    idx_low = idx.droplevel(0).unique()
                    yt = dict()
                    for ix in idx_low:
                        levels = list(range(1, n))
                        if len(levels) == 1:
                            levels = levels[0]
                        yt[ix] = y.xs(ix, level=levels, axis=1)
                        # todo 0.32.0 - check why this cannot be easily removed
                        # in theory, we should get rid of the "Coverage" case treatment
                        # (the legacy naming convention was removed in 0.23.0)
                        # deal with the "Coverage" case, we need to get rid of this
                        #   i.d., special 1st level name of prediction object
                        #   in the case where there is only one variable
                        if len(yt[ix].columns) == 1:
                            temp = yt[ix].columns
                            yt[ix].columns = self._y.columns
                        yt[ix] = transformer.inverse_transform(yt[ix], X)
                        if len(yt[ix].columns) == 1:
                            yt[ix].columns = temp
                    y = pd.concat(yt, axis=1)
                    flipcols = [n - 1] + list(range(n - 1))
                    y.columns = y.columns.reorder_levels(flipcols)
                    y = y.loc[:, idx]
                else:
                    raise ValueError('mode arg must be None or "proba"')
        return y

    @property
    def named_steps(self):
        """Map the steps to a dictionary."""
        return dict(self._steps)

    @property
    def _steps(self):
        return self._get_estimator_tuples(self.steps, clone_ests=False)

    @_steps.setter
    def _steps(self, value):
        self.steps = value

    def _components(self, base_class=None):
        """Return references to all state changing BaseObject type attributes.

        This *excludes* the blue-print-like components passed in the __init__.

        Caution: this method returns *references* and not *copies*.
            Writing to the reference will change the respective attribute of self.

        Parameters
        ----------
        base_class : class, optional, default=None, must be subclass of BaseObject
            if None, behaves the same as ``base_class=BaseObject``
            if not None, return dict collects descendants of ``base_class``

        Returns
        -------
        dict with key = attribute name, value = reference to attribute
        dict contains all attributes of ``self`` that inherit from ``base_class``, and:
            whose names do not contain the string "__", e.g., hidden attributes
            are not class attributes, and are not hyper-parameters (``__init__`` args)
        """
        import inspect

        from sktime.base import BaseObject

        if base_class is None:
            base_class = BaseObject
        if base_class is not None and not inspect.isclass(base_class):
            raise TypeError(f"base_class must be a class, but found {type(base_class)}")
        # if base_class is not None and not issubclass(base_class, BaseObject):
        #     raise TypeError("base_class must be a subclass of BaseObject")

        fitted_estimator_tuples = self.steps_

        comp_dict = {name: comp for (name, comp) in fitted_estimator_tuples}
        return comp_dict

    # both children use the same step params for testing, so putting it here
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sklearn.preprocessing import StandardScaler

        from sktime.forecasting.compose._reduce import YfromX
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.transformations.series.detrend import Detrender
        from sktime.transformations.series.exponent import ExponentTransformer

        # StandardScaler does not skip fit, NaiveForecaster is not probabilistic
        STEPS1 = [
            ("transformer", TabularToSeriesAdaptor(StandardScaler())),
            ("forecaster", NaiveForecaster()),
        ]
        params1 = {"steps": STEPS1}

        # ARIMA has probabilistic methods, ExponentTransformer skips fit
        STEPS2 = [
            ("transformer", ExponentTransformer()),
            ("forecaster", YfromX.create_test_instance()),
        ]
        params2 = {"steps": STEPS2}

        params3 = {"steps": [Detrender(), YfromX.create_test_instance()]}

        return [params1, params2, params3]


# we ensure that internally we convert to pd.DataFrame for now
SUPPORTED_MTYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]


class ForecastingPipeline(_Pipeline):
    """Pipeline for forecasting with exogenous data.

    ForecastingPipeline is only applying the given transformers
    to X. The forecaster can also be a TransformedTargetForecaster containing
    transformers to transform y.

    For a list ``t1``, ``t2``, ..., ``tN``, ``f``
        where ``t[i]`` are transformers, and ``f`` is an ``sktime`` forecaster,
        the pipeline behaves as follows:

    ``fit(y, X, fh)`` changes state by running ``t1.fit_transform`` with ``X=X`, ``y=y``
        then ``t2.fit_transform`` on ``X=`` the output of ``t1.fit_transform``, ``y=y``,
        etc, sequentially, with ``t[i]`` receiving the output of ``t[i-1]`` as ``X``,
        then running ``f.fit`` with ``X`` being the output of ``t[N]``, and ``y=y``

    ``predict(X, fh)`` - result is of executing ``f.predict``, with ``fh=fh``, and ``X``
        being the result of the following process:
        running ``t1.fit_transform`` with ``X=X``,
        then ``t2.fit_transform`` on ``X=`` the output of ``t1.fit_transform``, etc
        sequentially, with ``t[i]`` receiving the output of ``t[i-1]`` as ``X``,
        and returning th output of ``tN`` to pass to ``f.predict`` as ``X``.

    ``predict_interval(X, fh)``, ``predict_quantiles(X, fh)`` - as ``predict(X, fh)``,
        with ``predict_interval`` or ``predict_quantiles`` substituted for ``predict``

    ``predict_var``, ``predict_proba`` - uses base class default to obtain
        crude normal estimates from ``predict_quantiles``.

    ``get_params``, ``set_params`` uses ``sklearn`` compatible nesting interface:

        * if list is unnamed, names are generated as names of classes
        * if names are non-unique, ``f"_{str(i)}"`` is appended to each name string
          where ``i`` is the total count of occurrence of a non-unique string
          inside the list of names leading up to it (inclusive)

    ``ForecastingPipeline`` can also be created by using the magic multiplication
        on any forecaster, i.e., if ``my_forecaster`` inherits from ``BaseForecaster``,
        and ``my_t1``, ``my_t2``, inherit from ``BaseTransformer``,
        then, for instance, ``my_t1 ** my_t2 ** my_forecaster``
        will result in the same object as  obtained from the constructor
        ``ForecastingPipeline([my_t1, my_t2, my_forecaster])``.
        Magic multiplication can also be used with (str, transformer) pairs,
        as long as one element in the chain is a transformer.

    Parameters
    ----------
    steps : list of sktime transformers and forecasters, or
        list of tuples (str, estimator) of ``sktime`` transformers or forecasters.
        The list must contain exactly one forecaster.
        These are "blueprint" transformers resp forecasters,
        forecaster/transformer states do not change when ``fit`` is called.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of ``sktime`` transformers or forecasters
        clones of estimators in ``steps`` which are fitted in the pipeline
        is always in (str, estimator) format, even if ``steps`` is just a list
        strings not passed in ``steps`` are replaced by unique generated strings
        i-th transformer in ``steps_`` is clone of i-th in ``steps``
    forecaster_ : estimator, reference to the unique forecaster in ``steps_``

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.split import temporal_train_test_split
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> y, X = load_longley()
    >>> y_train, _, X_train, X_test = temporal_train_test_split(y, X)
    >>> fh = ForecastingHorizon(X_test.index, is_relative=False)

        Example 1: string/estimator pairs
    >>> pipe = ForecastingPipeline(steps=[
    ...     ("imputer", Imputer(method="mean")),
    ...     ("minmaxscaler", TabularToSeriesAdaptor(MinMaxScaler())),
    ...     ("forecaster", NaiveForecaster(strategy="drift")),
    ... ])
    >>> pipe.fit(y_train, X_train)
    ForecastingPipeline(...)
    >>> y_pred = pipe.predict(fh=fh, X=X_test)

        Example 2: without strings
    >>> pipe = ForecastingPipeline([
    ...     Imputer(method="mean"),
    ...     TabularToSeriesAdaptor(MinMaxScaler()),
    ...     NaiveForecaster(strategy="drift"),
    ... ])

        Example 3: using the dunder method
        Note: * (= apply to ``y``) has precedence over ** (= apply to ``X``)

    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> imputer = Imputer(method="mean")
    >>> pipe = (imputer * MinMaxScaler()) ** forecaster

        Example 3b: using the dunder method, alternative
    >>> pipe = imputer ** MinMaxScaler() ** forecaster
    """

    _tags = {
        "authors": ["mloning", "fkiraly", "aiwalter"],
        "scitype:y": "both",
        "y_inner_mtype": SUPPORTED_MTYPES,
        "X_inner_mtype": SUPPORTED_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "capability:pred_int": True,
        "X-y-must-have-same-index": False,
    }

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps(steps, allow_postproc=False)
        super().__init__()
        tags_to_clone = [
            "ignores-exogeneous-X",  # does estimator ignore the exogeneous X?
            "capability:pred_int",  # can the estimator produce prediction intervals?
            "capability:pred_int:insample",  # ... for in-sample horizons?
            "capability:insample",  # can the estimator make in-sample predictions?
            "requires-fh-in-fit",  # is forecasting horizon already required in fit?
            "enforce_index_type",  # index type that needs to be enforced in X/y
        ]
        # we do not clone X-y-must-have-same-index, since transformers can
        #   create indices, and that behaviour is not tag-inspectable
        self.clone_tags(self.forecaster_, tags_to_clone)
        self._anytagis_then_set("fit_is_empty", False, True, self.steps_)

    @property
    def forecaster_(self):
        """Return reference to the forecaster in the pipeline.

        Valid after _fit.
        """
        return self.steps_[-1][1]

    def __rpow__(self, other):
        """Magic ** method, return (left) concatenated ForecastingPipeline.

        Implemented for ``other`` being a transformer,
        otherwise returns ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from ``BaseTransformer``
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        ForecastingPipeline object,
            concatenation of ``other`` (first) with ``self`` (last).
            not nested, contains only non-TransformerPipeline ``sktime`` steps
        """
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline

        _, ests = zip(*self.steps_)
        names = tuple(self._get_estimator_names(self.steps))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.steps_)
            names_o = tuple(other._get_estimator_names(other.steps))
            new_names = names_o + names
            new_ests = trafos_o + ests
        elif isinstance(other, BaseTransformer):
            new_names = (type(other).__name__,) + names
            new_ests = (other,) + ests
        elif self._is_name_and_est(other, BaseTransformer):
            other_name = other[0]
            other_trafo = other[1]
            new_names = (other_name,) + names
            new_ests = (other_trafo,) + ests
        else:
            return NotImplemented

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_ests)):
            return ForecastingPipeline(steps=list(new_ests))
        else:
            return ForecastingPipeline(steps=list(zip(new_names, new_ests)))

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, required
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        # skip transformers if X is ignored
        # condition 1 for ignoring X: X is None and required in fit of 1st transformer
        first_trafo = self.steps_[0][1]
        cond1 = len(self.steps_) > 1 and first_trafo.get_tag("requires_X")
        cond1 = cond1 and X is None

        # condition 2 for ignoring X: tag "ignores-exogeneous-X" is True
        # in this case the forecaster at the end ignores what comes out of the trafos
        cond2 = self.get_tag("ignores-exogeneous-X")

        # X ignored = condition 1 or condition 2
        skip_trafos = cond1 or cond2
        self.skip_trafos_ = skip_trafos

        # If X is ignored, just ignore the transformers and pass through to forecaster
        if not skip_trafos:
            # transform X
            for step_idx, name, transformer in self._iter_transformers():
                t = transformer.clone()
                X = t.fit_transform(X=X, y=y)
                self.steps_[step_idx] = (name, t)

        # fit forecaster
        name, forecaster = self.steps_[-1]
        f = forecaster.clone()
        f.fit(y=y, X=X, fh=fh)
        self.steps_[-1] = (name, f)

        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, required
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        X = self._transform(X=X, y=fh)
        return self.forecaster_.predict(fh, X)

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        X = self._transform(X=X, y=fh)
        return self.forecaster_.predict_quantiles(fh=fh, X=X, alpha=alpha)

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        X = self._transform(X=X, y=fh)
        return self.forecaster_.predict_interval(fh=fh, X=X, coverage=coverage)

    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on ``cov`` variable
            If cov=False:
                Column names are exactly those of ``y`` passed in ``fit``/``update``.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh. Entries are variance forecasts, for var in col index.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
        """
        X = self._transform(X=X, y=fh)
        return self.forecaster_.predict_var(fh=fh, X=X, cov=cov)

    # todo: does not work properly for multivariate or hierarchical
    #   still need to implement this - once interface is consolidated
    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        X = self._transform(X=X, y=fh)
        return self.forecaster_.predict_proba(fh=fh, X=X, marginal=marginal)

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame, required
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        if not self.skip_trafos_:
            for _, _, transformer in self._iter_transformers():
                if hasattr(transformer, "update"):
                    transformer.update(X=X, y=y, update_params=update_params)
                    X = transformer.transform(X=X, y=y)

        _, forecaster = self.steps_[-1]
        forecaster.update(y=y, X=X, update_params=update_params)
        return self

    def _transform(self, X=None, y=None):
        # If X is not given or ignored, just passthrough the data without transformation
        if not self.skip_trafos_:
            for _, _, transformer in self._iter_transformers():
                # if y is required but not passed,
                # we create a zero-column y from the forecasting horizon
                requires_y = transformer.get_tag("requires_y", False)
                if isinstance(y, ForecastingHorizon) and requires_y:
                    y = y.to_absolute_index(self.cutoff)
                    y = pd.DataFrame(index=y)
                elif isinstance(y, ForecastingHorizon) and not requires_y:
                    y = None
                # else we just pass on y
                X = transformer.transform(X=X, y=y)
        return X


# removed transform and inverse_transform as long as y can only be a pd.Series
# def transform(self, Z, X=None):
#     self.check_is_fitted()
#     Zt = check_series(Z, enforce_multivariate=True)
#     for _, _, transformer in self._iter_transformers():
#         Zt = transformer.transform(Zt)
#     return Zt
#
# def inverse_transform(self, Z, X=None):
#     self.check_is_fitted()
#     Zt = check_series(Z, enforce_multivariate=True)
#     for _, _, transformer in self._iter_transformers(reverse=True):
#         if not _has_tag(transformer, "skip-inverse-transform"):
#             Zt = transformer.inverse_transform(Zt)
#     return Zt


class TransformedTargetForecaster(_Pipeline):
    """Meta-estimator for forecasting transformed time series.

    Pipeline functionality to apply transformers to endogeneous time series, ``y``.
    The exogeneous data, ``X``, is not transformed.
    To transform ``X``, the ``ForecastingPipeline`` can be used.

    For a list ``t1``, ``t2``, ..., ``tN``, ``f``, ``tp1``, ``tp2``, ..., ``tpM``,
        where ``t[i]`` and ``tp[i]`` are transformers
        (``t`` to pre-, ``tp`` to post-process),
        and ``f`` is an sktime forecaster,
        the pipeline behaves as follows:

    ``fit(y, X, fh)`` - changes state by running ``t1.fit_transform``
        with ``X=y``, ``y=X``,
        then ``t2.fit_transform`` on ``X=`` the output of ``t1.fit_transform``, ``y=X``,
        etc, sequentially, with ``t[i]`` receiving the output of ``t[i-1]`` as ``X``,
        then running ``f.fit`` with ``y`` being the output of ``t[N]``, and ``X=X``,
        then running ``tp1.fit_transform``  with ``X=y``, ``y=X``,
        then ``tp2.fit_transform`` on ``X=`` the output of ``tp1.fit_transform``, etc
        sequentially, with ``tp[i]`` receiving the output of ``tp[i-1]``,

    ``predict(X, fh)`` - result is of executing ``f.predict``, with ``X=X``, ``fh=fh``,
        then running ``tp1.inverse_transform`` with ``X=`` the output of ``f``, ``y=X``,
        then ``t2.inverse_transform`` on ``X=`` the output of ``t1.inverse_transform``,
        etc, sequentially, with ``t[i]`` receiving the output of ``t[i-1]`` as ``X``,
        then running ``tp1.fit_transform`` with ``X=`` the output of ``t[N]s``, ``y=X``,
        then ``tp2.fit_transform`` on ``X=`` the output of ``tp1.fit_transform``, etc,
        sequentially, with ``tp[i]`` receiving the output of ``tp[i-1]``,

    ``predict_interval(X, fh)``, ``predict_quantiles(X, fh)`` - as ``predict(X, fh)``,
        with ``predict_interval`` or ``predict_quantiles`` substituted for ``predict``

    ``predict_var``, ``predict_proba`` - uses base class default to obtain
        crude normal estimates from ``predict_quantiles``.

    ``get_params``, ``set_params`` uses ``sklearn`` compatible nesting interface:

        * if list is unnamed, names are generated as names of classes
        * if names are non-unique, ``f"_{str(i)}"`` is appended to each name string
          where ``i`` is the total count of occurrence of a non-unique string
          inside the list of names leading up to it (inclusive)

    ``TransformedTargetForecaster`` can also be created by using the magic
        multiplication
        on any forecaster, i.e., if ``my_forecaster`` inherits from ``BaseForecaster``,
        and ``my_t1``, ``my_t2``, ``my_tp`` inherit from ``BaseTransformer``,
        then, for instance, ``my_t1 * my_t2 * my_forecaster * my_tp``
        will result in the same object as  obtained from the constructor
        ``TransformedTargetForecaster([my_t1, my_t2, my_forecaster, my_tp])``.
        Magic multiplication can also be used with (str, transformer) pairs,
        as long as one element in the chain is a transformer.

    Parameters
    ----------
    steps : list of ``sktime`` transformers and forecasters, or
        list of tuples (str, estimator) of ``sktime`` transformers or forecasters.
        The list must contain exactly one forecaster.
        These are "blueprint" transformers resp forecasters,
        forecaster/transformer states do not change when ``fit`` is called.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of ``sktime`` transformers or forecasters
        clones of estimators in ``steps`` which are fitted in the pipeline
        is always in (str, estimator) format, even if ``steps`` is just a list
        strings not passed in ``steps`` are replaced by unique generated strings
        i-th transformer in ``steps_`` is clone of i-th in ``steps``
    forecaster_ : estimator, reference to the unique forecaster in ``steps_``
    transformers_pre_ : list of tuples (str, transformer) of sktime transformers
        reference to pairs in ``steps_`` that precede ``forecaster_``
    transformers_ost_ : list of tuples (str, transformer) of sktime transformers
        reference to pairs in ``steps_`` that succeed ``forecaster_``

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> y = load_airline()

        Example 1: string/estimator pairs

    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("imputer", Imputer(method="mean")),
    ...     ("detrender", Detrender()),
    ...     ("forecaster", NaiveForecaster(strategy="drift")),
    ... ])
    >>> pipe.fit(y)
    TransformedTargetForecaster(...)
    >>> y_pred = pipe.predict(fh=[1,2,3])

        Example 2: without strings

    >>> pipe = TransformedTargetForecaster([
    ...     Imputer(method="mean"),
    ...     Detrender(),
    ...     NaiveForecaster(strategy="drift"),
    ...     ExponentTransformer(),
    ... ])

        Example 3: using the dunder method

    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> imputer = Imputer(method="mean")
    >>> pipe = imputer * Detrender() * forecaster * ExponentTransformer()
    """

    _tags = {
        "authors": ["mloning", "fkiraly", "aiwalter"],
        "scitype:y": "both",
        "y_inner_mtype": SUPPORTED_MTYPES,
        "X_inner_mtype": SUPPORTED_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "capability:pred_int": True,
        "X-y-must-have-same-index": False,
    }

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps(steps, allow_postproc=True)
        super().__init__()

        # set the tags based on forecaster
        tags_to_clone = [
            "ignores-exogeneous-X",  # does estimator ignore the exogeneous X?
            "capability:pred_int",  # can the estimator produce prediction intervals?
            "capability:pred_int:insample",  # ... for in-sample horizons?
            "capability:insample",  # can the estimator make in-sample predictions?
            "requires-fh-in-fit",  # is forecasting horizon already required in fit?
            "enforce_index_type",  # index type that needs to be enforced in X/y
        ]
        # we do not clone X-y-must-have-same-index, since transformers can
        #   create indices, and that behaviour is not tag-inspectable
        self.clone_tags(self.forecaster_, tags_to_clone)
        self._anytagis_then_set("fit_is_empty", False, True, self.steps_)

        # above, we cloned the ignores-exogeneous-X tag,
        # but we also need to check whether X is used as y in some transformer
        # in this case X is not ignored by the pipe, even if the forecaster ignores it
        # logic below checks whether there is at least one such transformer
        # if there is, we override the ignores-exogeneous-X tag to False
        # also see discussion in bug issue #5518
        pre_ts = self.transformers_pre_
        post_ts = self.transformers_post_
        pre_use_y = [est.get_tag("y_inner_mtype") != "None" for _, est in pre_ts]
        post_use_y = [est.get_tag("y_inner_mtype") != "None" for _, est in post_ts]
        any_t_use_y = any(pre_use_y) or any(post_use_y)

        if any_t_use_y:
            self.set_tags(**{"ignores-exogeneous-X": False})

    @property
    def forecaster_(self):
        """Return reference to the forecaster in the pipeline.

        Returns
        -------
        sktime forecaster
            reference to unique forecaster in steps_ (without the name)
        """
        return self.steps_[self._get_forecaster_index(self.steps_)][1]

    @property
    def transformers_pre_(self):
        """Return reference to the list of pre-forecast transformers.

        Returns
        -------
        list of tuples (str, estimator) of sktime transformers
            reference to tuples that come before the unique (str, forecaster) in steps_
        """
        return self.steps_[: self._get_forecaster_index(self.steps_)]

    @property
    def transformers_post_(self):
        """Return reference to the list of post-forecast transformers.

        Returns
        -------
        list of tuples (str, estimator) of sktime transformers
            reference to tuples that come after the unique (str, forecaster) in steps_
        """
        return self.steps_[(1 + self._get_forecaster_index(self.steps_)) :]

    def __mul__(self, other):
        """Magic * method, return (right) concatenated TransformedTargetForecaster.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformedTargetForecaster object,
            concatenation of ``self`` (first) with ``other`` (last).
            not nested, contains only non-TransformerPipeline ``sktime`` transformers
        """
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline

        # we don't use names but _get_estimator_names to get the *original* names
        #   to avoid multiple "make unique" calls which may grow strings too much
        _, ests = zip(*self.steps_)
        names = tuple(self._get_estimator_names(self.steps))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.steps_)
            names_o = tuple(other._get_estimator_names(other.steps))
            new_names = names + names_o
            new_ests = ests + trafos_o
        elif isinstance(other, BaseTransformer):
            new_names = names + (type(other).__name__,)
            new_ests = ests + (other,)
        elif self._is_name_and_est(other, BaseTransformer):
            other_name = other[0]
            other_trafo = other[1]
            new_names = names + (other_name,)
            new_ests = ests + (other_trafo,)
        else:
            return NotImplemented

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_ests)):
            return TransformedTargetForecaster(steps=list(new_ests))
        else:
            return TransformedTargetForecaster(steps=list(zip(new_names, new_ests)))

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated TransformerPipeline.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformedTargetForecaster object,
            concatenation of ``other`` (first) with ``self`` (last).
            not nested, contains only non-TransformerPipeline ``sktime`` steps
        """
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline

        _, ests = zip(*self.steps_)
        names = tuple(self._get_estimator_names(self.steps))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.steps_)
            names_o = tuple(other._get_estimator_names(other.steps))
            new_names = names_o + names
            new_ests = trafos_o + ests
        elif isinstance(other, BaseTransformer):
            new_names = (type(other).__name__,) + names
            new_ests = (other,) + ests
        elif self._is_name_and_est(other, BaseTransformer):
            other_name = other[0]
            other_trafo = other[1]
            new_names = (other_name,) + names
            new_ests = (other_trafo,) + ests
        else:
            return NotImplemented

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_ests)):
            return TransformedTargetForecaster(steps=list(new_ests))
        else:
            return TransformedTargetForecaster(steps=list(zip(new_names, new_ests)))

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        self.steps_ = self._get_estimator_tuples(self.steps, clone_ests=True)

        # transform pre
        yt = y
        for _, t in self.transformers_pre_:
            yt = t.fit_transform(X=yt, y=X)

        # fit forecaster
        f = self.forecaster_
        f.fit(y=yt, X=X, fh=fh)

        # transform post
        for _, t in self.transformers_post_:
            y = t.fit_transform(X=y, y=X)

        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_pred = self.forecaster_.predict(fh=fh, X=X)
        # inverse transform y_pred
        y_pred = self._get_inverse_transform(self.transformers_pre_, y_pred, X)

        # transform post
        for _, t in self.transformers_post_:
            y_pred = t.transform(X=y_pred, y=X)

        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        # transform pre
        for _, t in self.transformers_pre_:
            if hasattr(t, "update"):
                t.update(X=y, y=X, update_params=update_params)
                y = t.transform(X=y, y=X)

        self.forecaster_.update(y=y, X=X, update_params=update_params)

        # transform post
        for _, t in self.transformers_post_:
            t.update(X=y, y=X, update_params=update_params)
            y = t.transform(X=y, y=X)

        return self

    def transform(self, Z, X=None):
        """Return transformed version of input series ``Z``.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the transformation on.
        X : pd.DataFrame, default=None
            Exogenous data used in transformation.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed version of input series ``Z``.
        """
        self.check_is_fitted()
        zt = check_series(Z)
        for _, transformer in self.transformers_pre_:
            zt = transformer.transform(zt, X)
        return zt

    def inverse_transform(self, Z, X=None):
        """Reverse transformation on input series ``Z``.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to reverse the transformation on.
        X : pd.DataFrame, default=None
            Exogenous data used in transformation.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        return self._get_inverse_transform(self.transformers_pre_, Z, X)

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        pred_int = self.forecaster_.predict_quantiles(fh=fh, X=X, alpha=alpha)
        pred_int_transformed = self._get_inverse_transform(
            self.transformers_pre_, pred_int, mode="proba"
        )
        return pred_int_transformed

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        pred_int = self.forecaster_.predict_interval(fh=fh, X=X, coverage=coverage)
        pred_int_transformed = self._get_inverse_transform(
            self.transformers_pre_, pred_int, mode="proba"
        )
        return pred_int_transformed


class ForecastX(BaseForecaster):
    """Forecaster that forecasts exogeneous data for use in an endogeneous forecast.

    In ``predict``, this forecaster carries out a ``predict`` step on exogeneous ``X``.
    Then, a forecast is made for ``y``,
    using exogeneous data plus its forecasts as ``X``.
    If ``columns`` argument is provided, will carry ``predict`` out only for the columns
    in ``columns``, and will use other columns in ``X`` unchanged.

    The two forecasters and forecasting horizons (for forecasting ``y`` resp ``X``)
    can be selected independently, but default to the same.

    The typical use case is extending exogeneous data available only up until the cutoff
    into the future, for use by an exogeneous forecaster that requires such future data.

    If no X is passed in ``fit``, behaves like ``forecaster_y``.
    In such a case (no exogeneous data), there is no benefit in using this compositor.

    If variables in ``columns`` are present in the provided ``X`` during ``predict``,
    by default these are still forecasted and the forecasts are used for prediction of
    ``y`` variables. This behaviour can be modified by passing ``predict_behaviour``
    argument as ``"use_actuals"`` instead of the default value of ``"use_forecasts"``.

    Parameters
    ----------
    forecaster_y : BaseForecaster
        sktime forecaster to use for endogeneous data ``y``

    forecaster_X : BaseForecaster, optional
        sktime forecaster to use for exogeneous data ``X``,
        default = None = same as ``forecaster_y``

    fh_X : None, ForecastingHorizon, or valid input to construct ForecastingHorizon
        optional, default = None = same as used for ``y`` in any instance.
        valid inputs to construct ``ForecastingHorizon`` are:
        int, list of int, 1D np.ndarray, pandas.Index (see ForecastingHorizon)

    behaviour : str, one of "update" or "refit", optional, default = "update"

        * if "update", ``forecaster_X`` is fit to the data batch seen in ``fit``,
        and updated with any ``X`` seen in calls of ``update``.
        Forecast added to ``X`` in ``predict`` is obtained from this state.

        * if "refit", then ``forecaster_X`` is fit to ``X`` in ``predict`` only,
        Forecast added to ``X`` in ``predict`` is obtained from this state.

    columns : None, or pandas compatible index iterator (e.g., list of str), optional
        default = None = all columns in ``X`` are used for forecast columns to which
        ``forecaster_X`` is applied.
        If not ``None``, must be a non-empty list of valid column names.
        Note that ``[]`` and ``None`` do not imply the same.

    fit_behaviour : str, one of "use_actual" (default), "use_forecast", optional,

        * if "use_actual", then ``forecaster_y`` uses the actual ``X`` as
        exogenous features in ``fit``
        * if "use_forecast", then ``forecaster_y`` uses the ``X`` predicted by
        ``forecaster_X`` as exogenous features in ``fit``

    forecaster_X_exogeneous : optional, str, one of "None" (default), or "complement",
        or ``pandas.Index`` coercible

        * if "None", then ``forecaster_X`` uses no exogenous data
        * if "complement", then ``forecaster_X`` uses the complement of the
        ``columns`` as exogenous data to forecast. This is typically useful
        if the complement of ``columns`` is known to be available in the future.
        * if a ``pandas.Index`` coercible, then uses columns indexed by the index
        after coercion, in ``X`` passed (converted to pandas)

    predict_behaviour : str, optional (default = "use_forecasts")

        * if "use_forecasts", then ``forecaster_X`` predictions are always used as
            inputs in ``forecaster_y``, even if passed ``X`` has future values
        * if "use_actuals", then ``forecaster_X`` predictions are only used if
            passed ``X`` lacks future values for the variables in ``columns``

    Attributes
    ----------
    forecaster_X_ : BaseForecaster
        clone of ``forecaster_X``, state updates with ``fit`` and ``update``
        created only if ``behaviour="update"`` and ``X`` passed is not None
        and ``forecaster_y`` has ``ignores-exogeneous-X`` tag as ``False``
    forecaster_y_ : BaseForecaster
        clone of ``forecaster_y``, state updates with ``fit`` and ``update``

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.arima import ARIMA
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.compose import ForecastX
    >>> from sktime.forecasting.var import VAR

    >>> y, X = load_longley()
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> pipe = ForecastX(  # doctest: +SKIP
    ...     forecaster_X=VAR(),
    ...     forecaster_y=ARIMA(),
    ... )
    >>> pipe = pipe.fit(y, X=X, fh=fh)  # doctest: +SKIP
    >>> # this now works without X from the future of y!
    >>> y_pred = pipe.predict(fh=fh)  # doctest: +SKIP

    to forecast only some columns, use the ``columns`` arg,
    and pass known columns to ``predict``:

    >>> columns = ["ARMED", "POP"]
    >>> pipe = ForecastX(  # doctest: +SKIP
    ...     forecaster_X=VAR(),
    ...     forecaster_y=SARIMAX(),
    ...     columns=columns,
    ... )
    >>> pipe = pipe.fit(y_train, X=X_train, fh=fh)  # doctest: +SKIP
    >>> # dropping ["ARMED", "POP"] = columns where we expect not to have future values
    >>> y_pred = pipe.predict(fh=fh, X=X_test.drop(columns=columns))  # doctest: +SKIP

    Notes
    -----
    * ``predict_behaviour="use_actuals"`` is as of now unused if future values are
        passed for a subset of exogeneous variables in ``columns``. In that case, it
        behaves as if ``predict_behaviour="use_forecasts"``.
    """

    _tags = {
        "authors": ["fkiraly", "benheid", "yarnabrina"],
        "X_inner_mtype": SUPPORTED_MTYPES,
        "y_inner_mtype": SUPPORTED_MTYPES,
        "scitype:y": "both",
        "X-y-must-have-same-index": False,
        "fit_is_empty": False,
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "handles-missing-data": True,
    }

    def __init__(
        self,
        forecaster_y,
        forecaster_X=None,
        fh_X=None,
        behaviour="update",
        columns=None,
        fit_behaviour="use_actual",
        forecaster_X_exogeneous="None",
        predict_behaviour="use_forecasts",
    ):
        if fit_behaviour not in ["use_actual", "use_forecast"]:
            raise ValueError(
                'fit_behaviour must be one of "use_actual", "use_forecast"'
            )
        self.fit_behaviour = fit_behaviour
        if behaviour not in ["update", "refit"]:
            raise ValueError('behaviour must be one of "update", "refit"')

        self.forecaster_y = forecaster_y
        self.forecaster_X = forecaster_X

        # forecaster_X_c handles the "if forecaster_X is None, use forecaster_y"
        if forecaster_X is None:
            self.forecaster_X_c = forecaster_y
        else:
            self.forecaster_X_c = forecaster_X
        self.fh_X = fh_X
        self.behaviour = behaviour
        self.columns = columns
        if isinstance(forecaster_X_exogeneous, str):
            if forecaster_X_exogeneous not in ["None", "complement"]:
                raise ValueError(
                    'forecaster_X_exogeneous must be one of "None", "complement",'
                    "or a pandas.Index coercible"
                )
        self.forecaster_X_exogeneous = forecaster_X_exogeneous

        if predict_behaviour not in ["use_forecasts", "use_actuals"]:
            raise ValueError(
                "predict_behaviour must be one of 'use_forecasts', 'use_actuals'"
            )

        self.predict_behaviour = predict_behaviour

        super().__init__()

        tags_to_clone_from_forecaster_y = [
            "capability:pred_int",
            "capability:pred_int:insample",
            "capability:insample",
            "ignores-exogeneous-X",
        ]

        self.clone_tags(forecaster_y, tags_to_clone_from_forecaster_y)

        # tag_translate_dict = {
        #    "handles-missing-data": forecaster.get_tag("handles-missing-data")
        # }
        # self.set_tags(**tag_translate_dict)

        if (
            self.fit_behaviour == "use_forecast"
            and self.predict_behaviour == "use_actuals"
        ):
            warn(
                "ForecastX is configured with fit_behaviour='use_forecast' and "
                "predict_behaviour='use_actuals'. This implies in-sample predictions "
                "generated by trained `forecaster_X` will be used as exogenous data to "
                "fit `forecaster_y`, but future predictions by `forecaster_X` may not "
                "be used as exogenous data during `forecaster_y` predictions. This is "
                "an unusual configuration and may lead to unexpected results.",
                obj=self,
                stacklevel=2,
            )

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : time series in sktime compatible format
            Target time series to which to fit the forecaster
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : time series in sktime compatible format, optional, default=None
            Exogenous time series to which to fit the forecaster

        Returns
        -------
        self : returns an instance of self.
        """
        if self.fh_X is None:
            fh_X = fh
        else:
            fh_X = self.fh_X
        self.fh_X_ = fh_X

        # remember if X seen was None
        self.X_was_None_ = X is None

        # initialize forecaster_X_ and forecaster_y_
        self.forecaster_y_ = self.forecaster_y.clone()
        if X is not None:
            self.forecaster_X_ = self.forecaster_X_c.clone()

        if self.behaviour == "update" and X is not None:
            X_for_fcX = self._get_X_for_fcX(X)
            self.forecaster_X_.fit(y=self._get_Xcols(X), fh=fh_X, X=X_for_fcX)

        if X is None or self.fit_behaviour == "use_actual":
            X_for_fcy = X
        elif self.fit_behaviour == "use_forecast":
            if not self.forecaster_X_.get_tag("capability:insample"):
                raise ValueError(
                    "forecaster_X does not have `capability:insample`. "
                    "Thus, it is not valid with `fit_behaviour=use_forecast`."
                )
            if isinstance(X.index, pd.MultiIndex):
                X_times = X.index.get_level_values(-1).unique()
            else:
                X_times = X.index
            fh_for_fcst = ForecastingHorizon(X_times, is_relative=False)
            X_for_fcy = self.forecaster_X_.predict(fh=fh_for_fcst, X=X)

        self.forecaster_y_.fit(y=y, X=X_for_fcy, fh=fh)

        return self

    def _get_Xcols(self, X):
        """Shorthand to obtain X at self.columns."""
        if self.columns is not None:
            return X[self.columns]
        else:
            return X

    def _check_unknown_exog(
        self: "ForecastX", X: typing.Optional[pd.DataFrame]
    ) -> bool:
        """Check if all future-unknown exogenous columns are present.

        Parameters
        ----------
        X : typing.Optional[pd.DataFrame]
            user input for exogeneous data in ``predict``

        Returns
        -------
        bool
            indicator of presence of all future-unknown columns in `X`
        """
        # user has not passed any `X` argument to predict call
        # obviously, future-unknown exeogenous features are absent
        if X is None:
            return False

        # get list of columns storing future-unknown exogenous features
        # either columns explicitly specified through the `columns` argument
        # or all columns in the `X` argument passed in `fit` call are future-unknown
        if self.columns is None or len(self.columns) == 0:
            # `self._X` is guaranteed to exist and be a DataFrame at this point
            # ensured by `self.X_was_None_` check in `_get_forecaster_X_prediction`
            unknown_columns = self._X.columns
        else:
            unknown_columns = self.columns

        # check if all future-unknown columns are present
        # in the `X` argument passed in `predict` call
        return all(column in X.columns for column in unknown_columns)

    def _get_forecaster_X_prediction(self, X=None, fh=None, method="predict"):
        """Shorthand to obtain a prediction from forecaster_X, depending on behaviour.

        If behaviour = "update": uses self.forecaster_X_, this is already fitted.
        If behaviour = "refitted", uses a local clone of self.forecaster_X,
            after fitting it to self._X, i.e., all exogeneous data seen so far.

        Parameters
        ----------
        X : pandas.DataFrame, optional, default=None
            exogeneous data seen in predict
        fh : ForecastingHorizon, should be the input of the predict method, optional
        method : str, optional, default="predict"
            method of forecaster to call to obtain prediction

        Returns
        -------
        X : sktime time series container
            a forecast obtained using a clone of forecaster_X, state as above
        """
        if self.X_was_None_:
            return None

        if isinstance(self.columns, (list, pd.Index)) and len(self.columns) == 0:
            return X

        # if user passes data for future unknown variables, do not forecast them
        # this is done only if predict_behaviour is "use_actuals"
        if self.predict_behaviour == "use_actuals" and self._check_unknown_exog(X):
            return X

        if self.behaviour == "update":
            forecaster = self.forecaster_X_
        elif self.behaviour == "refit":
            if self.fh_X_ is not None:
                fh = self.fh_X_
            forecaster = self.forecaster_X_c.clone()
            X_for_fcX = self._get_X_for_fcX(self._X)
            forecaster.fit(y=self._get_Xcols(self._X), fh=fh, X=X_for_fcX)

        X_for_fcX = self._get_X_for_fcX(X)
        X_pred = getattr(forecaster, method)(fh=fh, X=X_for_fcX)
        if X is not None:
            X_pred = X_pred.combine_first(X)

        # order columns so they are in the same order as in X seen
        X_cols_ordered = [col for col in self._X.columns if col in X_pred.columns]
        X_pred = X_pred[X_cols_ordered]

        return X_pred

    def _get_X_for_fcX(self, X):
        """Shorthand to obtain X for forecaster_X, depending on parameters."""
        ixx = self.forecaster_X_exogeneous
        if X is None or ixx is None or ixx == "None":
            return None

        # if columns is None, then we use all columns
        # so there is no complement
        if self.columns is None and ixx == "complement":
            return None

        # if ixx is iterable and is empty, then we use no columns
        if isinstance(ixx, (pd.Index, list)) and len(ixx) == 0:
            return None

        if ixx == "complement":
            X_for_fcX = X.drop(columns=self.columns, errors="ignore")

            if X_for_fcX.shape[1] < 1:
                return None

            return X_for_fcX

        ixx_pd = pd.Index(ixx)
        return X.loc[:, ixx_pd]

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : time series in sktime compatible format, optional, default=None
            Exogenous time series to use in prediction

        Returns
        -------
        y_pred : time series in sktime compatible format
            Point forecasts
        """
        X = self._get_forecaster_X_prediction(fh=fh, X=X)
        y_pred = self.forecaster_y_.predict(fh=fh, X=X)
        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : time series in sktime compatible format
            Target time series to which to fit the forecaster
        X : time series in sktime compatible format, optional, default=None
            Exogenous time series to which to fit the forecaster
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        if self.behaviour == "update" and X is not None:
            self.forecaster_X_.update(y=self._get_Xcols(X), update_params=update_params)
        self.forecaster_y_.update(y=y, X=X, update_params=update_params)

        return self

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction interval forecasts.

        private _predict_interval containing the core logic,
            called from predict_interval and default _predict_quantiles

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : float or list, optional (default=0.95)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        X = self._get_forecaster_X_prediction(fh=fh, X=X)
        y_pred = self.forecaster_y_.predict_interval(fh=fh, X=X, coverage=coverage)
        return y_pred

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        X = self._get_forecaster_X_prediction(fh=fh, X=X)
        y_pred = self.forecaster_y_.predict_quantiles(fh=fh, X=X, alpha=alpha)
        return y_pred

    def _predict_var(self, fh=None, X=None, cov=False):
        """Compute/return variance forecasts.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on ``cov`` variable
            If cov=False:
                Column names are exactly those of ``y`` passed in ``fit``/``update``.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.
        """
        X = self._get_forecaster_X_prediction(fh=fh, X=X)
        y_pred = self.forecaster_y_.predict_var(fh=fh, X=X, cov=cov)
        return y_pred

    # todo: does not work properly for multivariate or hierarchical
    #   still need to implement this - once interface is consolidated
    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        X = self._get_forecaster_X_prediction(fh=fh, X=X)
        y_pred = self.forecaster_y_.predict_proba(fh=fh, X=X, marginal=marginal)
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

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
        from sktime.forecasting.compose import YfromX
        from sktime.forecasting.naive import NaiveForecaster

        fs, _ = YfromX.create_test_instances_and_names()
        fx = fs[0]
        fy = fs[1]

        params1 = {"forecaster_X": fx, "forecaster_y": fy}

        # example with probabilistic capability
        if _check_soft_dependencies("pmdarima", severity="none"):
            from sktime.forecasting.arima import ARIMA

            fy_proba = ARIMA()
        else:
            fy_proba = NaiveForecaster()
        fx = NaiveForecaster()

        params2 = {"forecaster_X": fx, "forecaster_y": fy_proba, "behaviour": "refit"}

        params3 = {
            "forecaster_y": fy,
            "fit_behaviour": "use_forecast",
            "forecaster_X_exogeneous": "complement",
        }

        params4 = {"forecaster_y": fy, "predict_behaviour": "use_actuals"}

        return [params1, params2, params3, params4]


class Permute(_DelegatedForecaster, BaseForecaster, _HeterogenousMetaEstimator):
    """Permutation compositor for permuting forecasting pipeline steps.

    The compositor can be used to permute the sequence of any meta-forecaster,
    including ForecastingPipeline, TransformedTargetForecaster.

    The ``steps_arg`` parameter needs to be pointed to the "steps"-like parameter
    of the wrapped forecaster and ``permutation`` switches the sequence of steps.

    Not very useful on its own, but
    useful in combination with tuning or auto-ML wrappers on top of this.

    Parameters
    ----------
    estimator : sktime forecaster, inheriting from BaseForecaster
        must have parameter with name ``steps_arg``
        estimator whose steps are being permuted
    permutation : list of str, or None, optional, default = None
        if not None, must be equal length as getattr(estimator, steps_arg)
        and elements must be equal to names of estimator.steps_arg estimators
        names are unique names as created by _get_estimator_tuples (if unnamed list),
        or first string element of tuples, of estimator.steps_arg
        list is interpreted as range of permutation of names
        if None, is interpreted as the identity permutation
    steps_arg : string, optional, default="steps"
        name of the steps parameter. getattr(estimator, steps_arg) must be
        list of estimators, or list of (str, estimator) pairs

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.compose import ForecastingPipeline, Permute
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.series.boxcox import BoxCoxTransformer
    >>> from sktime.transformations.series.exponent import ExponentTransformer

    Simple example: permute sequence of estimator in forecasting pipeline

    >>> y = load_airline()
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> pipe = ForecastingPipeline(
    ...     [
    ...         ("boxcox", BoxCoxTransformer()),
    ...         ("exp", ExponentTransformer(3)),
    ...         ("naive", NaiveForecaster()),
    ...     ]
    ... )
    >>> # this results in the pipeline with sequence "exp", "boxcox", "naive"
    >>> permuted = Permute(pipe, ["exp", "boxcox", "naive"])
    >>> permuted = permuted.fit(y, fh=fh)
    >>> y_pred = permuted.predict()

    The permuter is useful in combination with grid search (toy example):

    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> fh = [1,2,3]
    >>> cv = ExpandingWindowSplitter(fh=fh)
    >>> forecaster = NaiveForecaster()
    >>> # check which of the two sequences of transformers is better
    >>> param_grid = {
    ...     "permutation" : [["boxcox", "exp", "naive"], ["exp", "boxcox", "naive"]]
    ... }
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=permuted,
    ...     param_grid=param_grid,
    ...     cv=cv)
    """

    _tags = {
        "authors": "aiwalter",
        "scitype:y": "both",
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "X-y-must-have-same-index": False,
    }

    _delegate_name = "estimator_"

    def __init__(self, estimator, permutation=None, steps_arg="steps"):
        self.estimator = estimator
        self.permutation = permutation
        self.steps_arg = steps_arg

        super().__init__()

        self._set_delegated_tags(estimator)

        self._set_permuted_estimator()

    def _set_permuted_estimator(self):
        """Set self.estimator_ based on permutation arg."""
        estimator = self.estimator
        permutation = self.permutation
        steps_arg = self.steps_arg

        self.estimator_ = estimator.clone()

        if permutation is not None:
            inner_estimators = getattr(estimator, steps_arg)
            estimator_tuples = self._get_estimator_tuples(inner_estimators)

            estimator_dict = {x[0]: x[1] for x in estimator_tuples}

            # check that permutation is list of str
            msg = "Error in Permutation, permutation must be None or a list of strings"
            if not isinstance(permutation, list):
                raise ValueError(msg)
            if not all(isinstance(item, str) for item in permutation):
                raise ValueError(msg)

            # check that permutation contains same step names as given in steps
            if not set(estimator_dict.keys()) == set(permutation):
                raise ValueError(
                    f"""Permutation hyperparameter permutation must contain exactly
                    the same step names as
                    the names of steps in getattr(estimator, steps_arg), but
                    found tuple names {set(estimator_dict.keys())} but got
                    permutation {set(permutation)}."""
                )

            estimator_tuples_permuted = [(k, estimator_dict[k]) for k in permutation]

            self.estimator_ = estimator.clone()
            self.estimator_.set_params(**{steps_arg: estimator_tuples_permuted})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.boxcox import BoxCoxTransformer
        from sktime.transformations.series.exponent import ExponentTransformer

        # transformers mixed with-without fit, ForecastingPipeline
        # steps are (str, estimator)
        params1 = {
            "estimator": ForecastingPipeline(
                [
                    ("foo", BoxCoxTransformer()),
                    ("bar", ExponentTransformer(3)),
                    ("foobar", NaiveForecaster()),
                ]
            ),
            "permutation": ["bar", "foo", "foobar"],
        }

        # transformers have no fit, TransformedTargetForecaster
        # steps are only estimator
        params2 = {
            "estimator": TransformedTargetForecaster(
                [ExponentTransformer(0.5), NaiveForecaster(), ExponentTransformer(3)]
            ),
            "permutation": [
                "NaiveForecaster",
                "ExponentTransformer_1",
                "ExponentTransformer_2",
            ],
        }

        return [params1, params2]

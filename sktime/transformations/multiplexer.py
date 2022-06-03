#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecaster for selecting among different model classes."""

from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations._delegate import _DelegatedTransformer
from sktime.transformations.base import BaseTransformer

__author__ = ["miraep8"]
__all__ = ["MultiplexTransformer"]


class MultiplexTransformer(_DelegatedTransformer, _HeterogenousMetaEstimator):
    """Facilitate an AutoML based selection of the best transformer.

    When used in combination with either TransformedTargetForecaster or
    ForecastingPipeline in combination with ForecastingGridSearchCV MultiplexTransformer
    provides a framework for transformer selection.  It can be used with exogenous
    transformers (ie data to be applied to exogenous X variables) or transformers on
    the y data, simply select the appropriate pipeline.

    MultiplexTransformer has 2 parameters:
    transformers, which is a list of (optionally named) transformers.  If names are not
        provided they will be automatically generated based on the transformer name.
    selected_transformer, which is a str acting as a hyper-parameter. It must match one
        of the names in transformers.  This is generally the parameter being tuned in
        the grid search.
    MultiplexTransformer delegated all transforming tasks to a copy of the transformer
    in transformers whose name matches selected_transformer.  All other transformers
    in transformers will be ignored.

    Parameters
    ----------
    transformers : list of sktime transformers, or
        list of tuples (str, estimator) of named sktime transformers
        MultiplexTransformer can switch ("multiplex") between these transformers.
        Note - all the transformers passed in "transformers" should be thought of as
        blueprints.  Calling transformation functions on MultiplexTransformer will not
        change their state at all. - Rather a copy of each is created and this is what
        is updated.
    selected_transformer: str or None, optional, Default=None.
        If str, must be one of the forecaster names.
            If passed in transformers were unnamed then selected_transformer must
            coincide with auto-generated name strings.
            To inspect auto-generated name strings, call get_params.
        If None, selected_transformer defaults to the name of the first transformer
           in transformers.
        selected_transformer represents the name of the transformer MultiplexTransformer
           should behave as (ie delegate all relevant transformation functionality to)

    Attributes
    ----------
    transformer_ : sktime transformer
        clone of the transformer named by selected_transformer to which all the
        transformation functionality is delegated to.
    transformers_ : copy of the list of transformers passed.  If transformers was
        passed without names, those will be auto-generated and put here.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.multiplexer import MultiplexTransformer
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import (
    ...     ForecastingGridSearchCV,
    ...     ExpandingWindowSplitter)
    >>> # create MultiplexTransformer:
    >>> multiplexer = MultiplexTransformer(transformers=[
    ...     ("impute_mean", Imputer(method="mean", missing_values = -1)),
    ...     ("impute_near", Imputer(method="nearest", missing_values = -1)),
    ...     ("impute_rand", Imputer(method="random", missing_values = -1))])
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
    ...     start_with_window=True,
    ...     fh=[1,2,3])
    >>> pipe = TransformedTargetForecaster(steps = [
    ...     ("multiplex", multiplexer),
    ...     ("forecaster", NaiveForecaster())
    ...     ])
    >>> gscv = ForecastingGridSearchCV(
    ...     cv=cv,
    ...     param_grid={"multiplex__selected_transformer":
    ...     ["impute_mean", "impute_near", "impute_rand"]},
    ...     forecaster=pipe,
    ...     )
    >>> y = load_shampoo_sales()
    >>> # randomly make some of the values nans:
    >>> y.loc[y.sample(frac=0.1).index] = -1
    >>> gscv = gscv.fit(y)
    """

    _tags = {}

    _delegate_name = "transformer_"

    def __init__(
        self,
        transformers: list,
        selected_transformer=None,
    ):
        super(MultiplexTransformer, self).__init__()
        self.selected_transformer = selected_transformer

        self.transformers = transformers
        self.transformers_ = self._check_estimators(
            transformers,
            attr_name="transformers",
            cls_type=BaseTransformer,
            clone_ests=False,
        )
        self._set_transformer()
        self.clone_tags(self.transformer_)
        self.set_tags(**{"fit_is_empty": False})

    def _check_selected_transformer(self):
        component_names = self._get_estimator_names(
            self.transformers_, make_unique=True
        )
        selected = self.selected_transformer
        if selected is not None and selected not in component_names:
            raise Exception(
                f"Invalid selected_forecaster parameter value provided, "
                f" found: {selected}. Must be one of these"
                f" valid selected_forecaster parameter values: {component_names}."
            )

    def _set_transformer(self):
        self._check_selected_transformer()
        # clone the selected transformer to self.transformer_
        if self.selected_transformer is not None:
            for name, transformer in self._get_estimator_tuples(self.transformers):
                if self.selected_transformer == name:
                    self.transformer_ = clone(transformer)
        else:
            # if None, simply clone the first forecaster to self.forecaster_
            self.transformer_ = clone(self._get_estimator_list(self.transformers)[0])

    def _fit(self, y, X=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self._set_transformer()
        self.clone_tags(self.transformer_)
        self.set_tags(**{"fit_is_empty": False})
        super()._fit(y=y, X=X)

        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("transformers", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params("transformers", **kwargs)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.transformations.series.impute import Imputer

        # test with 2 simple detrend transformations with selected_transformer
        params1 = {
            "transformers": [
                ("imputer_mean", Imputer(method="mean")),
                ("imputer_near", Imputer(method="nearest")),
            ],
            "selected_transformer": "imputer_near",
        }
        # test no selected_transformer
        params2 = {
            "transformers": [
                ("imputer_mean", Imputer(method="mean")),
                ("imputer_near", Imputer(method="nearest")),
            ],
        }
        return [params1, params2]

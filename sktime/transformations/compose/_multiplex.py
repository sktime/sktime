"""Multiplexer transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["miraep8", "fkiraly"]
__all__ = ["MultiplexTransformer"]

from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.transformations._delegate import _DelegatedTransformer
from sktime.transformations.base import BaseTransformer


class MultiplexTransformer(_HeterogenousMetaEstimator, _DelegatedTransformer):
    """Facilitate an AutoML based selection of the best transformer.

    When used in combination with either TransformedTargetForecaster or
    ForecastingPipeline in combination with ForecastingGridSearchCV
    MultiplexTransformer provides a framework for transformer selection.  Through
    selection of the appropriate pipeline (ie TransformedTargetForecaster vs
    ForecastingPipeline) the transformers in MultiplexTransformer will either be
    applied to exogenous data, or to the target data.

    MultiplexTransformer delegates all transforming tasks (ie, calls to fit, transform,
    inverse_transform, and update) to a copy of the transformer in transformers
    whose name matches selected_transformer.  All other transformers in transformers
    will be ignored.

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
        If str, must be one of the transformer names.
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
    _transformers : list of (name, est) tuples, where est are direct references to
        the estimators passed in transformers passed. If transformers was passed
        without names, those be auto-generated and put here.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.compose import MultiplexTransformer
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> # create MultiplexTransformer:
    >>> multiplexer = MultiplexTransformer(transformers=[
    ...     ("impute_mean", Imputer(method="mean", missing_values = -1)),
    ...     ("impute_near", Imputer(method="nearest", missing_values = -1)),
    ...     ("impute_rand", Imputer(method="random", missing_values = -1))])
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
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

    # tags will largely be copied from selected_transformer
    _tags = {
        "authors": ["miraep8", "fkiraly"],
        "fit_is_empty": False,
        "univariate-only": False,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
    }

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator) pairs for the default
    _steps_attr = "_transformers"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    _steps_fitted_attr = "transformers_"

    def __init__(
        self,
        transformers: list,
        selected_transformer=None,
    ):
        super().__init__()
        self.selected_transformer = selected_transformer

        self.transformers = transformers
        self._check_estimators(
            transformers,
            attr_name="transformers",
            cls_type=BaseTransformer,
            clone_ests=False,
        )
        self._set_transformer()
        self.clone_tags(self.transformer_)
        self.set_tags(**{"fit_is_empty": False})
        # this ensures that we convert in the inner estimator, not in the multiplexer
        self.set_tags(**{"X_inner_mtype": ALL_TIME_SERIES_MTYPES})

    @property
    def _transformers(self):
        """Forecasters turned into name/est tuples."""
        return self._get_estimator_tuples(self.transformers, clone_ests=False)

    @_transformers.setter
    def _transformers(self, value):
        self.transformers = value

    def _check_selected_transformer(self):
        component_names = self._get_estimator_names(
            self._transformers, make_unique=True
        )
        selected = self.selected_transformer
        if selected is not None and selected not in component_names:
            raise Exception(
                f"Invalid selected_transformer parameter value provided, "
                f" found: {selected}. Must be one of these"
                f" valid selected_transformer parameter values: {component_names}."
            )

    def _set_transformer(self):
        self._check_selected_transformer()
        # clone the selected transformer to self.transformer_
        if self.selected_transformer is not None:
            for name, transformer in self._get_estimator_tuples(self.transformers):
                if self.selected_transformer == name:
                    self.transformer_ = transformer.clone()
        else:
            # if None, simply clone the first transformer to self.transformer_
            self.transformer_ = self._get_estimator_list(self.transformers)[0].clone()

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
                Imputer(method="mean"),
                Imputer(method="nearest"),
            ],
        }
        return [params1, params2]

    def __or__(self, other):
        """Magic | (or) method, return (right) concatenated MultiplexTransformer.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        MultiplexTransformer object, concatenation of ``self`` (first) with ``other``
            (last).not nested, contains only non-MultiplexTransformer ``sktime``
            transformers

        Raises
        ------
        ValueError if other is not of type MultiplexTransformer or BaseTransformer.
        """
        from sktime.registry import coerce_scitype

        other = coerce_scitype(other, "transformer")
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=MultiplexTransformer,
            attr_name="transformers",
            concat_order="left",
        )

    def __ror__(self, other):
        """Magic | (or) method, return (left) concatenated MultiplexTransformer.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        MultiplexTransformer object, concatenation of ``self`` (last) with ``other``
            (first). not nested, contains only non-MultiplexTransformer ``sktime``
            transformers
        """
        from sktime.registry import coerce_scitype

        other = coerce_scitype(other, "transformer")
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=MultiplexTransformer,
            attr_name="forecasters",
            concat_order="right",
        )

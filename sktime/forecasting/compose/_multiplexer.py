#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecaster for selecting among different model classes."""

from sktime.base import _HeterogenousMetaEstimator
from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster

__author__ = ["kkoralturk", "aiwalter", "fkiraly", "miraep8"]
__all__ = ["MultiplexForecaster"]


class MultiplexForecaster(_HeterogenousMetaEstimator, _DelegatedForecaster):
    """MultiplexForecaster for selecting among different models in Auto-ML pipelines.

    ``MultiplexForecaster`` facilitates a framework for performing
    automated model selection process over different model classes.
    It should be used in conjunction with ``ForecastingGridSearchCV`` or similar tuners
    to build an Auto-ML pipeline for forecasters.
    ``MultiplexForecaster`` can be used with univariate and multivariate forecasters.

    ``MultiplexForecaster`` is specified with a (named) list of forecasters
    and a selected_forecaster hyper-parameter, which is one of the forecaster names.
    The ``MultiplexForecaster`` then behaves precisely as the forecaster with
    name ``selected_forecaster``, ignoring functionality in the other forecasters.

    When used with ``ForecastingGridSearchCV``, ``MultiplexForecaster``
    provides an ability to tune across multiple estimators, i.e., to perform Auto-ML,
    by tuning the ``,selected_forecaster``, hyper-parameter. This combination will then
    select one of the passed forecasters via the tuning algorithm.

    Parameters
    ----------
    forecasters : list of sktime forecasters, or
        list of tuples (str, estimator) of sktime forecasters
        ``MultiplexForecaster`` can switch ("multiplex") between these forecasters.
        These are "blueprint" forecasters, states do not change when ``fit`` is called.

    selected_forecaster: str or None, optional, Default=None.
        Name of the forecaster to be selected from the list of forecasters.

        * If str, must be one of the forecaster names.
          If no names are provided, must coincide with auto-generated name strings.
          To inspect auto-generated name strings, call ``get_params``.
        * If None, behaves as if the first forecaster in the list is selected.
          Selects the forecaster as which ``MultiplexForecaster`` behaves.

    Attributes
    ----------
    forecaster_ : sktime forecaster
        clone of the selected forecaster used for fitting and forecasting.
    _forecasters : list of (str, forecaster) tuples
        str are identical to those passed, if passed strings are unique
        otherwise unique strings are generated from class name; if not unique,
        the string ``_[i]`` is appended where ``[i]`` is count of occurrence up until
        then

    Examples
    --------
    >>> from sktime.forecasting.ets import AutoETS
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.forecasting.compose import MultiplexForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.datasets import load_shampoo_sales
    >>> y = load_shampoo_sales()
    >>> forecaster = MultiplexForecaster(forecasters=[
    ...     ("ets", AutoETS()),
    ...     ("theta", ThetaForecaster()),
    ...     ("naive", NaiveForecaster())])  # doctest: +SKIP
    >>> cv = ExpandingWindowSplitter(step_length=12)  # doctest: +SKIP
    >>> gscv = ForecastingGridSearchCV(
    ...     cv=cv,
    ...     param_grid={"selected_forecaster":["ets", "theta", "naive"]},
    ...     forecaster=forecaster)  # doctest: +SKIP
    >>> gscv.fit(y)  # doctest: +SKIP
    ForecastingGridSearchCV(...)
    """

    _tags = {
        "authors": ["kkoralturk", "aiwalter", "fkiraly", "miraep8"],
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "scitype:y": "both",
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "fit_is_empty": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods to those of same name in self.forecaster_
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_forecasters"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "forecasters_"

    def __init__(
        self,
        forecasters: list,
        selected_forecaster=None,
    ):
        super().__init__()
        self.selected_forecaster = selected_forecaster

        self.forecasters = forecasters
        self._check_estimators(
            forecasters,
            attr_name="forecasters",
            cls_type=BaseForecaster,
            clone_ests=False,
        )
        self._set_forecaster()

        self._set_delegated_tags()
        self.set_tags(**{"fit_is_empty": False})
        # this ensures that we convert in the inner estimator, not in the multiplexer
        self.set_tags(**{"y_inner_mtype": ALL_TIME_SERIES_MTYPES})
        self.set_tags(**{"X_inner_mtype": ALL_TIME_SERIES_MTYPES})

    @property
    def _forecasters(self):
        """Forecasters turned into name/est tuples."""
        return self._get_estimator_tuples(self.forecasters, clone_ests=False)

    @_forecasters.setter
    def _forecasters(self, value):
        self.forecasters = value

    def _check_selected_forecaster(self):
        component_names = self._get_estimator_names(self._forecasters, make_unique=True)
        selected = self.selected_forecaster
        if selected is not None and selected not in component_names:
            raise Exception(
                f"Invalid selected_forecaster parameter value provided, "
                f" found: {self.selected_forecaster}. Must be one of these"
                f" valid selected_forecaster parameter values: {component_names}."
            )

    def __or__(self, other):
        """Magic | (or) method, return (right) concatenated MultiplexForecaster.

        Implemented for ``other`` being a forecaster, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` forecaster, must inherit from BaseForecaster
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        MultiplexForecaster object, concatenation of ``self`` (first) with ``other``
        (last).
            not nested, contains only non-MultiplexForecaster ``sktime`` forecasters

        Raises
        ------
        ValueError if other is not of type MultiplexForecaster or BaseForecaster.
        """
        return self._dunder_concat(
            other=other,
            base_class=BaseForecaster,
            composite_class=MultiplexForecaster,
            attr_name="forecasters",
            concat_order="left",
        )

    def __ror__(self, other):
        """Magic | (or) method, return (left) concatenated MultiplexForecaster.

        Implemented for ``other`` being a forecaster, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` forecaster, must inherit from BaseForecaster
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        MultiplexForecaster object, concatenation of ``self`` (last) with ``other``
        (first).
            not nested, contains only non-MultiplexForecaster ``sktime`` forecasters
        """
        return self._dunder_concat(
            other=other,
            base_class=BaseForecaster,
            composite_class=MultiplexForecaster,
            attr_name="forecasters",
            concat_order="right",
        )

    def _set_forecaster(self):
        self._check_selected_forecaster()
        # clone the selected forecaster to self.forecaster_
        if self.selected_forecaster is not None:
            for name, forecaster in self._get_estimator_tuples(self.forecasters):
                if self.selected_forecaster == name:
                    self.forecaster_ = forecaster.clone()
        else:
            # if None, simply clone the first forecaster to self.forecaster_
            self.forecaster_ = self._get_estimator_list(self.forecasters)[0].clone()

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
        from sktime.forecasting.naive import NaiveForecaster

        params1 = {
            "forecasters": [
                ("Naive_mean", NaiveForecaster(strategy="mean")),
                ("Naive_last", NaiveForecaster(strategy="last")),
                ("Naive_drift", NaiveForecaster(strategy="drift")),
            ],
            "selected_forecaster": "Naive_last",
        }
        params2 = {
            "forecasters": [
                NaiveForecaster(strategy="last"),
                NaiveForecaster(strategy="mean"),
                NaiveForecaster(strategy="drift"),
            ],
        }
        return [params1, params2]

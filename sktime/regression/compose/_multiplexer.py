#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements regressor for selecting among different model classes."""
# based on MultiplexForecaster

from sktime.base import _HeterogenousMetaEstimator
from sktime.datatypes import MTYPE_LIST_PANEL, MTYPE_LIST_TABLE
from sktime.regression._delegate import _DelegatedRegressor
from sktime.regression.base import BaseRegressor

__author__ = ["ksharma6"]
__all__ = ["MultiplexRegressor"]


class MultiplexRegressor(_HeterogenousMetaEstimator, _DelegatedRegressor):
    """MultiplexRegressor for selecting among different models.

    MultiplexRegressor facilitates a framework for performing
    model selection process over different model classes.
    It should be used in conjunction with GridSearchCV to get full utilization.
    It can be used with univariate and multivariate regressors,
    single-output and multi-output regressors.

    MultiplexRegressor is specified with a (named) list of regressors
    and a selected_regressor hyper-parameter, which is one of the regressor names.
    The MultiplexRegressor then behaves precisely as the regressor with
    name selected_regressor, ignoring functionality in the other regressors.

    When used with GridSearchCV, MultiplexRegressor
    provides an ability to tune across multiple estimators, i.e., to perform AutoML,
    by tuning the selected_regressor hyper-parameter. This combination will then
    select one of the passed regressors via the tuning algorithm.

    Parameters
    ----------
    regressors : list of sktime regressors, or
        list of tuples (str, estimator) of sktime regressors
        MultiplexRegressor can switch ("multiplex") between these regressors.
        These are "blueprint" regressors, states do not change when `fit` is called.
    selected_regressor: str or None, optional, Default=None.
        If str, must be one of the regressor names.
            If no names are provided, must coincide with auto-generated name strings.
            To inspect auto-generated name strings, call get_params.
        If None, behaves as if the first regressor in the list is selected.
        Selects the regressor as which MultiplexRegressor behaves.

    Attributes
    ----------
    regressor_ : sktime regressor
        clone of the selected regressor used for fitting and regression.
    _regressors : list of (str, regressor) tuples
        str are identical to those passed, if passed strings are unique
        otherwise unique strings are generated from class name; if not unique,
        the string `_[i]` is appended where `[i]` is count of occurrence up until then
    """

    _tags = {
        "authors": ["ksharma6"],
        "capability:multioutput": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "X_inner_mtype": MTYPE_LIST_PANEL,
        "y_inner_mtype": MTYPE_LIST_TABLE,
        "fit_is_empty": False,
    }

    # attribute for _DelegatedRegressor, which then delegates
    #     all non-overridden methods to those of same name in self.regressor_
    #     see further details in _DelegatedRegressor docstring
    _delegate_name = "regressor_"

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_regressors"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "regressors_"

    def __init__(
        self,
        regressors: list,
        selected_regressor=None,
    ):
        super().__init__()
        self.selected_regressor = selected_regressor

        self.regressors = regressors
        self._check_estimators(
            regressors,
            attr_name="regressors",
            cls_type=BaseRegressor,
            clone_ests=False,
        )
        self._set_regressor()

        self.clone_tags(self.regressor_)
        self.set_tags(**{"fit_is_empty": False})
        # this ensures that we convert in the inner estimator, not in the multiplexer
        self.set_tags(**{"X_inner_mtype": MTYPE_LIST_PANEL})
        self.set_tags(**{"y_inner_mtype": MTYPE_LIST_TABLE})

    @property
    def _regressors(self):
        """Regressors turned into name/est tuples."""
        return self._get_estimator_tuples(self.regressors, clone_ests=False)

    @_regressors.setter
    def _regressors(self, value):
        self.regressors = value

    def _check_selected_regressor(self):
        component_names = self._get_estimator_names(self._regressors, make_unique=True)
        selected = self.selected_regressor
        if selected is not None and selected not in component_names:
            raise Exception(
                f"Invalid selected_regressor parameter value provided, "
                f" found: {self.selected_regressor}. Must be one of these"
                f" valid selected_regressor parameter values: {component_names}."
            )

    def __or__(self, other):
        """Magic | (or) method, return (right) concatenated MultiplexRegressor.

        Implemented for `other` being a regressor, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` regressor, must inherit from BaseRegressor
            otherwise, `NotImplemented` is returned

        Returns
        -------
        MultiplexRegressor object, concatenation of `self` (first) with `other` (last).
            not nested, contains only non-MultiplexRegressor `sktime` regressors

        Raises
        ------
        ValueError if other is not of type MultiplexRegressor or BaseRegressor.
        """
        return self._dunder_concat(
            other=other,
            base_class=BaseRegressor,
            composite_class=MultiplexRegressor,
            attr_name="regressors",
            concat_order="left",
        )

    def __ror__(self, other):
        """Magic | (or) method, return (left) concatenated MultiplexRegressor.

        Implemented for `other` being a regressor, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` regressor, must inherit from BaseRegressor
            otherwise, `NotImplemented` is returned

        Returns
        -------
        MultiplexRegressor object, concatenation of `self` (last) with `other` (first).
            not nested, contains only non-MultiplexRegressor `sktime` regressors
        """
        return self._dunder_concat(
            other=other,
            base_class=BaseRegressor,
            composite_class=MultiplexRegressor,
            attr_name="regressors",
            concat_order="right",
        )

    def _set_regressor(self):
        self._check_selected_regressor()
        # clone the selected regressor to self.regressor_
        if self.selected_regressor is not None:
            for name, regressor in self._get_estimator_tuples(self.regressors):
                if self.selected_regressor == name:
                    self.regressor_ = regressor.clone()
        else:
            # if None, simply clone the first regressor to self.regressor_
            self.regressor_ = self._get_estimator_list(self.regressors)[0].clone()

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
        from sktime.regression.dummy import DummyRegressor

        params1 = {
            "regressors": [
                ("Naive_mean", DummyRegressor(strategy="mean")),
                ("Naive_median", DummyRegressor(strategy="median")),
                ("Naive_quantile", DummyRegressor(strategy="quantile")),
                ("Naive_constant", DummyRegressor(strategy="constant")),
            ],
            "selected_regressor": "Naive_mean",
        }
        params2 = {
            "regressors": [
                DummyRegressor(strategy="mean"),
                DummyRegressor(strategy="median"),
                DummyRegressor(strategy="quantile"),
                DummyRegressor(strategy="constant"),
            ],
        }
        return [params1, params2]

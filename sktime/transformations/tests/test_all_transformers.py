# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests common to all transformers."""

__author__ = ["mloning"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes._panel._check import is_nested_dataframe
from sktime.registry import all_estimators
from sktime.tests._config import VALID_TRANSFORMER_TYPES
from sktime.transformations.base import (
    BaseTransformer,
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)
from sktime.utils._testing.estimator_checks import (
    _assert_array_almost_equal,
    _has_capability,
    _make_args,
)

ALL_TRANSFORMERS = all_estimators(estimator_types="transformer", return_names=False)


@pytest.mark.parametrize("Estimator", ALL_TRANSFORMERS)
def test_all_transformers(Estimator):
    check_transformer(Estimator)


def check_transformer(Estimator):
    for check in _yield_transformer_checks(Estimator):
        check(Estimator)


def _construct_fit_transform(Estimator, **kwargs):
    estimator = Estimator.create_test_instance()

    # For forecasters which are also transformations (e.g. pipelines), we cannot
    # the forecasting horizon to transform, so we only return the first two
    # arguments here. Note that this will fail for forecasters which require the
    # forecasting horizon in fit.
    args = _make_args(estimator, "fit", **kwargs)[:2]
    return estimator.fit_transform(*args)


def _construct_fit(Estimator, **kwargs):
    estimator = Estimator.create_test_instance()
    args = _make_args(estimator, "fit", **kwargs)[:2]
    return estimator.fit(*args)


def check_series_to_primitive_transform_univariate(Estimator, **kwargs):
    out = _construct_fit_transform(Estimator, **kwargs)
    assert isinstance(out, pd.DataFrame)


def _check_raises_error(Estimator, **kwargs):
    with pytest.raises(ValueError, match=r"univariate"):
        estimator = Estimator.create_test_instance()
        if estimator.get_tag("fit-in-transform", False):
            # As some estimators have an empty fit method, we here check if
            # they raise the appropriate error in transform rather than fit.
            _construct_fit_transform(Estimator, **kwargs)
        else:
            # All other estimators should raise the error in fit.
            _construct_fit(Estimator, **kwargs)


def check_series_to_primitive_transform_multivariate(Estimator):
    n_columns = 3
    estimator = Estimator.create_test_instance()
    if estimator.get_tag("univariate-only", False):
        _check_raises_error(Estimator, n_columns=n_columns)
    else:
        out = _construct_fit_transform(Estimator, n_columns=n_columns)
        assert isinstance(out, pd.DataFrame)
        # We don't know how many primitives wil be returned, so # of columns unknown.
        assert out.shape[0] == n_columns


def check_series_to_series_transform_univariate(Estimator):
    estimator = Estimator.create_test_instance()
    n_timepoints = 15
    out = _construct_fit_transform(
        Estimator,
        n_timepoints=n_timepoints,
        add_nan=estimator.get_tag("handles-missing-data", False),
    )
    assert isinstance(out, (pd.Series, np.ndarray, pd.DataFrame))


def check_series_to_series_transform_multivariate(Estimator):
    estimator = Estimator.create_test_instance()
    n_columns = 3
    n_timepoints = 15
    if estimator.get_tag("univariate-only", False):
        _check_raises_error(Estimator, n_timepoints=n_timepoints, n_columns=n_columns)
    else:
        out = _construct_fit_transform(
            Estimator, n_timepoints=n_timepoints, n_columns=n_columns
        )
        assert isinstance(out, (pd.DataFrame, np.ndarray))
        if estimator.get_tag("transform-returns-same-time-index"):
            assert out.shape[0] == n_timepoints


def check_panel_to_tabular_transform_univariate(Estimator):
    n_instances = 5
    out = _construct_fit_transform(Estimator, n_instances=n_instances)
    assert isinstance(out, (pd.DataFrame, np.ndarray))
    assert out.shape[0] == n_instances


def check_panel_to_tabular_transform_multivariate(Estimator):
    estimator = Estimator.create_test_instance()
    n_instances = 5
    if estimator.get_tag("univariate-only", False):
        _check_raises_error(Estimator, n_instances=n_instances, n_columns=3)
    else:
        out = _construct_fit_transform(Estimator, n_instances=n_instances, n_columns=3)
        assert isinstance(out, (pd.DataFrame, np.ndarray))
        assert out.shape[0] == n_instances


def check_panel_to_panel_transform_univariate(Estimator):
    n_instances = 5
    out = _construct_fit_transform(Estimator, n_instances=n_instances)
    assert isinstance(out, (pd.DataFrame, np.ndarray))
    assert out.shape[0] == n_instances
    if isinstance(out, np.ndarray):
        assert out.ndim == 3
    if isinstance(out, pd.DataFrame):
        assert is_nested_dataframe(out)


def check_panel_to_panel_transform_multivariate(Estimator):
    estimator = Estimator.create_test_instance()
    n_instances = 5
    if estimator.get_tag("univariate-only", False):
        _check_raises_error(Estimator, n_instances=n_instances, n_columns=3)
    else:
        out = _construct_fit_transform(Estimator, n_instances=n_instances, n_columns=3)
        assert isinstance(out, (pd.DataFrame, np.ndarray))
        assert out.shape[0] == n_instances
        if isinstance(out, np.ndarray):
            assert out.ndim == 3
        if isinstance(out, pd.DataFrame):
            assert is_nested_dataframe(out)


def check_transform_returns_same_time_index(Estimator):
    estimator = Estimator.create_test_instance()
    if estimator.get_tag("transform-returns-same-time-index"):
        assert issubclass(Estimator, (_SeriesToSeriesTransformer, BaseTransformer))
        estimator = Estimator.create_test_instance()
        fit_args = _make_args(estimator, "fit")
        estimator.fit(*fit_args)
        for method in ["transform", "inverse_transform"]:
            if _has_capability(estimator, method):
                X = _make_args(estimator, method)[0]
                Xt = estimator.transform(X)
                np.testing.assert_array_equal(X.index, Xt.index)


def check_transform_inverse_transform_equivalent(Estimator):
    estimator = Estimator.create_test_instance()
    X = _make_args(estimator, "fit")[0]
    Xt = estimator.fit_transform(X)
    Xit = estimator.inverse_transform(Xt)
    if estimator.get_tag("transform-returns-same-time-index"):
        _assert_array_almost_equal(X, Xit)
    else:
        _assert_array_almost_equal(X.loc[Xit.index], Xit)


def check_transformer_type(Estimator):
    assert issubclass(Estimator, BaseTransformer)
    assert issubclass(Estimator, VALID_TRANSFORMER_TYPES)


all_transformer_checks = [check_transformer_type]
series_to_primitive_checks = [
    check_series_to_primitive_transform_univariate,
    check_series_to_primitive_transform_multivariate,
]
series_to_series_checks = [
    check_series_to_series_transform_univariate,
    check_series_to_series_transform_multivariate,
]
panel_to_tabular_checks = [
    check_panel_to_tabular_transform_univariate,
    check_panel_to_tabular_transform_multivariate,
]
panel_to_panel_checks = [
    check_panel_to_panel_transform_univariate,
    check_panel_to_panel_transform_multivariate,
]

OLD_TRAFO_CLASSES = (
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
    _PanelToTabularTransformer,
    _PanelToPanelTransformer,
)


def _yield_transformer_checks(Estimator):
    yield from all_transformer_checks
    if _has_capability(Estimator, "inverse_transform"):
        yield check_transform_inverse_transform_equivalent
    if issubclass(Estimator, _SeriesToPrimitivesTransformer):
        yield from series_to_primitive_checks
    if issubclass(Estimator, _SeriesToSeriesTransformer):
        yield from series_to_series_checks
    if issubclass(Estimator, _PanelToTabularTransformer):
        yield from panel_to_tabular_checks
    if issubclass(Estimator, _PanelToPanelTransformer):
        yield from panel_to_panel_checks
    if not issubclass(Estimator, OLD_TRAFO_CLASSES):
        if Estimator.get_class_tag("scitype:transform-output") == "Primitives":
            yield from series_to_primitive_checks
            # yield from panel_to_tabular_checks
        if Estimator.get_class_tag("scitype:transform-output") == "Series":
            yield from series_to_series_checks
            # yield from panel_to_panel_checks

    if Estimator.get_class_tag("transform-returns-same-time-index", False):
        yield check_transform_returns_same_time_index

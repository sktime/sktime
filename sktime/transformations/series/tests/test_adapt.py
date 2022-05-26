# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for adapter transformers."""

import pytest

from sktime.transformations.series.adapt import PandasTransformAdaptor
from sktime.utils._testing.series import _make_series

params_list = PandasTransformAdaptor.get_test_params()

apply_to_str = ["call", "all", "all_subset"]


@pytest.mark.parametrize("apply_to", apply_to_str)
@pytest.mark.parametrize("param", params_list)
def test_pandastransformadaptor_consistency(param, apply_to):
    """Test that pandas.method and PandasTransformAdaptor.fit.transform is the same."""
    X = _make_series(n_columns=2)

    X_fit = X[:12]
    X_trafo = X[12:]

    kwargs = param.get("kwargs")
    if kwargs is None:
        kwargs = {}

    method = param.get("method")

    trafo = PandasTransformAdaptor(method=method, kwargs=kwargs, apply_to=apply_to)

    trafo_result = trafo.fit(X_fit).transform(X_trafo)

    if apply_to == "call":
        expected_result = getattr(X_trafo, method)(**kwargs)
    elif apply_to == "all":
        expected_result = getattr(X, method)(**kwargs)
    elif apply_to == "all_subset":
        expected_result = getattr(X, method)(**kwargs)[12:]

    assert expected_result.equals(trafo_result)

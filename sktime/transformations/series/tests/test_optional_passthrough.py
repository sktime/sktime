# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for OptionalPassthrough transformer."""

import pytest
from pandas.testing import assert_series_equal

from sktime.transformations.series.compose import OptionalPassthrough
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.utils._testing.series import _make_series


@pytest.mark.parametrize("passthrough", [True, False])
def test_passthrough(passthrough):
    """Test that pandas.method and PandasTransformAdaptor.fit.transform is the same."""
    y = _make_series(n_columns=1)

    transformer = OptionalPassthrough(ExponentTransformer(), passthrough=passthrough)
    y_hat = transformer.fit_transform(y)
    y_inv = transformer.inverse_transform(y_hat)

    assert_series_equal(y, y_inv)

    if passthrough:
        assert_series_equal(y, y_hat)
    else:
        with pytest.raises(AssertionError):
            assert_series_equal(y, y_hat)

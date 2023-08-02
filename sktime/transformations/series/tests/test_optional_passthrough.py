# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for OptionalPassthrough transformer."""

import pytest
from pandas.testing import assert_series_equal

from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.utils._testing.series import _make_series


@pytest.mark.parametrize("passthrough", [True, False])
def test_passthrough(passthrough):
    """Test that passthrough works as expected."""
    y = _make_series(n_columns=1)

    optional_passthourgh = OptionalPassthrough(
        BoxCoxTransformer(), passthrough=passthrough
    )
    box_cox = BoxCoxTransformer()

    y_hat_passthrough = optional_passthourgh.fit_transform(y)
    y_inv_passthrough = optional_passthourgh.inverse_transform(y_hat_passthrough)

    y_hat_boxcox = box_cox.fit_transform(y)
    y_inv_boxcox = box_cox.inverse_transform(y_hat_boxcox)

    assert_series_equal(y, y_inv_passthrough)

    if passthrough:
        assert_series_equal(y, y_hat_passthrough)
        with pytest.raises(AssertionError):
            assert_series_equal(y_hat_boxcox, y_hat_passthrough)
    else:
        assert_series_equal(y_hat_passthrough, y_hat_boxcox)
        assert_series_equal(y_inv_passthrough, y_inv_boxcox)

        with pytest.raises(AssertionError):
            assert_series_equal(y, y_hat_passthrough)

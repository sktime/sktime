"""Tests for the capability:update tag on transformers.

Covers the contract introduced in issue #9548:
    - BaseTransformer defaults capability:update to False.
    - Transformers with a meaningful _update method set capability:update to True.
    - If fit_is_empty is True, capability:update must be False (no-op).
"""

__author__ = ["AnimeshPatra2005"]
__all__ = []

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.lag import Lag
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_base_transformer_update_tag_default():
    """Test that BaseTransformer defaults capability:update to False.

    BaseTransformer._tags should have "capability:update" = False.
    This ensures that transformers which do not implement _update
    are not falsely reported as supporting update.
    """
    from sktime.transformations.base import BaseTransformer

    tag_val = BaseTransformer._tags.get("capability:update")
    assert tag_val is False, (
        "BaseTransformer default for 'capability:update' should be False, "
        f"but got {tag_val!r}"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
@pytest.mark.parametrize(
    "transformer_class",
    [Detrender, Deseasonalizer, Lag],
)
def test_transformer_update_tag_is_true(transformer_class):
    """Test that transformers with _update set capability:update to True.

    Detrender, Deseasonalizer, and Lag all implement meaningful _update logic,
    so their capability:update tag should be True.
    """
    tag_val = transformer_class._tags.get("capability:update")
    assert tag_val is True, (
        f"{transformer_class.__name__} has a meaningful _update method, "
        f"so 'capability:update' should be True, but got {tag_val!r}"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations"),
    reason="run test only if anything in sktime.transformations module has changed",
)
def test_update_tag_false_implies_noop():
    """Test that a transformer with capability:update=False does not change state.

    If a transformer declares capability:update=False, calling update()
    should be a no-op: the transformer state must not change.
    We verify this by checking the transformer returns self and
    that transform output is identical before and after update.
    """
    from sktime.transformations.series.exponent import ExponentTransformer

    # ExponentTransformer has fit_is_empty=True and capability:update=False
    est = ExponentTransformer()
    assert not est.get_class_tag("capability:update", tag_value_default=False), (
        "ExponentTransformer should have capability:update=False"
    )

    X_train = _make_series(n_timepoints=20)
    X_new = _make_series(n_timepoints=5)

    est.fit(X_train)
    Xt_before = est.transform(X_train)

    # update should be a no-op and return self
    result = est.update(X_new)
    assert result is est, "update() should return self"

    Xt_after = est.transform(X_train)
    assert (Xt_before == Xt_after).all().all(), (
        "Transform output should not change after update() on a "
        "transformer with capability:update=False"
    )

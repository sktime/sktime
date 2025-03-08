"""Test function for TabTransformerRegressor"""

__author__ = ["Ankit-1204"]

import pytest

from sktime.regression.deep_learning.tab_transformer import TabTransformerRegressor
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TabTransformerRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tab_tranformer():
    pass

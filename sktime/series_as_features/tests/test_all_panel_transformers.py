# -*- coding: utf-8 -*-
"""Unit tests for classifier/regressor input output."""

__author__ = ["mloning", "TonyBagnall"]
__all__ = [
    "test_3d_numpy_input",
]

import pytest

from sktime.registry import all_estimators
from sktime.tests._config import EXCLUDE_ESTIMATORS, NON_STATE_CHANGING_METHODS
from sktime.transformations.base import (
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
)
from sktime.utils._testing.estimator_checks import _has_capability, _make_args

PANEL_TRANSFORMERS = all_estimators(
    estimator_types=[_PanelToPanelTransformer, _PanelToTabularTransformer],
    return_names=False,
    exclude_estimators=EXCLUDE_ESTIMATORS,
)

PANEL_ESTIMATORS = PANEL_TRANSFORMERS

# We here only check the ouput for a single number of classes
N_CLASSES = 3


@pytest.mark.parametrize("Estimator", PANEL_TRANSFORMERS)
def test_3d_numpy_input(Estimator):
    """Test classifiers handle 3D numpy input correctly."""
    estimator = Estimator.create_test_instance()
    fit_args = _make_args(estimator, "fit", return_numpy=True)
    estimator.fit(*fit_args)

    for method in NON_STATE_CHANGING_METHODS:
        if _has_capability(estimator, method):

            # try if methods can handle 3d numpy input data
            try:
                args = _make_args(estimator, method, return_numpy=True)
                getattr(estimator, method)(*args)

            # if not, check if they raise the appropriate error message
            except ValueError as e:
                error_msg = "This method requires X to be a nested pd.DataFrame"
                assert error_msg in str(e), (
                    f"{estimator.__class__.__name__} does "
                    f"not handle 3d numpy input data correctly"
                )

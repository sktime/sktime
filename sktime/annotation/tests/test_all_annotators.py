"""Tests for sktime annotators."""

__author__ = ["miraep8", "fkiraly", "klam-data", "pyyim", "mgorlin"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.registry import all_estimators
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.annotation import make_annotation_problem

ALL_ANNOTATORS = all_estimators(estimator_types="series-annotator", return_names=False)


@pytest.mark.parametrize("Estimator", ALL_ANNOTATORS)
def test_output_type(Estimator):
    """Test annotator output type."""
    if not run_test_for_class(Estimator):
        return None

    estimator = Estimator.create_test_instance()

    arg = make_annotation_problem(
        n_timepoints=50, estimator_type=estimator.get_tag("distribution_type")
    )
    estimator.fit(arg)
    arg = make_annotation_problem(
        n_timepoints=10, estimator_type=estimator.get_tag("distribution_type")
    )
    y_pred = estimator.predict(arg)
    assert isinstance(y_pred, (pd.Series, np.ndarray))

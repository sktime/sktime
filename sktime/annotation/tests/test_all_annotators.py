# -*- coding: utf-8 -*-
"""Tests for sktime annotators."""

import numpy as np
import pandas as pd
import pytest

from sktime.registry import all_estimators
from sktime.utils._testing.annotation import make_annotation_problem
from sktime.utils.validation._dependencies import _check_estimator_deps

ALL_ANNOTATORS = all_estimators(estimator_types="series-annotator", return_names=False)


@pytest.mark.parametrize("Estimator", ALL_ANNOTATORS)
def test_output_type(Estimator):
    """Test annotator output type."""
    if not _check_estimator_deps(Estimator, severity="none"):
        return None

    estimator = Estimator.create_test_instance()

    arg = make_annotation_problem(n_timepoints=50)
    estimator.fit(arg)
    arg = make_annotation_problem(n_timepoints=10)
    y_pred = estimator.predict(arg)
    assert isinstance(y_pred, (pd.Series, np.ndarray))

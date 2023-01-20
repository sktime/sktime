# -*- coding: utf-8 -*-
"""Tests for sktime annotators."""

import numpy as np
import pandas as pd
import pytest

from sktime.annotation.datagen import piecewise_poisson
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

    if estimator.get_tag("distribution_type") == "Poisson":
        arg = piecewise_poisson(lambdas=[1, 2, 3], lengths=[2, 4, 8], random_state=42)
        estimator.fit(arg)
        arg = piecewise_poisson(lambdas=[1, 3, 6], lengths=[2, 4, 8], random_state=42)
        y_pred = estimator.predict(arg)
        assert isinstance(y_pred, (pd.Series, np.ndarray))

    else:
        arg = make_annotation_problem(n_timepoints=50)
        estimator.fit(arg)
        arg = make_annotation_problem(n_timepoints=10)
        y_pred = estimator.predict(arg)
        assert isinstance(y_pred, (pd.Series, np.ndarray))

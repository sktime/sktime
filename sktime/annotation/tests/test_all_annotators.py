# -*- coding: utf-8 -*-
import pandas as pd

import pytest
from sktime.utils._testing.estimator_checks import _make_args
from sktime.registry import all_estimators

ALL_ANNOTATORS = all_estimators(estimator_types="series-annotator", return_names=False)


@pytest.mark.parametrize("Estimator", ALL_ANNOTATORS)
def test_output_type(Estimator):
    """Test annotator output type."""
    estimator = Estimator.create_test_instance()

    args = _make_args(estimator, "fit")
    estimator.fit(*args)
    args = _make_args(estimator, "predict")
    y_pred = estimator.predict(*args)
    assert isinstance(y_pred, pd.Series)

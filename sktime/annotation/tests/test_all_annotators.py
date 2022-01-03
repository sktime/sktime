# -*- coding: utf-8 -*-
import pandas as pd
from sktime.annotation.tests._config import FMT, LABELS
import pytest
from sktime.utils._testing.estimator_checks import _construct_instance, _make_args
from sktime.registry import all_estimators

ALL_ANNOTATORS = all_estimators(estimator_types="series-annotator", return_names=False)


@pytest.mark.parametrize("Estimator", ALL_ANNOTATORS)
@pytest.mark.parametrize("fmt", FMT)
@pytest.mark.parametrize("labels", LABELS)
def test_output_type(Estimator, fmt, labels):
    estimator = _construct_instance(Estimator)
    estimator.set_params(**{"fmt": fmt, "labels": labels})
    args = _make_args(estimator, "fit")
    estimator.fit(*args)
    args = _make_args(estimator, "predict")
    y_pred = estimator.predict(*args)

    if labels == "indicator":
        assert (y_pred.dtype == bool) or (y_pred.dtype == int)

    assert isinstance(y_pred, pd.Series)

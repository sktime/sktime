# -*- coding: utf-8 -*-
"""Tests for probabilistic quantiles."""
import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting.probabilistic import (
    ConstraintViolation,
    EmpiricalCoverage,
    PinballLoss,
)
from sktime.utils._testing.series import _make_series

warnings.filterwarnings("ignore", category=FutureWarning)

all_metrics = [PinballLoss, EmpiricalCoverage, ConstraintViolation]

quantile_metrics = [
    PinballLoss,
]

interval_metrics = [
    EmpiricalCoverage,
    ConstraintViolation,
]

y_uni = _make_series(n_columns=1)
y_train_uni, y_test_uni = temporal_train_test_split(y_uni)
fh_uni = np.arange(len(y_test_uni)) + 1
f_uni = ColumnEnsembleForecaster(ThetaForecaster(sp=12))
f_uni.fit(y_train_uni)

y_multi = _make_series(n_columns=3)
y_train_multi, y_test_multi = temporal_train_test_split(y_multi)
fh_multi = np.arange(len(y_test_multi)) + 1
f_multi = ColumnEnsembleForecaster(ThetaForecaster(sp=12))
f_multi.fit(y_train_multi)
"""
Cases we need to test
score average = TRUE/FALSE
multivariable = TRUE/FALSE
multiscores = TRUE/FALSE

Data types
Univariate and single score
Univariate and multi score
Multivariate and single score
Multivariate and multiscor

For each of the data types we need to test with score average = T/F \
    and multioutput with "raw_values" and "uniform_average"
"""
QUANTILE_PRED_UNI_S = f_uni.predict_quantiles(fh=fh_uni, alpha=[0.5])
INTERVAL_PRED_UNI_S = f_uni.predict_interval(fh=fh_uni, coverage=0.9)
QUANTILE_PRED_UNI_M = f_uni.predict_quantiles(fh=fh_uni, alpha=[0.05, 0.5, 0.95])
INTERVAL_PRED_UNI_M = f_uni.predict_interval(fh=fh_uni, coverage=[0.7, 0.8, 0.9, 0.99])

uni_data = [
    QUANTILE_PRED_UNI_S,
    INTERVAL_PRED_UNI_S,
    QUANTILE_PRED_UNI_M,
    INTERVAL_PRED_UNI_M,
]
QUANTILE_PRED_MULTI_S = f_multi.predict_quantiles(fh=fh_multi, alpha=[0.5])
INTERVAL_PRED_MULTI_S = f_multi.predict_interval(fh=fh_multi, coverage=0.9)
QUANTILE_PRED_MULTI_M = f_multi.predict_quantiles(fh=fh_multi, alpha=[0.05, 0.5, 0.95])
INTERVAL_PRED_MULTI_M = f_multi.predict_interval(
    fh=fh_multi, coverage=[0.7, 0.8, 0.9, 0.99]
)

multi_data = [
    QUANTILE_PRED_MULTI_S,
    INTERVAL_PRED_MULTI_S,
    QUANTILE_PRED_MULTI_M,
    INTERVAL_PRED_MULTI_M,
]


@pytest.mark.parametrize(
    "y_true, y_pred",
    # list(zip([y_test_uni] * 4, uni_data))
    list(zip([y_test_uni] * 4, uni_data)) + list(zip([y_test_multi] * 4, multi_data)),
)
@pytest.mark.parametrize("metric", all_metrics)
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
@pytest.mark.parametrize("score_average", [True, False])
def test_output(metric, score_average, multioutput, y_true, y_pred):
    """Test output is correct class and shape."""
    loss = metric.create_test_instance()
    loss.set_params(score_average=score_average, multioutput=multioutput)

    eval_loss = loss(y_true, y_pred)
    index_loss = loss.evaluate_by_index(y_true, y_pred)

    no_vars = len(y_pred.columns.get_level_values(0).unique())
    no_scores = len(y_pred.columns.get_level_values(1).unique())

    if (
        0.5 in y_pred.columns.get_level_values(1)
        and loss.get_tag("scitype:y_pred") == "pred_interval"
        and y_pred.columns.nlevels == 2
    ):
        no_scores = no_scores - 1
        no_scores = no_scores / 2  # one interval loss per two quantiles given
        if no_scores == 0:  # if only 0.5 quant, no output to interval loss
            no_vars = 0

    if score_average and multioutput == "uniform_average":
        assert isinstance(eval_loss, float)
        assert isinstance(index_loss, pd.Series)

        assert len(index_loss) == y_pred.shape[0]

    if not score_average and multioutput == "uniform_average":
        assert isinstance(eval_loss, pd.Series)
        assert isinstance(index_loss, pd.DataFrame)

        # get two quantiles from each interval so if not score averaging
        # get twice number of unique coverages
        if (
            loss.get_tag("scitype:y_pred") == "pred_quantiles"
            and y_pred.columns.nlevels == 3
        ):
            assert len(eval_loss) == 2 * no_scores
        else:
            assert len(eval_loss) == no_scores

    if not score_average and multioutput == "raw_values":
        assert isinstance(eval_loss, pd.Series)
        assert isinstance(index_loss, pd.DataFrame)

        true_len = no_vars * no_scores

        if (
            loss.get_tag("scitype:y_pred") == "pred_quantiles"
            and y_pred.columns.nlevels == 3
        ):
            assert len(eval_loss) == 2 * true_len
        else:
            assert len(eval_loss) == true_len

    if score_average and multioutput == "raw_values":
        assert isinstance(eval_loss, pd.Series)
        assert isinstance(index_loss, pd.DataFrame)

        assert len(eval_loss) == no_vars

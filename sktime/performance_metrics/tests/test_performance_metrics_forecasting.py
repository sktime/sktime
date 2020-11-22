#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Tomasz Chodakowski"]

import pytest

from sktime.performance_metrics.forecasting import mape_loss
from sktime.performance_metrics.forecasting import smape_loss

from sktime.performance_metrics.tests._config import TEST_YS, TEST_YS_ZERO


@pytest.mark.parametrize("test_y", TEST_YS_ZERO)
def test_mape_loss_near_zero(test_y):
    # mape should be large when predicitons are close to zero
    y_test = test_y
    y_pred = y_test + 0.001
    assert mape_loss(y_test, y_pred) > 1e6


@pytest.mark.parametrize("test_y", TEST_YS)
def test_mape_loss(test_y):
    y_test = test_y
    y_pred = y_test
    assert mape_loss(y_test, y_pred) == pytest.approx(0.0)

    y_pred = y_test * 1.1
    assert mape_loss(y_test, y_pred) == pytest.approx(0.1)

    y_pred = y_test * 0.9
    assert mape_loss(y_test, y_pred) == pytest.approx(0.1)

    y_pred = y_test * 1.000001
    assert mape_loss(y_test, y_pred) == pytest.approx(0.000001)

    y_pred = y_test * 2
    assert mape_loss(y_test, y_pred) == pytest.approx(1.0)

    y_pred = y_test * 100
    assert mape_loss(y_test, y_pred) == pytest.approx(99.0)


@pytest.mark.parametrize("test_y", TEST_YS)
def test_smape_loss(test_y):
    y_test = test_y
    y_pred = y_test
    assert smape_loss(y_test, y_pred) == pytest.approx(0.0)
    assert smape_loss(y_pred, y_test) == pytest.approx(0.0)

    y_pred = y_test * 1.1
    assert smape_loss(y_test, y_pred) == pytest.approx(0.095238095238)
    assert smape_loss(y_pred, y_test) == pytest.approx(0.095238095238)

    y_pred = y_test * 1.000001
    assert smape_loss(y_test, y_pred) == pytest.approx(0.000001)
    assert smape_loss(y_pred, y_test) == pytest.approx(0.000001)

    y_pred = y_test * 2.0
    assert smape_loss(y_test, y_pred) == pytest.approx(0.6666666)
    assert smape_loss(y_pred, y_test) == pytest.approx(0.6666666)

    y_pred = y_test * 100
    assert smape_loss(y_test, y_pred) == smape_loss(y_pred, y_test)

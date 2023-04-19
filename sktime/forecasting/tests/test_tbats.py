# -*- coding: utf-8 -*-
"""Tests for TBATS."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly", "ngupta23"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.tbats import TBATS
from sktime.utils.validation._dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(TBATS, severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_tbats_long_fh():
    """Test TBATS with long fh, checks for failure condition in bug #4491."""
    LEN_HISTORY = 50
    train = pd.Series(data=np.random.randint(1, 100, LEN_HISTORY))

    # train model
    estimator = TBATS(
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=False,
        sp=10,
        use_arma_errors=False,
    )
    estimator.fit(train)

    # failure condition is fh being longer than training data
    long_fh = np.array(range(1, LEN_HISTORY + 2))

    fcst = estimator.predict_interval(coverage=0.8, fh=long_fh)
    assert len(fcst) == len(long_fh)

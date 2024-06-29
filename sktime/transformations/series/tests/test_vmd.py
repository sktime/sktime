"""Tests for VmdTransformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly", "DaneLyttinen"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.libs.vmdpy import VMD
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.vmd import VmdTransformer


def _generate_vmd_testdata(T=1000, f_1=2, f_2=24, f_3=288, noise=0.1):
    """Generate test data for VMD tests.

    Based on example of DaneLyttinen in #5128

    Parameters
    ----------
    T : int
        length of time series
    f_1 : int
    f_2 : int
    f_3 : int
        center frequencies of components
        f_i = frequency of component i
    noise : float
        noise level
    """
    # Time Domain 0 to T
    T = 1000
    t = np.arange(1, T + 1) / T

    # modes
    v_1 = np.cos(2 * np.pi * f_1 * t)
    v_2 = 1 / 4 * (np.cos(2 * np.pi * f_2 * t))
    v_3 = 1 / 16 * (np.cos(2 * np.pi * f_3 * t))

    f = v_1 + v_2 + v_3 + noise * np.random.randn(v_1.size)

    return pd.DataFrame(data={"y": f})


@pytest.mark.skipif(
    not run_test_for_class([VmdTransformer, VMD]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_vmd_in_pipeline():
    """Test vmd as part of a TransformedTargetForecaster pipeline."""
    y = _generate_vmd_testdata()

    pipe = TransformedTargetForecaster(
        steps=[
            ("vmd", VmdTransformer()),
            ("forecaster", TrendForecaster()),
        ]
    )

    pipe.fit(y, fh=[1, 2, 3])
    pipe.predict()

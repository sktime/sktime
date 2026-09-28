"""Tests for VmdTransformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly", "DaneLyttinen", "danferns"]

import numpy as np
import pandas as pd
import pytest

from sktime.libs.vmdpy import VMD
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.vmd import VmdTransformer


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

    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.forecasting.trend import TrendForecaster

    pipe = TransformedTargetForecaster(
        steps=[
            ("vmd", VmdTransformer()),
            ("forecaster", TrendForecaster()),
        ]
    )

    pipe.fit(y, fh=[1, 2, 3])
    pipe.predict()


@pytest.mark.parametrize("length", [1000, 1001])
def test_vmd_sequence_length(length):
    """Test vmd decomposition length matches input data length."""
    y = _generate_vmd_testdata(T=length)

    transformer = VmdTransformer()
    modes = transformer.fit_transform(y)
    assert len(modes) == length


@pytest.mark.skipif(
    not run_test_for_class([VmdTransformer, VMD]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("returned_decomp", ["u", "u_hat", "u_both"])
def test_vmd_preserves_time_index(returned_decomp):
    """Test vmd output keeps the time index of the input."""
    y = _generate_vmd_testdata(T=100)
    y.index = pd.period_range("2000-01", periods=100, freq="M")

    transformer = VmdTransformer(K=3, returned_decomp=returned_decomp)
    modes = transformer.fit_transform(y)
    assert modes.index.equals(y.index)


@pytest.mark.skipif(
    not run_test_for_class([VmdTransformer, VMD]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_vmd_pipeline_forecast_index():
    """Test decompose-forecast-recompose pipeline returns the correct fh index."""
    from sktime.forecasting.trend import TrendForecaster

    y = _generate_vmd_testdata(T=100)["y"]
    y.index = pd.period_range("2000-01", periods=100, freq="M")

    pipe = VmdTransformer(K=3) * TrendForecaster()
    pipe.fit(y, fh=[1, 2, 3])
    y_pred = pipe.predict()

    expected_index = pd.period_range("2008-05", periods=3, freq="M")
    assert y_pred.index.equals(expected_index)


@pytest.mark.skipif(
    not run_test_for_class([VmdTransformer, VMD]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_vmd_multivariate_decomposes_each_variable():
    """Test that multivariate input is decomposed variable by variable."""
    y = _generate_vmd_testdata(T=100)
    X = pd.concat(
        [y.rename(columns={"y": "a"}), (2 * y).rename(columns={"y": "b"})], axis=1
    )

    transformer = VmdTransformer(K=3)
    modes = transformer.fit_transform(X)

    assert modes.shape == (100, 6)
    assert modes.index.equals(X.index)
    modes_a = VmdTransformer(K=3).fit_transform(X[["a"]])
    np.testing.assert_allclose(modes.iloc[:, :3].to_numpy(), modes_a.to_numpy())

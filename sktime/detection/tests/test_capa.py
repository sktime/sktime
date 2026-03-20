"""Tests for CAPA change point detection method."""

__author__ = ["CloseChoice"]
__all__ = []
import numpy as np
import pandas as pd
import pytest
from skchange.anomaly_scores import Saving
from skchange.costs import L1Cost
from skchange.datasets import generate_anomalous_data

from sktime.detection.skchange_aseg import CAPA
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(CAPA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_capa_params():
    """Test that the interface to skchange works correctly for all parameters."""
    l1_saving = Saving(L1Cost(1))
    detector = CAPA(
        segment_saving=l1_saving,
        segment_penalty=2.0,
        point_saving=l1_saving,
        point_penalty=1.0,
        min_segment_length=2,
        max_segment_length=1000,
        ignore_point_anomalies=False,
        find_affected_components=False,
    )
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    y = pd.Series(np.zeros(len(x)))
    fit_detector = detector.fit(x, y)
    result = fit_detector.predict(x)
    assert isinstance(detector, CAPA)
    assert isinstance(fit_detector, CAPA)
    assert isinstance(result, pd.DataFrame)

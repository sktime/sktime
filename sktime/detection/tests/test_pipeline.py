"""Tests for detector pipelines."""

import pytest

from sktime.detection.compose import DetectorPipeline
from sktime.detection.lof import SubLOF
from sktime.transformations.series.detrend import Detrender


def test_pipeline_raises_type_error_for_transformer_last():
    """Test that placing a transformer after a detector raises TypeError."""
    with pytest.raises(TypeError, match="last estimator must be a time series detector"):
        DetectorPipeline(
            steps=[
                ("det", SubLOF(n_neighbors=2, window_size=2)),
                ("trafo", Detrender()),
            ]
        )

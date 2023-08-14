"""Simple ClaSP test."""

__author__ = ["patrickzib"]
__all__ = []

import numpy as np
import pytest

from sktime.annotation.clasp import ClaSPSegmentation
from sktime.datasets import load_gun_point_segmentation
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ClaSPSegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_clasp_sparse():
    """Test ClaSP sparse segmentation.

    Check if the predicted change points match.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, n_cps=1)
    clasp.fit(ts)
    found_cps = clasp.predict(ts)
    scores = clasp.predict_scores(ts)

    assert len(found_cps) == 1 and found_cps[0] == 893
    assert len(scores) == 1 and scores[0] > 0.74


@pytest.mark.skipif(
    not run_test_for_class(ClaSPSegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_clasp_dense():
    """Tests ClaSP dense segmentation.

    Check if the predicted segmentation matches.
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, n_cps=1, fmt="dense")
    clasp.fit(ts)
    segmentation = clasp.predict(ts)
    scores = clasp.predict_scores(ts)

    assert len(segmentation) == 2 and segmentation[0].right == 893
    assert np.argmax(scores) == 893

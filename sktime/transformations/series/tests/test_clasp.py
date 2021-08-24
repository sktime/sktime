# -*- coding: utf-8 -*-
"""Simple ClaSP test."""

__author__ = ["Arik Ermshaus, Patrick Schäfer"]
__all__ = []

from sktime.annotation.clasp import ClaSPSegmentation
from sktime.datasets import load_gun_point_segmentation


def test_clasp():
    """
    Test ClaSP.

    :return:
    """
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, n_cps=1).fit(ts)
    found_cps, _, _ = clasp.predict(ts)
    assert len(found_cps) == 1 and found_cps[0] == 893

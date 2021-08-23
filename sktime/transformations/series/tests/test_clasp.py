#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Arik Ermshaus, Patrick Schäfer"]
__all__ = []

from sktime.annotation.clasp import ClaSPSegmentation
from sktime.datasets import load_gun_point_segmentation


def test_clasp():
    # load the test dataset
    ts, period_size, cps = load_gun_point_segmentation()

    # compute a ClaSP segmentation
    clasp = ClaSPSegmentation(period_size, len(cps)).fit(ts)
    profile, found_cps, scores = clasp.predict(ts)
    assert len(found_cps) == 1 and found_cps[0] == 893

#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Arik Ermshaus, Patrick Schäfer"]
__all__ = []

from sktime.transformations.series.clasp import segmentation
from sktime.datasets import load_gun_point_segmentation


def test_clasp():
    ts, window_size, cps = load_gun_point_segmentation()
    _, found_cps, _ = segmentation(ts, window_size, len(cps))
    assert len(found_cps) == 1 and found_cps[0] == 893

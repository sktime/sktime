# -*- coding: utf-8 -*-
"""Test the distance calculations are correct.

Compare the distance calculations on the 1D and 2D (d,m) format input against the
results generated with tsml, LINK TO CODE.
"""

distance_functions = [
    "euclidean",
    "dtw",
    "wdtw",
    "msm",
    "erp",
    "lcss",
    "ddtw",
    "wddtw",
]

unit_test_distances = {
    "euclidean": 0.0,
    "dtw": 0.0,
    "wdtw": 0.0,
    "msm": 0.0,
    "erp": 0.0,
    "lcss": 0.0,
    "ddtw": 0.0,
    "wddtw": 0.0,
}
basic_motions_distances = {
    "euclidean": 0.0,
    "dtw": 0.0,
    "wdtw": 0.0,
    "msm": 0.0,
    "erp": 0.0,
    "lcss": 0.0,
    "ddtw": 0.0,
    "wddtw": 0.0,
}

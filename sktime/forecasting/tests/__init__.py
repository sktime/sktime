#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np

# default parameter testing grid
TEST_WINDOW_LENGTHS = [1, 5]
TEST_STEP_LENGTHS = [1, 5]
TEST_OOS_FHS = [1, np.array([2, 5])]  # out-of-sample
TEST_INS_FHS = [
    -3,  # single in-sample
    np.array([-2, -5]),  # multiple in-sample
    0,  # last training point
    np.array([-3, 2])  # mixed in-sample and out-of-sample
]
TEST_FHS = TEST_OOS_FHS + TEST_INS_FHS
TEST_SPS = [3, 7, 12]
TEST_ALPHAS = [0.05, 0.1]

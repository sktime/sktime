# -*- coding: utf-8 -*-
from scipy import stats
import numpy as np


def build_e_distance_params(X, y=None):
    return {}


def build_dtw_distance_params(X, y=None):
    return {
        "w": stats.uniform(0, 1),
    }


def build_wdtw_distance_params(X, y=None):
    return {
        "g": stats.uniform(0, 1),
    }


def build_twe_distance_params(X, y=None):
    return {
        "penalty": [
            0,
            0.011111111,
            0.022222222,
            0.033333333,
            0.044444444,
            0.055555556,
            0.066666667,
            0.077777778,
            0.088888889,
            0.1,
        ],
        "stiffness": [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    }


def build_lcss_distance_params(X, y=None):
    stdp = np.std(X)
    instance_length = X[0]
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    return {
        "epsilon": stats.uniform(0.2 * stdp, stdp - 0.2 * stdp),
        # scipy stats randint is exclusive on the max value, hence + 1
        "delta": stats.randint(low=0, high=max_raw_warping_window + 1),
    }


def build_erp_distance_params(X, y=None):
    stdp = np.std(X)
    instance_length = X[0]
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    return {
        "g": stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
        # scipy stats randint is exclusive on the max value, hence + 1
        "band_size": stats.randint(low=0, high=max_raw_warping_window + 1),
    }


def build_msm_distance_params(X, y=None):
    return {
        "c": [
            0.01,
            0.01375,
            0.0175,
            0.02125,
            0.025,
            0.02875,
            0.0325,
            0.03625,
            0.04,
            0.04375,
            0.0475,
            0.05125,
            0.055,
            0.05875,
            0.0625,
            0.06625,
            0.07,
            0.07375,
            0.0775,
            0.08125,
            0.085,
            0.08875,
            0.0925,
            0.09625,
            0.1,
            0.136,
            0.172,
            0.208,
            0.244,
            0.28,
            0.316,
            0.352,
            0.388,
            0.424,
            0.46,
            0.496,
            0.532,
            0.568,
            0.604,
            0.64,
            0.676,
            0.712,
            0.748,
            0.784,
            0.82,
            0.856,
            0.892,
            0.928,
            0.964,
            1,
            1.36,
            1.72,
            2.08,
            2.44,
            2.8,
            3.16,
            3.52,
            3.88,
            4.24,
            4.6,
            4.96,
            5.32,
            5.68,
            6.04,
            6.4,
            6.76,
            7.12,
            7.48,
            7.84,
            8.2,
            8.56,
            8.92,
            9.28,
            9.64,
            10,
            13.6,
            17.2,
            20.8,
            24.4,
            28,
            31.6,
            35.2,
            38.8,
            42.4,
            46,
            49.6,
            53.2,
            56.8,
            60.4,
            64,
            67.6,
            71.2,
            74.8,
            78.4,
            82,
            85.6,
            89.2,
            92.8,
            96.4,
            100,
        ],
    }

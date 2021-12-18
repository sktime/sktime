# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for panel transformers of time series augmentation."""

import numpy as np
import pandas as pd
import pytest
from sklearn import preprocessing

from sktime.datasets import load_basic_motions
from sktime.transformations.panel import augmenter as aug

expected_shapes_seq_aug_pipeline = [(20, 2)]
expected_checksums_seq_aug_pipeline = [0.0]


def test_seq_aug_pipeline():
    """Test of the sequential augmentation pipeline."""
    np.random.seed(42)
    shapes = []
    checksums = []
    pipe = aug.SeqAugPipeline(
        [
            ("invert", aug.InvertAugmenter(p=1.0)),
            ("reverse", aug.ReverseAugmenter(p=1.0)),
            (
                "white_noise",
                aug.WhiteNoiseAugmenter(
                    p=0.0,
                    param=1.0,
                    use_relative_fit=True,
                    relative_fit_stat_fun=np.std,
                    relative_fit_type="instance-wise",
                ),
            ),
        ]
    )
    # create naive panel with 20 instances and two variables and binary target
    n_vars = 2
    n_instances = 20
    X = pd.DataFrame([[pd.Series(np.linspace(-1, 1, 5))] * n_vars] * n_instances)
    y = pd.Series(np.random.rand(n_instances) > 0.5)
    pipe.fit(X, y)
    Xt = pipe.transform(X)
    checksum = _calc_checksum(Xt)
    checksums.append(checksum)
    shapes.append(X.shape)
    assert shapes == expected_shapes_seq_aug_pipeline
    assert checksums == expected_checksums_seq_aug_pipeline


def _load_test_data():
    # get some multivariate panel data
    le = preprocessing.LabelEncoder()
    X_tr, y_tr = load_basic_motions(split="train", return_X_y=True)
    X_te, y_te = load_basic_motions(split="test", return_X_y=True)
    y_tr = pd.Series(le.fit(y_tr).transform(y_tr))
    y_te = pd.Series(le.fit(y_te).transform(y_te))
    return (X_tr, X_te, y_tr, y_te)


def _train_test(data, augmentator):
    X_tr, X_te, y_tr, y_te = data
    # fit augmenter object (if necessary)
    augmentator.fit(X_tr, y_tr)
    # transform new data with (fitted) augmenter
    Xt = augmentator.transform(X_te, y_te)
    # check if result seems (trivially) invalid
    return Xt


def _calc_checksum(X):
    if isinstance(X, pd.DataFrame):
        checksum = sum([sum([sum(x) for x in X[c]]) for c in X.columns])
    else:
        checksum = sum(X)
    return checksum


# Test Data
expected_checksums_data = [646.1844410000003, -278.36259900000056, 60, 60]


def test_loaded_data():
    """Test of the loaded motion data."""
    data = _load_test_data()
    checksums = []
    for d in data:
        checksums.append(_calc_checksum(d))
    assert checksums == expected_checksums_data


# Test WhiteNoiseAugmenter
expected_shapes_white_noise = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_white_noise = [
    -353.4417418033514,
    -450.1602614030863,
    -373.39111902606186,
    8730.662994619102,
    -1181.9703691109278,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": 4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": 11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": 2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_white_noise(parameter):
    """Test of the White Noise Augmenter."""
    np.random.seed(42)
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.WhiteNoiseAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_white_noise
    assert checksums == expected_checksums_white_noise


# Test InvertAugmenter
expected_shapes_invert = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_invert = [
    -321.1576750000005,
    3027.2553650000013,
    2039.247608999999,
    -5084.939050999999,
    5567.400410999999,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_invert(parameter):
    """Test of the Invert Augmenter."""
    np.random.seed(42)
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.InvertAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_invert
    assert checksums == expected_checksums_invert


# Test ReverseAugmenter
expected_shapes_reverse = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_reverse = [
    -278.36259900000056,
    -278.36259900000056,
    -278.36259900000067,
    -278.36259900000243,
    -278.36259900000056,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_reverse(parameter):
    """Test of the Reverse Augmenter."""
    np.random.seed(42)
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ReverseAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_reverse
    assert checksums == expected_checksums_reverse


# Test ScaleAugmenter
expected_shapes_scale = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_scale = [
    -388.9508193002375,
    -4837.6827805519515,
    -1166.3225152032387,
    -360655.26179077814,
    47750.129927408474,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_scale(parameter):
    """Test of the Scale Augmenter."""
    np.random.seed(42)
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ScaleAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_scale
    assert checksums == expected_checksums_scale


# Test OffsetAugmenter
expected_shapes_offset = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_offset = [
    -18201.91688318999,
    7238.693795544632,
    7669.8614458559205,
    -335331.9467949644,
    -32685.349670806663,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_offset(parameter):
    """Test of the Offset Augmenter."""
    np.random.seed(42)
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.OffsetAugmenter(**para)
        Xt = _train_test(data, augmentator)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_offset
    assert checksums == expected_checksums_offset


# Test DriftAugmenter
expected_shapes_drift = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_drift = [
    3050.00379820752,
    -13181.787867767554,
    1489.572937646976,
    50027.573433620084,
    -1089.8973404995957,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": 4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": 11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": 2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_drift(parameter):
    """Test of the Drift Augmenter."""
    np.random.seed(42)
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.DriftAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    print(checksums)
    assert shapes == expected_shapes_drift
    assert checksums == expected_checksums_drift


# def test_mtype_compatibility():
#    pass


# def test_variable_inconsistency():
#    """ValueError if the number of variables differ from fit to transform."""
#    pass

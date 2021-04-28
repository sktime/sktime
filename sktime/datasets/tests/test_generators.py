#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Stuart Miller"]

import pytest
import numpy as np
import pandas as pd

from sktime.datasets.generators import NoiseGenerator
from sktime.datasets.generators import ArmaGenerator
from sktime.datasets.generators import LinearGenerator

RANDOM_SEED = 42
RANDOM_GENERATOR = np.random.RandomState(42)


def test_noise_generator_init():
    # construct instance without errors
    # without seed
    generator = NoiseGenerator()
    assert generator
    # int type seed
    generator = NoiseGenerator(RANDOM_SEED)
    assert generator
    # random state type seed
    generator = NoiseGenerator(RANDOM_GENERATOR)
    assert generator


def test_noise_generator_basic():
    generator = NoiseGenerator(RANDOM_SEED)

    # 1D time series
    sample = generator.sample(100, 1)
    # test ret type
    assert isinstance(sample, pd.Series)
    # verify shape of return
    assert len(sample.shape) == 1
    assert sample.shape[0] == 100

    # 2D time series
    sample = generator.sample(100, 2)
    # test ret type
    assert isinstance(sample, pd.DataFrame)
    # verify shape of return
    assert len(sample.shape) == 2
    # expect long dataframe
    assert sample.shape[0] == 2
    assert sample.shape[1] == 100


def test_noise_generator_invalid_input():
    # verify value errors from bad inputs
    generator = NoiseGenerator(RANDOM_SEED)
    # invalid in both cases
    with pytest.raises(ValueError):
        generator.sample(0, 0)
    # invalid n_instance
    with pytest.raises(ValueError):
        generator.sample(100, 0)
    # invalid n_sample
    with pytest.raises(ValueError):
        generator.sample(0, 1)


def test_arma_generator_init():
    # construct instance without errors
    # without seed
    generator = ArmaGenerator(ar=np.array([0.9]), ma=np.array([0.7, 0.3]))
    assert generator
    # int type seed
    generator = ArmaGenerator(
        ar=np.array([0.9]), ma=np.array([0.7, 0.3]), random_state=RANDOM_SEED
    )
    assert generator
    # random state type seed
    generator = ArmaGenerator(
        ar=np.array([0.9]), ma=np.array([0.7, 0.3]), random_state=RANDOM_GENERATOR
    )
    assert generator


def test_arma_generator_params_transform():
    # check params with no input
    generator = ArmaGenerator()
    assert 1.0 == pytest.approx(generator.get_params().get("ar"))
    assert 1.0 == pytest.approx(generator.get_params().get("ma"))

    # check params with input
    generator = ArmaGenerator(ar=[0.9], ma=[0.5])
    assert 1.0 == pytest.approx(generator.get_params().get("ar")[0])
    assert -0.9 == pytest.approx(generator.get_params().get("ar")[1])
    assert 1.0 == pytest.approx(generator.get_params().get("ma")[0])
    assert 0.5 == pytest.approx(generator.get_params().get("ma")[1])


def test_arma_generator_basic():
    generator = ArmaGenerator(
        ar=np.array([0.9]), ma=np.array([0.7, 0.3]), random_state=RANDOM_SEED
    )

    # 1D time series
    sample = generator.sample(100, 1)
    # test ret type
    assert isinstance(sample, pd.Series)
    # verify shape of return
    assert len(sample.shape) == 1
    assert sample.shape[0] == 100

    # 2D time series
    sample = generator.sample(100, 2)
    # test ret type
    assert isinstance(sample, pd.DataFrame)
    # verify shape of return
    assert len(sample.shape) == 2
    # expect long dataframe
    assert sample.shape[0] == 2
    assert sample.shape[1] == 100


def test_arma_generator_invalid_input():
    # verify value errors from bad inputs
    generator = ArmaGenerator(RANDOM_SEED)
    # invalid in both cases
    with pytest.raises(ValueError):
        generator.sample(0, 0)
    # invalid n_instance
    with pytest.raises(ValueError):
        generator.sample(100, 0)
    # invalid n_sample
    with pytest.raises(ValueError):
        generator.sample(0, 1)


def test_linear_generator_basic():
    generator = LinearGenerator(slope=4, intercept=2)

    # 1D time series; no noise
    sample = generator.sample(100, 1)
    # test ret type
    assert isinstance(sample, pd.Series)
    # verify shape of return
    assert len(sample.shape) == 1
    assert sample.shape[0] == 100
    # expect 4 * 0 + 2 = 2
    assert 2 == pytest.approx(sample[0])
    # expect 4 * 99 + 2 = 398
    assert 398 == pytest.approx(sample[99])

    # 2D time series; no noise
    sample = generator.sample(100, 2)
    # test ret type
    assert isinstance(sample, pd.DataFrame)
    # verify shape of return
    assert len(sample.shape) == 2
    # expect long dataframe
    assert sample.shape[0] == 2
    assert sample.shape[1] == 100
    # expect 4 * 0 + 2 = 2
    assert 2 == pytest.approx(sample.iloc[0, 0])
    assert 2 == pytest.approx(sample.iloc[1, 0])
    # expect 4 * 99 + 2 = 398
    assert 398 == pytest.approx(sample.iloc[0, 99])
    assert 398 == pytest.approx(sample.iloc[1, 99])


def test_linear_generator_with_noise():
    noise_generator = NoiseGenerator(RANDOM_SEED)
    generator = LinearGenerator(1, 0, noise_generator=noise_generator)

    # verify compatible shapes
    sample_1d = generator.sample(10, 1)
    # test ret type
    assert isinstance(sample_1d, pd.Series)
    # verify shape of return
    assert len(sample_1d.shape) == 1
    assert sample_1d.shape[0] == 10

    sample_2d = generator.sample(10, 3)
    # test ret type
    assert isinstance(sample_2d, pd.DataFrame)
    # expect long dataframe
    assert sample_2d.shape[0] == 3
    assert sample_2d.shape[1] == 10

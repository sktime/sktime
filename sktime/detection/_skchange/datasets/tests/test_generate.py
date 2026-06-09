import numpy as np
import pandas as pd
import pytest
from scipy.stats import (
    bernoulli,
    binom,
    expon,
    hypergeom,
    multivariate_normal,
    multivariate_t,
    norm,
    poisson,
    rv_continuous,
    rv_discrete,
    uniform,
)

from sktime.detection._skchange.datasets import GENERATORS, generate_piecewise_data


@pytest.mark.parametrize("generate", GENERATORS)
def test_generate_piecewise_data_expected_output_lengths(generate):
    def get_df_and_params(output: tuple):
        df = None
        params = None
        for item in output:
            if isinstance(item, pd.DataFrame):
                df = item
            if isinstance(item, dict):
                params = item
        return df, params

    lengths = [10, 20]
    n_segments = 5
    n_samples = 100

    output = generate(
        lengths=lengths,
        n_segments=n_segments,
        n_samples=n_samples,
        return_params=True,
    )
    df, params = get_df_and_params(output)
    assert df.shape[0] == params["n_samples"]
    assert df.shape[0] == np.sum(lengths)

    output = generate(
        n_segments=n_segments,
        n_samples=n_samples,
        return_params=True,
    )
    df, params = get_df_and_params(output)
    assert df.shape[0] == params["n_samples"]
    assert df.shape[0] == n_samples

    output = generate(
        lengths=lengths[0],
        n_segments=n_segments,
        n_samples=n_samples,
        return_params=True,
    )
    df, params = get_df_and_params(output)
    assert df.shape[0] == params["n_samples"]
    assert df.shape[0] == lengths[0] * n_segments


SCIPY_DISTRIBUTIONS = [
    norm(),
    uniform(),
    poisson(5),
    binom(n=10, p=0.5),
    expon(),
    multivariate_normal(),
    multivariate_t(),
    bernoulli(p=0.5),
    hypergeom(M=20, n=7, N=12),
]


@pytest.mark.parametrize("distribution", SCIPY_DISTRIBUTIONS + [None])
def test_generate_piecewise_data(distribution: rv_continuous | rv_discrete):
    length = 10
    df = generate_piecewise_data(
        distributions=distribution,
        lengths=[length],
        seed=42,
    )
    assert df.shape[0] == length
    assert df.columns == pd.RangeIndex(start=0, stop=df.shape[1])


def test_generate_piecewise_data_invalid_distributions():
    with pytest.raises(ValueError):
        generate_piecewise_data(
            distributions=[],  # Empty list of distributions
            lengths=[50],
            seed=42,
        )
    with pytest.raises(ValueError):
        generate_piecewise_data(
            distributions=["haha", norm],  # Does not have an rvs method.
            n_segments=2,
            seed=0,
        )
    with pytest.raises(ValueError):
        generate_piecewise_data(
            distributions=[
                norm,
                multivariate_normal(mean=[0, 1]),
            ],  # Mismatching output sizes.
            n_segments=2,
            seed=0,
        )
    with pytest.raises(ValueError):
        generate_piecewise_data(
            distributions=[
                norm,
                binom(n=10, p=0.5),
            ],  # Mismatching dtypes.
            n_segments=2,
            seed=0,
        )


@pytest.mark.parametrize(
    "lengths",
    [
        [-1],
        np.array([[10, 20]]),  # 2d array
        "hehe",
    ],
)
def test_generate_piecewise_data_invalid_lengths(lengths: list):
    with pytest.raises((ValueError, TypeError)):
        generate_piecewise_data(
            distributions=[norm, norm(2), norm(5)],
            lengths=lengths,
        )
    with pytest.raises(ValueError):
        generate_piecewise_data(
            n_samples=2,
            n_segments=3,
        )


def test_generate_piecewise_data_seed():
    length = 100
    df1 = generate_piecewise_data(
        distributions=[norm],
        lengths=length,
        seed=42,
    )
    df2 = generate_piecewise_data(
        distributions=[norm],
        lengths=length,
        seed=42,
    )
    pd.testing.assert_frame_equal(df1, df2)

    df3 = generate_piecewise_data(
        distributions=[norm],
        lengths=[length],
    )

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)

    with pytest.raises(TypeError):
        generate_piecewise_data(
            distributions=[norm],
            lengths=length,
            seed=np.random.RandomState(),
        )

"""Tests for FABBA symbolic representation transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ravindra-RKB"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.fabba import FABBA


def _make_synthetic_series(length=120, kind="sine", seed=42):
    """Generate deterministic synthetic time series for testing.

    Parameters
    ----------
    length : int, default=120
        Number of time steps.
    kind : {"sine", "trend_noise", "piecewise_constant"}, default="sine"
        Waveform pattern of the series.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Synthetic univariate series.
    """
    rng = np.random.default_rng(seed)
    time_idx = np.linspace(0, 4 * np.pi, length)

    if kind == "sine":
        values = np.sin(time_idx) + 0.5 * np.cos(2 * time_idx)
    elif kind == "trend_noise":
        values = 0.05 * np.arange(length) + 0.1 * rng.standard_normal(length)
    elif kind == "piecewise_constant":
        values = np.repeat([1.0, 3.0, 2.0, 5.0, 4.0], length // 5)
        if len(values) < length:
            values = np.pad(values, (0, length - len(values)), mode="edge")
    else:
        values = rng.standard_normal(length)

    return pd.Series(values, name="val")


def _make_panel_series(n_instances=3, length=50, seed=42):
    """Generate a panel of univariate series as a list of DataFrames.

    Parameters
    ----------
    n_instances : int, default=3
        Number of series in the panel.
    length : int, default=50
        Length of each individual series.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    list of pd.DataFrame
        List of single-column DataFrames representing panel time series.
    """
    rng = np.random.default_rng(seed)
    panel = []
    for i in range(n_instances):
        t = np.linspace(0, 2 * np.pi, length)
        noise = 0.05 * rng.standard_normal(length)
        series = np.sin(t + i * 0.5) + noise
        panel.append(pd.DataFrame({"val": series}))
    return panel


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "method, k, return_as_strings",
    [
        ("agg", 2, True),
        ("agg", 2, False),
        ("kmeans", 3, True),
        ("kmeans", 4, False),
        ("mini-kmeans", 3, True),
        ("mini-kmeans", 4, False),
    ],
)
def test_fabba_clustering_methods(method, k, return_as_strings):
    """Verify fit and transform lifecycle across supported clustering methods."""
    y = _make_synthetic_series(length=80, kind="sine")
    transformer = FABBA(
        method=method,
        k=k,
        return_as_strings=return_as_strings,
        return_start_values=True,
        random_state=42,
        n_jobs=1,
    )

    y_transformed = transformer.fit_transform(y)

    assert hasattr(transformer, "cluster_centers_")
    assert hasattr(transformer, "num_grps_")
    assert transformer.num_grps_ >= 1
    assert transformer.cluster_centers_.shape[1] == 2

    # Transformed output must be non-empty and preserve series container type
    assert isinstance(y_transformed, (pd.Series, pd.DataFrame))
    assert len(y_transformed) >= 1

    first_val = y_transformed.iloc[0]
    if isinstance(y_transformed, pd.DataFrame):
        first_val = y_transformed.iloc[0, 0]

    # When return_start_values=True, first element corresponds to start scalar
    assert isinstance(first_val, (float, np.floating, int, np.integer))

    if return_as_strings:
        assert transformer.alphabets_ is not None
        assert len(transformer.alphabets_) == transformer.num_grps_
    else:
        assert transformer.alphabets_ is None


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("sorting", ["norm", "pca", None])
def test_fabba_sorting_methods(sorting):
    """Verify aggregation sorting algorithms in aggregate clustering mode."""
    y = _make_synthetic_series(length=100, kind="trend_noise")
    transformer = FABBA(
        method="agg",
        sorting=sorting,
        tolerance=0.1,
        alpha=0.2,
        return_as_strings=False,
        return_start_values=True,
        n_jobs=1,
    )

    y_transformed = transformer.fit_transform(y)

    assert hasattr(transformer, "cluster_centers_")
    assert transformer.num_grps_ >= 1
    assert len(y_transformed) > 0


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "alphabet_set",
    [
        0,
        1,
        ["A0", "B1", "C2", "D3", "E4", "F5", "G6", "H7", "I8", "J9", "K10", "L11"],
    ],
)
def test_fabba_alphabet_configurations(alphabet_set):
    """Verify custom and built-in alphabet set mappings."""
    y = _make_synthetic_series(length=80, kind="sine")
    transformer = FABBA(
        method="agg",
        alphabet_set=alphabet_set,
        return_as_strings=True,
        return_start_values=True,
        tolerance=0.1,
        alpha=0.3,
        n_jobs=1,
    )

    y_transformed = transformer.fit_transform(y)
    assert transformer.alphabets_ is not None

    if isinstance(alphabet_set, list):
        for sym in transformer.alphabets_:
            assert sym in alphabet_set

    # Tokens in output (excluding the first start value) must belong to alphabet
    tokens = y_transformed.iloc[1:]
    if isinstance(y_transformed, pd.DataFrame):
        tokens = y_transformed.iloc[1:, 0]

    for token in tokens:
        assert token in transformer.alphabets_


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["agg", "kmeans"])
def test_fabba_inverse_transform_reconstruction(method):
    """Verify round-trip reconstruction accuracy and length preservation."""
    y = _make_synthetic_series(length=90, kind="sine")

    transformer = FABBA(
        method=method,
        k=4,
        tolerance=0.01,
        alpha=0.05,
        return_start_values=True,
        random_state=42,
        n_jobs=1,
    )

    y_transformed = transformer.fit_transform(y)
    y_reconstructed = transformer.inverse_transform(y_transformed)

    # Reconstructed series length matches within polygonal piece quantization tolerance
    assert abs(len(y_reconstructed) - len(y)) <= 4
    # Check that reconstructed curve correlates strongly with original
    orig_vals = y.to_numpy()[: len(y_reconstructed)]
    rec_vals = y_reconstructed.to_numpy().flatten()[: len(orig_vals)]

    corr = np.corrcoef(orig_vals, rec_vals)[0, 1]
    min_corr = 0.85 if method == "agg" else 0.60
    assert corr > min_corr


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_tolerance_fidelity_monotonicity():
    """Verify that smaller polygonal tolerance yields lower reconstruction RMSE."""
    y = _make_synthetic_series(length=100, kind="sine", seed=10)

    # Tight tolerance
    t_tight = FABBA(
        method="agg",
        tolerance=0.005,
        alpha=0.05,
        return_start_values=True,
        n_jobs=1,
    )
    rec_tight = t_tight.fit_transform(y)
    y_rec_tight = t_tight.inverse_transform(rec_tight).to_numpy().flatten()

    # Loose tolerance
    t_loose = FABBA(
        method="agg",
        tolerance=0.2,
        alpha=0.05,
        return_start_values=True,
        n_jobs=1,
    )
    rec_loose = t_loose.fit_transform(y)
    y_rec_loose = t_loose.inverse_transform(rec_loose).to_numpy().flatten()

    orig = y.to_numpy()
    rmse_tight = np.sqrt(np.mean((orig - y_rec_tight) ** 2))
    rmse_loose = np.sqrt(np.mean((orig - y_rec_loose) ** 2))

    assert rmse_tight < rmse_loose


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_auto_digitize():
    """Verify automatic alpha determination in aggregation clustering."""
    y = _make_synthetic_series(length=80, kind="sine")
    transformer = FABBA(
        method="agg",
        auto_digitize=True,
        return_start_values=True,
        n_jobs=1,
    )

    transformer.fit(y)
    assert hasattr(transformer, "_alpha")
    assert transformer._alpha > 0.0

    y_transformed = transformer.transform(y)
    assert len(y_transformed) > 0


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_panel_multiseries_input():
    """Verify transformation and inverse transformation on panel inputs."""
    panel_X = _make_panel_series(n_instances=3, length=40)

    transformer = FABBA(
        method="kmeans",
        k=3,
        return_start_values=True,
        random_state=42,
        n_jobs=1,
    )

    transformed_panel = transformer.fit_transform(panel_X)
    assert isinstance(transformed_panel, (pd.DataFrame, list))
    assert len(transformed_panel) > 0

    reconstructed_panel = transformer.inverse_transform(transformed_panel)
    assert isinstance(reconstructed_panel, (pd.DataFrame, list))
    assert len(reconstructed_panel) > 0


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_constant_series_edge_case():
    """Verify stability on flat constant time series."""
    y_const = pd.Series(np.full(60, 5.0), name="const")

    transformer = FABBA(
        method="agg",
        tolerance=0.05,
        alpha=0.1,
        return_start_values=True,
        n_jobs=1,
    )

    y_transformed = transformer.fit_transform(y_const)
    y_rec = transformer.inverse_transform(y_transformed)

    assert len(y_rec) == len(y_const)
    np.testing.assert_allclose(y_rec.to_numpy().flatten(), 5.0, atol=1e-5)


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_zero_scaling_length_centers():
    """Verify clustering behavior when length scaling factor scl is zero."""
    y = _make_synthetic_series(length=70, kind="sine")
    transformer = FABBA(
        method="kmeans",
        k=3,
        scl=0.0,
        return_start_values=True,
        random_state=42,
        n_jobs=1,
    )

    y_transformed = transformer.fit_transform(y)
    y_reconstructed = transformer.inverse_transform(y_transformed)

    assert len(y_reconstructed) == len(y)
    assert transformer.cluster_centers_.shape[0] == 3


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_missing_start_values_error():
    """Verify ValueError when attempting inverse transform without start values."""
    y = _make_synthetic_series(length=50, kind="sine")
    transformer = FABBA(
        method="agg",
        return_start_values=False,
        n_jobs=1,
    )

    y_transformed = transformer.fit_transform(y)

    with pytest.raises(
        ValueError, match="Start values are needed for inverse transform"
    ):
        transformer.inverse_transform(y_transformed)


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_undersized_alphabet_error():
    """Verify ValueError when custom alphabet list has fewer tokens than clusters."""
    y = _make_synthetic_series(length=80, kind="sine")
    # Provide only 1 symbol for an aggregation that will generate multiple clusters
    transformer = FABBA(
        method="kmeans",
        k=5,
        alphabet_set=["only_one"],
        return_as_strings=True,
        n_jobs=1,
    )

    with pytest.raises(
        ValueError, match="Alphabet set provided is smaller than"
    ):
        transformer.fit(y)


@pytest.mark.skipif(
    not run_test_for_class(FABBA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fabba_invalid_symbols_inverse_error():
    """Verify ValueError when unrecognized symbol or out of range label is passed."""
    y = _make_synthetic_series(length=50, kind="sine")

    # Strings mode with corrupt token
    t_str = FABBA(
        method="kmeans",
        k=3,
        return_as_strings=True,
        return_start_values=True,
        random_state=42,
        n_jobs=1,
    )
    t_str.fit(y)
    with pytest.raises(ValueError, match="Symbol not in alphabet set"):
        t_str._custom_inverse_symbolize(["UNKNOWN_TOKEN"])

    # Labels mode with out of range label
    t_lbl = FABBA(
        method="kmeans",
        k=3,
        return_as_strings=False,
        return_start_values=True,
        random_state=42,
        n_jobs=1,
    )
    t_lbl.fit(y)
    with pytest.raises(ValueError, match="Label not in range of clusters"):
        t_lbl._custom_inverse_symbolize([9999])

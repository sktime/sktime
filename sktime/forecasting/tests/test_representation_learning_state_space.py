"""Unit tests for representation_learning state-space helpers.

These tests are intentionally small and synthetic: they only check basic
shape and finiteness invariants for the factor-dynamics backends used by
DeepDynamicFactor and potential future representation-learning forecasters.
"""

import numpy as np

from sktime.forecasting.representation_learning._state_space import (
    _generic_var1_dynamics,
    _linear_decoder_ddfm_dynamics,
    _StateSpaceParams,
)


def _make_random_factors(n_timepoints: int = 20, n_factors: int = 3) -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.normal(size=(n_timepoints, n_factors))


def test_generic_var1_dynamics_shapes_and_finiteness():
    """_generic_var1_dynamics returns a well-formed _StateSpaceParams."""
    factors = _make_random_factors(n_timepoints=30, n_factors=4)
    # y_scaled: simple linear combination with noise, same n_timepoints
    rng = np.random.RandomState(1)
    W = rng.normal(size=(4, 2))
    Y = factors @ W + 0.1 * rng.normal(size=(30, 2))

    state = _generic_var1_dynamics(factors=factors, y_scaled=Y)

    assert isinstance(state, _StateSpaceParams)
    assert state.F.shape == (4, 4)
    assert state.H.shape == (2, 4)
    assert state.b.shape == (2,)
    assert np.all(np.isfinite(state.F))
    assert np.all(np.isfinite(state.H))
    assert np.all(np.isfinite(state.b))


def test_linear_decoder_ddfm_dynamics_uses_decoder_weight_and_observed_mask():
    """_linear_decoder_ddfm_dynamics produces finite F, H with correct shapes."""
    n_timepoints = 25
    n_factors = 3
    n_targets = 5

    rng = np.random.RandomState(2)
    factors = rng.normal(size=(n_timepoints, n_factors))
    eps = rng.normal(size=(n_timepoints, n_targets))
    decoder_weight = rng.normal(size=(n_targets, n_factors + 1))
    # All observed in this synthetic case
    observed_y = np.ones((n_timepoints, n_targets), dtype=bool)

    state = _linear_decoder_ddfm_dynamics(
        factors=factors,
        eps=eps,
        decoder_weight=decoder_weight,
        observed_y=observed_y,
    )

    assert isinstance(state, _StateSpaceParams)
    assert state.F.shape == (n_factors, n_factors)
    assert state.H.shape == (n_targets, n_factors)
    assert state.b.shape == (n_targets,)
    assert np.all(np.isfinite(state.F))
    assert np.all(np.isfinite(state.H))
    assert np.all(np.isfinite(state.b))


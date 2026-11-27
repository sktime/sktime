"""Tests for the TensorFlow MCDCNN network."""

import numpy as np
import pytest

from sktime.networks.mcdcnn import MCDCNNNetwork
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(MCDCNNNetwork),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("params", MCDCNNNetwork.get_test_params())
@pytest.mark.parametrize("n_vars", [1, 3])
def test_mcdcnn_build_network(params, n_vars):
    """Check forward passes for every configuration and input type."""
    from tensorflow import keras

    keras.backend.clear_session()
    try:
        network = MCDCNNNetwork(**params)
        input_layers, output_layer = network.build_network((24, n_vars))
        model = keras.Model(inputs=input_layers, outputs=output_layer)

        X = np.random.default_rng(42).normal(size=(2, 24, n_vars)).astype(np.float32)
        output = model(network._prepare_input(X), training=False).numpy()

        assert output.shape == (2, network.dense_units)
        assert np.isfinite(output).all()
    finally:
        keras.backend.clear_session()

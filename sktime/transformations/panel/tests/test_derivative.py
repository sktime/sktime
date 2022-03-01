# -*- coding: utf-8 -*-
"""Test derivative transformer."""
import numpy as np

from sktime.distances.tests._utils import create_test_distance_numpy
from sktime.transformations.panel.derivative import DerivativeTransformer


def test_derivative():
    """Test derivative transformer."""
    X = create_test_distance_numpy(10, 10, 10)
    derivative_transformer = DerivativeTransformer()
    result = derivative_transformer.fit_transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 10

# -*- coding: utf-8 -*-
"""Tests for PCATransformer."""
import numpy as np
import pytest

from sktime.transformations.panel.pca import PCATransformer
from sktime.utils._testing.panel import _make_nested_from_array


@pytest.mark.parametrize("bad_components", ["str", 1.2, -1.2, -1, 11])
def test_bad_input_args(bad_components):
    """Check that exception is raised for bad input args."""
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if isinstance(bad_components, str):
        with pytest.raises(TypeError):
            PCATransformer(n_components=bad_components).fit(X)
    else:
        with pytest.raises(ValueError):
            PCATransformer(n_components=bad_components).fit(X)

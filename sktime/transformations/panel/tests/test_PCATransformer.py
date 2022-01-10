# -*- coding: utf-8 -*-
import numpy as np
import pytest

from sktime.exceptions import NotFittedError
from sktime.transformations.panel.pca import PCATransformer
from sktime.utils._testing.panel import _make_nested_from_array


# Check that exception is raised for bad input args.
@pytest.mark.parametrize("bad_components", ["str", 1.2, -1.2, -1, 11])
def test_bad_input_args(bad_components):
    X = _make_nested_from_array(np.ones(10), n_instances=10, n_columns=1)

    if isinstance(bad_components, str):
        with pytest.raises(TypeError):
            PCATransformer(n_components=bad_components).fit(X)
    else:
        with pytest.raises(ValueError):
            PCATransformer(n_components=bad_components).fit(X)


# Test that PCATransformer fails if attempt to transform before fit
def test_early_trans_fail():
    X = _make_nested_from_array(np.ones(10), n_instances=1, n_columns=1)
    pca = PCATransformer(n_components=1)

    with pytest.raises(NotFittedError):
        pca.transform(X)


# Test output format and dimensions.
@pytest.mark.parametrize(
    "n_instances,len_series,n_components",
    [
        (5, 2, 1),
        (5, 10, 1),
        (5, 10, 3),
        (5, 10, 5),
    ],
)

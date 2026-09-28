"""Tests for PCATransformer."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.pca import PCATransformer
from sktime.utils._testing.panel import _make_nested_from_array


@pytest.mark.skipif(
    not run_test_for_class(PCATransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


def test_set_params_is_applied_to_inner_pca_on_fit():
    """Check parameters changed after construction are used by sklearn PCA."""
    X = np.random.default_rng(0).normal(size=(10, 2, 4))
    transformer = PCATransformer(n_components=1, whiten=False)

    transformer.set_params(n_components=2, whiten=True)
    transformer.fit(X)

    assert transformer.get_params()["n_components"] == 2
    assert transformer.get_params()["whiten"] is True
    assert transformer.pca.n_components == 2
    assert transformer.pca.whiten is True
    assert transformer.pca.components_.shape[0] == 2

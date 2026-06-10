"""Channel selection test code."""

import pytest

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.channel_selection import ElbowClassPairwise


@pytest.mark.skipif(
    not run_test_for_class(ElbowClassPairwise),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cs_basic_motions():
    """Test channel selection on basic motions dataset."""
    X, y = load_basic_motions(split="train", return_X_y=True)

    ecp = ElbowClassPairwise()

    ecp.fit(X, y)

    # transform the training data

    ecp.transform(X, y)

    # test shape pf transformed data should be (n_samples, n_channels_selected)
    assert ecp.transform(X, y).shape == (X.shape[0], len(ecp.channels_selected_))

    # test shape of transformed data should be (n_samples, n_channels_selected)

    X_test, y_test = load_basic_motions(split="test", return_X_y=True)

    assert ecp.transform(X_test).shape == (X_test.shape[0], len(ecp.channels_selected_))

"""Tests for sklearn init specification conformance in deep learning regressors.

Regression tests for #10208: parameters must be stored exactly as passed
to ``__init__``, so that ``get_params``, ``clone`` and ``set_params`` work.
"""

__author__ = ["Rythamo8055"]

import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.base import clone

from sktime.regression.deep_learning.mcdcnn._mcdcnn_torch import (
    MCDCNNRegressorTorch,
)

torch_available = _check_soft_dependencies("torch", severity="none")


@pytest.mark.skipif(not torch_available, reason="torch not available")
def test_mcdcnn_regressor_torch_init_spec():
    """MCDCNNRegressorTorch stores optim and optim_kwargs as passed."""
    reg = MCDCNNRegressorTorch(optim_kwargs=None)
    params = reg.get_params()
    assert params["optim"] is None
    assert params["optim_kwargs"] is None

    # clone round-trip preserves parameters
    cloned = clone(reg)
    assert cloned.get_params() == params

    # explicit values are stored as-is
    reg2 = MCDCNNRegressorTorch(optim="Adam", optim_kwargs={"lr": 0.05})
    params2 = reg2.get_params()
    assert params2["optim"] == "Adam"
    assert params2["optim_kwargs"] == {"lr": 0.05}
    assert clone(reg2).get_params() == params2

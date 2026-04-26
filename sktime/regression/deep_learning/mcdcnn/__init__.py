"""MCDCNN deep learning regressors implemented in PyTorch."""

__all__ = [
    "MCDCNNRegressor",
    "MCDCNNRegressorTorch",
]

from sktime.regression.deep_learning.mcdcnn._mcdcnn_torch import MCDCNNRegressorTorch


class MCDCNNRegressor(MCDCNNRegressorTorch):
    """PyTorch implementation of ``MCDCNNRegressor``.

    This class keeps the legacy estimator name while using the
    ``MCDCNNRegressorTorch`` backend.
    """

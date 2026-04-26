"""LSTM-FCN deep learning regressors implemented in PyTorch."""

__all__ = [
    "LSTMFCNRegressor",
    "LSTMFCNRegressorTorch",
]

from sktime.regression.deep_learning.lstmfcn._lstmfcn_torch import (
    LSTMFCNRegressorTorch,
)


class LSTMFCNRegressor(LSTMFCNRegressorTorch):
    """PyTorch implementation of ``LSTMFCNRegressor``.

    This class keeps the legacy estimator name while using the
    ``LSTMFCNRegressorTorch`` backend.
    """

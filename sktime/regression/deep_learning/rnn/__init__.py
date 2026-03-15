"""RNN deep learning regressors implemented in PyTorch."""

__all__ = [
    "SimpleRNNRegressor",
    "SimpleRNNRegressorTorch",
]

from sktime.regression.deep_learning.rnn._rnn_torch import SimpleRNNRegressorTorch


class SimpleRNNRegressor(SimpleRNNRegressorTorch):
    """PyTorch implementation of ``SimpleRNNRegressor``.

    This class keeps the legacy estimator name while using the
    ``SimpleRNNRegressorTorch`` backend.
    """

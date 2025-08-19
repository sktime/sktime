# Pytorch and Tensorflow LSTMFCN regressor

"""Exports both TF and Torch versions of LSTMFCNRegressor for backwards compatibility."""

from sktime.regression.deep_learning._lstmfcn_tf.lstmfcn import LSTMFCNRegressorTF
from sktime.regression.deep_learning._lstmfcn_torch import LSTMFCNRegressorTorch

LSTMFCNRegressor = LSTMFCNRegressorTF

__all__ = [
    "LSTMFCNRegressor",     
    "LSTMFCNRegressorTF",  
    "LSTMFCNRegressorTorch" 
]
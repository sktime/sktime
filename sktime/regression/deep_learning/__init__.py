"""Deep learning based regressors."""
__all__ = [
    "CNNRegressor",
    "FCNRegressor",
    "LSTMFCNRegressor",
    "MCDCNNRegressor",
    "MLPRegressor",
    "ResNetRegressor",
    "SimpleRNNRegressor",
    "TapNetRegressor",
]

from sktime.regression.deep_learning.cnn import CNNRegressor
from sktime.regression.deep_learning.fcn import FCNRegressor
from sktime.regression.deep_learning.lstmfcn import LSTMFCNRegressor
from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressor
from sktime.regression.deep_learning.mlp import MLPRegressor
from sktime.regression.deep_learning.resnet import ResNetRegressor
from sktime.regression.deep_learning.rnn import SimpleRNNRegressor
from sktime.regression.deep_learning.tapnet import TapNetRegressor

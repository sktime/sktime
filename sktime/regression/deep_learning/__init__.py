"""Deep learning based regressors."""
__all__ = [
    "CNNRegressor",
    "MCDCNNRegressor",
    "ResNetRegressor",
    "SimpleRNNRegressor",
    "TapNetRegressor",
]

from sktime.regression.deep_learning.cnn import CNNRegressor
from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressor
from sktime.regression.deep_learning.resnet import ResNetRegressor
from sktime.regression.deep_learning.rnn import SimpleRNNRegressor
from sktime.regression.deep_learning.tapnet import TapNetRegressor

"""Deep learning based regressors."""

__all__ = [
    "CNNRegressor",
    "CNTCRegressor",
    "FCNRegressor",
    "InceptionTimeRegressor",
    "LSTMFCNRegressor",
    "MACNNRegressor",
    "MCDCNNRegressor",
    "MLPRegressor",
    "ResNetRegressor",
    "SimpleRNNRegressor",
    "TapNetRegressor",
]

from sktime.regression.deep_learning.cnn import CNNRegressor
from sktime.regression.deep_learning.cntc import CNTCRegressor
from sktime.regression.deep_learning.fcn import FCNRegressor
from sktime.regression.deep_learning.inceptiontime import InceptionTimeRegressor
from sktime.regression.deep_learning.lstmfcn import LSTMFCNRegressor
from sktime.regression.deep_learning.macnn import MACNNRegressor
from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressor
from sktime.regression.deep_learning.mlp import MLPRegressor
from sktime.regression.deep_learning.resnet import ResNetRegressor
from sktime.regression.deep_learning.rnn import SimpleRNNRegressor
from sktime.regression.deep_learning.tapnet import TapNetRegressor

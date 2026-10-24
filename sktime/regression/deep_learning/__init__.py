"""Deep learning based regressors."""

__all__ = [
    "CNNRegressor",
    "CNNRegressorTorch",
    "CNTCRegressor",
    "CNTCRegressorTorch",
    "FCNRegressor",
    "FCNRegressorTorch",
    "InceptionTimeRegressor",
    "InceptionTimeRegressorTorch",
    "LSTMFCNRegressor",
    "MACNNRegressor",
    "MACNNRegressorTorch",
    "MCDCNNRegressor",
    "MCDCNNRegressorTorch",
    "MLPRegressor",
    "MLPRegressorTorch",
    "ResNetRegressor",
    "ResNetRegressorTorch",
    "SimpleRNNRegressor",
    "SimpleRNNRegressorTorch",
    "TapNetRegressor",
    "TapNetRegressorTorch",
]

from sktime.regression.deep_learning.cnn import CNNRegressor, CNNRegressorTorch
from sktime.regression.deep_learning.cntc import CNTCRegressor, CNTCRegressorTorch
from sktime.regression.deep_learning.fcn import (
    FCNRegressor,
    FCNRegressorTorch,
)
from sktime.regression.deep_learning.inceptiontime import (
    InceptionTimeRegressor,
    InceptionTimeRegressorTorch,
)
from sktime.regression.deep_learning.lstmfcn import LSTMFCNRegressor
from sktime.regression.deep_learning.macnn import MACNNRegressor, MACNNRegressorTorch
from sktime.regression.deep_learning.mcdcnn import (
    MCDCNNRegressor,
    MCDCNNRegressorTorch,
)
from sktime.regression.deep_learning.mlp import (
    MLPRegressor,
    MLPRegressorTorch,
)
from sktime.regression.deep_learning.resnet import (
    ResNetRegressor,
    ResNetRegressorTorch,
)
from sktime.regression.deep_learning.rnn import (
    SimpleRNNRegressor,
    SimpleRNNRegressorTorch,
)
from sktime.regression.deep_learning.tapnet import (
    TapNetRegressor,
    TapNetRegressorTorch,
)

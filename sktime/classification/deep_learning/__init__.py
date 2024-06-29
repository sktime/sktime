"""Deep learning based classifiers."""

__all__ = [
    "CNNClassifier",
    "CNTCClassifier",
    "FCNClassifier",
    "InceptionTimeClassifier",
    "LSTMFCNClassifier",
    "MACNNClassifier",
    "MCDCNNClassifier",
    "MLPClassifier",
    "ResNetClassifier",
    "SimpleRNNClassifier",
    "TapNetClassifier",
]

from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.deep_learning.cntc import CNTCClassifier
from sktime.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
from sktime.classification.deep_learning.macnn import MACNNClassifier
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier
from sktime.classification.deep_learning.mlp import MLPClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.deep_learning.rnn import SimpleRNNClassifier
from sktime.classification.deep_learning.tapnet import TapNetClassifier

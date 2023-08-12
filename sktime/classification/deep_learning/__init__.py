"""Deep learning based classifiers."""
__all__ = [
    "CNNClassifier",
    "FCNClassifier",
    "InceptionTimeClassifier",
    "LSTMFCNClassifier",
    "MLPClassifier",
    "TapNetClassifier",
]

from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
from sktime.classification.deep_learning.mlp import MLPClassifier
from sktime.classification.deep_learning.tapnet import TapNetClassifier

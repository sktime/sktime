from ._classifiers import TSDummyClassifier, TSExampleClassifier
from ._regressors import TSDummyRegressor, TSExampleRegressor

from ._version import __version__

__all__ = ['TSDummyClassifier',
           'TSDummyRegressor',
           'TSExampleRegressor',
           'TSExampleClassifier',
           '__version__']

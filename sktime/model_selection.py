'''
classes and functions for model validation
'''
from sklearn.model_selection import GridSearchCV as skGSCV
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sktime.regressors.base import BaseRegressor
from sktime.classifiers.base import BaseClassifier


class GridSearchCV(skGSCV):
    '''
    A wrapper to provide default scorers
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scoring is None:
            # using accuracy score as default for classifiers
            if isinstance(self.estimator, BaseClassifier):
                self.scoring = make_scorer(accuracy_score)
            # using mean squared error as default for regressors
            elif isinstance(self.estimator, BaseRegressor):
                self.scoring = make_scorer(mean_squared_error)

from classifiers.proximity_forest.distance_measure_parameter_permutations.parameter_space import ParameterSpace

import numpy as np


class PredefinedLambda(ParameterSpace):
    lambda_values_key = 'lambda_values'

    def __init__(self, **params):
        self._lambda_values = np.array([])
        super(PredefinedLambda, self).__init__(**params)

    def set_params(self, **params):
        super(PredefinedLambda, self).set_params(**params)
        self._lambda_values = np.array(
            [0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
             0.077777778, 0.088888889, 0.1])

    def get_params(self):
        return {self.lambda_values_key: self._lambda_values, **super(PredefinedLambda, self).get_params()}

    def hasNext(self):
        return True

    def next(self):
        rand = self.get_rand()
        return rand.choice(self._lambda_values)

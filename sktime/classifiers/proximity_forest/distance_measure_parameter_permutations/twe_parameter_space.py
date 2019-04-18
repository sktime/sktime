import numpy as np

from classifiers.proximity_forest.distance_measure_parameter_permutations.distance_measure_parameter_space import \
    DistanceMeasureParameterSpace
from classifiers.proximity_forest.distance_measure_parameter_permutations.dtw_parameter_space import DtwParameterSpace
from classifiers.proximity_forest.dms.Twe import Twe
from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint


class TweParameterSpace(DistanceMeasureParameterSpace):
    'Twe parameter space'

    nu_values_key = 'nu_values'
    lambda_values_key = 'lambda_values_key'

    def __init__(self, **params):
        self._lambda_values = np.array([0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
                                        0.077777778, 0.088888889, 0.1])
        self._nu_values = np.array([0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
        super(TweParameterSpace, self).__init__(**params)

    def get_params(self):
        return {self.nu_values_key: self._nu_values,
                self.lambda_values_key: self._lambda_values}

    def get_random_parameter_permutation(self):
        nu = self._rand.choice(self._nu_values)
        lambda_value = self._rand.choice(self._lambda_values)
        return {Twe.nu_key: nu, }


if __name__ == "__main__":
    instances, class_labels = load_gunpoint(return_X_y=True)
    ps = TweParameterSpace(**{TweParameterSpace.instances_key: instances})
    print(ps.get_params())

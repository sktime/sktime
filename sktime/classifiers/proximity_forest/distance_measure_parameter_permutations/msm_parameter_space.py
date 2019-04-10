import numpy as np

from classifiers.proximity_forest.distance_measure_parameter_permutations.dtw_parameter_space import DtwParameterSpace
from classifiers.proximity_forest.dms.msm import Msm
from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint


class MsmParameterSpace(DtwParameterSpace):
    'Msm parameter space'

    cost_values_key = 'cost_values'

    def __init__(self, **params):
        super(MsmParameterSpace, self).__init__(**params)
        self._cost_values = np.array(
            [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375, 0.0475, 0.05125,
             0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125, 0.085, 0.08875, 0.0925, 0.09625, 0.1,
             0.136, 0.172, 0.208,
             0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748,
             0.784, 0.82, 0.856,
             0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68, 6.04,
             6.4, 6.76, 7.12,
             7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6,
             53.2, 56.8, 60.4,
             64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100])

    def get_params(self):
        return {**super(MsmParameterSpace, self).get_params(), self.cost_values_key: self._cost_values}

    def get_random_parameter_permutation(self):
        cost = self._rand.choice(self._cost_values)
        return {Msm.cost_key: cost, Msm.delta_key: 1}  # msm has no window


if __name__ == "__main__":
    instances, class_labels = load_gunpoint(return_X_y=True)
    ps = MsmParameterSpace(**{MsmParameterSpace.instances_key: instances})
    print(ps.get_params())

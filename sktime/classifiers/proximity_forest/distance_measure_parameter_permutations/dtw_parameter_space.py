from classifiers.proximity_forest.distance_measure_parameter_permutations.distance_measure_parameter_space import \
    DistanceMeasureParameterSpace
import numpy as np

from classifiers.proximity_forest.dms.dtw import Dtw
from datasets import load_gunpoint


class DtwParameterSpace(DistanceMeasureParameterSpace):
    'Dtw parameter space'

    max_delta_key = 'max_delta'

    def __init__(self, **params):
        self._max_delta = -1
        super(DtwParameterSpace, self).__init__(**params)

    def set_params(self, **params):
        super(DtwParameterSpace, self).set_params(**params)
        instances = params[self.instances_key]  # will raise an exception if instances not in params
        instance = instances.iloc[0, 0]
        self._max_delta = instance.shape[0] * 0.25

    def get_params(self):
        return {self.max_delta_key: self._max_delta}

    def get_random_parameter_permutation(self):
        delta = self._rand.random() * self._max_delta  # todo should delta be rounded to integer?
        return {Dtw.delta_key: delta}


if __name__ == "__main__":
    instances, class_labels = load_gunpoint(return_X_y=True)
    ps = DtwParameterSpace(**{DtwParameterSpace.instances_key: instances})
    print(ps.get_params())

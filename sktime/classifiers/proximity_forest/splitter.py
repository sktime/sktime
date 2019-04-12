from scipy.stats import rv_continuous, rv_discrete

from classifiers.proximity_forest.randomised import Randomised
import numpy as np

from classifiers.proximity_forest.utilities import Utilities


class Splitter(Randomised):

    def __init__(self, **params):
        self._exemplars = None
        self._distance_measure_permutation = None
        self._distance_measure_pool = None
        self._exemplar_selector = None
        super(Splitter, self).__init__(**params)

    def get_params(self):
        raise Exception('not implemented')

    def get_exemplars(self):
        return self._exemplars

    def get_distance_measure(self):

    def _get_param_permutation(self, params): # list of param dicts

        # example:
        # param_grid = [
        #   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #   {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}], 'kernel': ['rbf']},
        #  ]

        param_pool = self.get_rand().choice(params)
        return self.__pick_param_permutation(param_pool)


    def __pick_param_permutation(self, param_pool): # dict of params
        param_permutation = {}
        for param_name, param_values in param_pool.items():
            if isinstance(param_values, list):
                param_value = self.get_rand().choice(param_values)
                if isinstance(param_value, dict):
                    param_value = self._get_param_permutation(param_value)
            elif isinstance(param_values, rv_continuous) or isinstance(param_values, rv_discrete):
                param_value = param_values.rvs()
            else:
                raise Exception('unknown type')
            param_permutation[param_name] = param_value
        return param_permutation

    distance_measure_key = 'distance_measure'
    distance_measure_params_key = 'distance_measure_params'

    def split(self, instances, class_labels):
        exemplars = self._exemplar_selector.select(instances, class_labels)
        self._exemplars = exemplars
        num_exemplars = len(exemplars)
        distance_measure_permutation = self._get_param_permutation(self._distance_measure_pool)
        distance_measure = distance_measure_permutation.pop(self.distance_measure_key)
        self._distance_measure_permutation = {
            self.distance_measure_key: distance_measure,
            self.distance_measure_params_key: distance_measure_permutation
        }
        instance_bins = np.empty(num_exemplars, dtype=np.ndarray)
        class_label_bins = np.empty(num_exemplars, dtype=np.ndarray)
        distances = np.empty(num_exemplars, dtype=float)
        num_instances = instances.shape(0)
        for instance_index in range(0, num_instances):
            instance = instances.iloc[instance_index] # need to convert to nparray
            for exemplar_index in range(0, num_exemplars):
                exemplar = exemplars[exemplar_index]
                distance = distance_measure(exemplar, instance, **distance_measure_permutation)
                distances[exemplar_index] = distance
            min_index = Utilities.arg_min(distances, self.get_rand())
            instance_bins[min_index].append(instance)
            class_label_bins[min_index].append(class_labels[instance_index])
        return instance_bins, class_label_bins

    distance_measure_pool_key = 'distance_measure_pool'
    exemplar_selector_key = 'exemplar_selector'

    def set_params(self, **params):
        super(Splitter, self).set_params(**params)
        self._exemplar_selector = params[self.exemplar_selector_key]
        self._distance_measure_pool = params[self.distance_measure_pool_key]

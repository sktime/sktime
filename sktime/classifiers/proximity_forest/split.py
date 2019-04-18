from scipy.stats import rv_continuous, rv_discrete

from classifiers.proximity_forest.randomised import Randomised
import numpy as np

from classifiers.proximity_forest.utilities import Utilities


class Split(Randomised):

    def __init__(self, **params):
        self._distance_measure_params = None
        self._exemplars = None
        self._distance_measure = None
        self._bins = None
        self._gain = None
        super(Split, self).__init__(**params)

    def get_params(self):
        raise Exception('not implemented')

    def get_exemplars(self):
        return self._exemplars

    def get_distance_measure(self):
        raise Exception('not implemented')

    def set_params(self, **params):
        super(Split, self).set_params(**params)
        params = params.copy()
        self._distance_measure = params.pop(self.distance_measure_key)
        self._exemplars = params.pop(self.exemplars_key)
        self._distance_measure_params = params

    exemplars_key = 'exemplars'
    exemplar_instance_key = 'exemplar_instance'
    exemplar_class_key = 'exemplar_class'
    distance_measure_key = 'dm'

    def get_bins(self):
        return self._bins

    def get_gain(self):
        return self._gain

    def split(self, instance_bins):
        bins = {}
        for index in range(0, len(self._exemplars)):
            bins[index] = {}
        for class_label in instance_bins.keys():
            for index in range(0, len(self._exemplars)):
                bins[index][class_label] = []
            instance_bin = instance_bins[class_label]
            for instance in instance_bin:
                index = self.find_closest_exemplar_index(instance)
                bins[index][class_label].append(instance)
        self._bins = bins
        self._gain = self._find_gain()

    def _find_gain(self):
        return -1 # todo

    def find_closest_exemplar_index(self, instance):
        closest_exemplars_indices = []
        min_distance = np.Infinity
        for index in range(0, len(self._exemplars)):
            exemplar_entry = self._exemplars[index]
            exemplar = exemplar_entry[self.exemplar_instance_key]
            distance = self._distance_measure(exemplar, instance, **self._distance_measure_params)
            if distance <= min_distance:
                if distance < min_distance:
                    closest_exemplars_indices.clear()
                closest_exemplars_indices.append(index)
        closest_exemplar_index = self.get_rand().choice(closest_exemplars_indices)
        return closest_exemplar_index


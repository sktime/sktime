from sklearn.base import BaseEstimator, ClassifierMixin

from classifiers.proximity_forest.one_per_class_selector import OnePerClassSelector
from classifiers.proximity_forest.randomised import Randomised
from classifiers.proximity_forest.splitter import Splitter
import numpy as np

class Tree(Randomised, BaseEstimator, ClassifierMixin):

    distance_measure_pool_key = 'distance_measure_pool'

    def __init__(self, **params):
        self._distance_measure_pool = None
        super(Tree, self).__init__(**params)

    def fit(self, instances, class_labels):
        self.branch(instances, class_labels)

    def branch(self, instances, class_labels):
        splitter = Splitter(**{Splitter.distance_measure_pool_key: self._distance_measure_pool,
                               Splitter.exemplar_selector_key: OnePerClassSelector()})
        instance_bins, class_label_bins = splitter.split(instances, class_labels)
        for branch_index in range(0, len(instance_bins)):
            class_labels = class_label_bins[branch_index]
            num_classes = np.count_nonzero(class_labels)
            if num_classes == 1:
                # node is pure, stop branching
                pass
            else:
                instances = instance_bins[branch_index]
                self.branch(instances, class_labels)

    def set_params(self, **params):
        super(Tree, self).set_params(**params)
        self._distance_measure_pool = params[self.distance_measure_pool_key]

    def get_params(self):
        raise Exception('not implemented')





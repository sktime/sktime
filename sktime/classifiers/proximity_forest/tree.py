from scipy.stats import rv_continuous, rv_discrete
from sklearn.base import BaseEstimator, ClassifierMixin

from classifiers.proximity_forest.dms.dtw import Dtw
from classifiers.proximity_forest.dms.erp import Erp
from classifiers.proximity_forest.dms.lcss import Lcss
from classifiers.proximity_forest.randomised import Randomised
from classifiers.proximity_forest.split import Split
import numpy as np

from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint
from distances.elastic_cython import dtw_distance, lcss_distance, erp_distance


class Tree(Randomised, BaseEstimator, ClassifierMixin):
    distance_measure_pool_key = 'distance_measure_pool'
    class_labels_key = 'class_label_bins'
    instances_key = 'instance_bins'
    r_key = 'r'

    def __init__(self, **params):
        self._distance_measure_pool = None
        self._r = None
        self._split = None
        self._branches = None
        super(Tree, self).__init__(**params)

    def _predict_proba_inst(self, instance):
        pass

    def predict_proba(self, instances):
        # todo unpack panda

        for instance in instances:
            pass
        pass

    def fit(self, instances, class_labels):
        # todo unpack instances into numpy arr
        binned_instances = Utilities.bin_instances_by_class(instances, class_labels)
        self._branch(binned_instances)

    def _get_rand_split(self, binned_instances):
        distance_measure_permutation = self._get_rand_distance_measure_permutation()
        exemplars = self._get_rand_exemplars(binned_instances)
        split = Split(**{**distance_measure_permutation, Split.exemplars_key: exemplars})
        return split

    def _get_best_split(self, binned_instances):
        split = self._get_rand_split(binned_instances)
        split.split(binned_instances)
        splits = [split]
        best_gain_so_far = split.get_gain()
        for r_index in range(1, self._r):
            split = self._get_rand_split(binned_instances)
            gain = split.get_gain()
            if gain >= best_gain_so_far:
                if gain > best_gain_so_far:
                    best_gain_so_far = gain
                    splits.clear()
                splits.append(split)
        split = self.get_rand().choice(splits)
        return split

    def _should_branch(self, binned_instances):
        # test if tree is pure
        return

    def _branch(self, binned_instances):
        self._branches = []
        self._split = self._get_best_split(binned_instances)
        if self._should_branch(binned_instances):
            for bin in binned_instances:
                tree = Tree(**{self.distance_measure_pool_key: self._distance_measure_pool,
                               self.r_key: self._r})
                self._branches.append(tree)
                tree._branch(bin)

    def set_params(self, **params):
        super(Tree, self).set_params(**params)
        self._distance_measure_pool = params[self.distance_measure_pool_key]
        self._r = params.get(self.r_key, 1)

    def get_params(self):
        raise Exception('not implemented yet') # todo

    def _get_rand_distance_measure_permutation(self, params=None):  # list of param dicts todo split into two methods
        # example:
        # param_grid = [
        #   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #   {'C': [1, 10, 100, 1000], 'gamma': [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}], 'kernel': ['rbf']},
        #  ]
        if params is None:
            params = self._distance_measure_pool
        param_pool = self.get_rand().choice(params)
        permutation = self.__pick_param_permutation(param_pool)
        return permutation

    def __pick_param_permutation(self, param_pool):  # dict of params
        param_permutation = {}
        for param_name, param_values in param_pool.items():
            if isinstance(param_values, list):
                param_value = self.get_rand().choice(param_values)
                if isinstance(param_value, dict):
                    param_value = self._get_rand_distance_measure_permutation(param_value)
            elif isinstance(param_values, rv_continuous) or isinstance(param_values, rv_discrete):
                param_value = param_values.rvs()
            else:
                raise Exception('unknown type')
            param_permutation[param_name] = param_value
        return param_permutation

    def _get_rand_exemplars(self, binned_instances):
        exemplars = []
        for class_label in binned_instances.keys():
            bin = binned_instances[class_label]
            exemplar_index = self.get_rand().randint(0, len(bin))
            instance = bin[exemplar_index]
            exemplars.append({Split.exemplar_class_key: class_label, Split.exemplar_instance_key: instance})
        return exemplars



    # todo str + other builtins


if __name__ == "__main__":
    x_train, y_train = load_gunpoint(return_X_y=True)
    x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
    params = [
        {Split.distance_measure_key: [Dtw()],
         Dtw.delta_key: [1, 2, 3, 4, 5, 6, 7]},
        {Split.distance_measure_key: [Lcss()],
         Lcss.delta_key: [1, 2, 3, 4, 5, 6, 7],
         Lcss.epsilon_key: [1, 2, 4, 5, 6, 7, 2]},
        {Split.distance_measure_key: [Erp()],
         Erp.delta_key: [1, 2, 3, 4, 5, 6, 7],
         Erp.g_key: [1, 2, 4, 5, 6, 7, 2]}
    ]
    tree = Tree(**{Randomised.rand_state_key: np.random.RandomState(3), Tree.distance_measure_pool_key: params})
    tree.fit(x_train, y_train)

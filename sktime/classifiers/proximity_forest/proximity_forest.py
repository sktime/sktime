# author: George Oastler (g.oastler@uea.ac.uk, linkedin.com/goastler)
from scipy.stats import uniform
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

from classifiers.proximity_forest.dms.dtw import Dtw
from classifiers.proximity_forest.dms.erp import Erp
from classifiers.proximity_forest.dms.lcss import Lcss
from classifiers.proximity_forest.dms.msm import Msm
from classifiers.proximity_forest.dms.twe import Twe
from classifiers.proximity_forest.dms.wddtw import Wddtw
from classifiers.proximity_forest.dms.wdtw import Wdtw
from classifiers.proximity_forest.parameterised import Parameterised
from classifiers.proximity_forest.randomised import Randomised
from classifiers.proximity_forest.utilities import Utilities
from datasets import load_gunpoint
from distance_measures.elastic import dtw_distance, lcss_distance, erp_distance, msm_distance


class ProximityForest(Parameterised, Randomised):
    'Proximity Forest - a distance-based time-series classifier. https://arxiv.org/pdf/1808.10594.pdf'

    _seed_key = 'seed'
    _verbose_key = 'verbose'

    def __get_erp_g_range(self, instances):
        stdp = Utilities.stdp(instances)
        max_epsilon = 0.8 * stdp
        min_epsilon = 0.2 * stdp
        return uniform(loc=min_epsilon, scale=max_epsilon - min_epsilon)

    def __get_dtw_delta_range(self, instances):
        instance = instances.iloc[0, 0]
        max_delta = instance.shape[0] * 0.25
        return uniform(loc=0, scale=max_delta)

    def __get_lcss_epsilon_range(self, instances):
        stdp = Utilities.stdp(instances)
        max_epsilon = stdp
        min_epsilon = 0.2 * stdp
        return uniform(loc = min_epsilon, scale = max_epsilon - min_epsilon)

    def get_default_dtw_param_pool(self, instances):
        return {
            Dtw.delta_key: self.__get_dtw_delta_range(instances)
        }

    def get_default_wdtw_param_pool(self, instances):
        return {
            Wddtw.g_key: uniform(loc=0, scale=1)
        }

    def get_default_lcss_param_pool(self, instances)
        return {
            Lcss.delta_key: self.__get_dtw_delta_range(instances),
            Lcss.epsilon_key: self.__get_lcss_epsilon_range(instances)
        }

    def get_default_msm_param_pool(self, instances):
        return {
            Msm.cost_key: [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375, 0.0475, 0.05125,
        0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125, 0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208,
        0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784, 0.82, 0.856,
        0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12,
        7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8, 60.4,
        64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100]
        }

    def get_default_erp_param_pool(self, instances):
        return {
            Erp.delta_key: self.__get_dtw_delta_range(instances),
            Erp.g_key: self.__get_erp_g_range(instances)
        }

    def get_default_twe_param_pool(self, instances):
        return {
            Twe.lambda_key: [0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
        0.077777778, 0.088888889, 0.1],
            Twe.nu_key: [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        }

    dm_key = 'dm'
    dm_param_pool_obtainer_key = 'dmp'
    erp_key = 'erp'
    twe_key = 'twe'
    msm_key = 'msm'
    dtw_key = 'dtw'
    ddtw_key = 'ddtw'
    wdtw_key = 'wdtw'
    wddtw_key = 'wddtw'
    lcss_key = 'lcss'

    def get_default_distance_measure_pool(self):
        return {
            # 'twe': {
            #     self._distance_measure_key: twe_distance,
            #     self._distance_measure_param_getter: get_default_twe_params
            # },
            self.erp_key: {
                self.dm_key: erp_distance,
                self.dm_param_pool_obtainer_key: self.get_default_erp_param_pool
            },
            self.msm_key: {
                self.dm_key: msm_distance,
                self.dm_param_pool_obtainer_key: self.get_default_msm_param_pool
            },
            self.lcss_key: {
                self.dm_key: lcss_distance,
                self.dm_param_pool_obtainer_key: self.get_default_lcss_param_pool
            },
            self.dtw_key: {
                self.dm_key: dtw_distance,
                self.dm_param_pool_obtainer_key: self.get_default_dtw_param_pool
            },
            # self.ddtw_key: {
            #     self.dm_key: ddtw_distance,
            #     self.dm_param_pool_obtainer_key: self.get_default_dtw_param_range
            # },
            # self.wdtw_key: {
            #     self.dm_key: wdtw_distance,
            #     self.dm_param_pool_obtainer_key: self.get_default_wdtw_param_range
            # },
            # self.wddtw_key: {
            #     self.dm_key: wddtw_distance,
            #     self.dm_param_pool_obtainer_key: self.get_default_wddtw_param_range
            # },
        }

    def __init__(self, **params):
        self._distance_measure_pool = None # distance measure pool of distance measures and their associated parameter pool
        if self._distance_measure_pool_key not in params:
            params.update({self._distance_measure_pool_key: self.get_default_distance_measure_pool()})
        super(ProximityForest, self).__init__(**params)

    def fit(self, x, y):
        # check x and y have correct shapes
        x, y = check_X_y(x, y)
        self._classes = unique_labels(y)
        self._x = x
        self._y = y

    def predict(self, x):
        raise Exception('not implemented')

    def predict_proba(self, x):
        raise Exception('not implemented')

    def get_params(self, deep=True):
        return {**super(ProximityForest, self).get_params(),

                }

    _distance_measure_pool_key = 'dmpool'
    # wants:
    #   - super params
    #   - distance measure pool
    #   - distance measure parameter pool for each distance measure
    def set_params(self, **params):
        super(ProximityForest, self).set_params(**params)
        self._distance_measure_pool = params[self._distance_measure_pool] # will throw exception if not in params
        # todo checks to make sure distance pool in correct structure, i.e.
    # {
    #   <distance_measure_name>: {
    #       <dict of params with either list of values or distribution
    #   }
    #
    # }

    def __str__(self):
        return 'ProximityForest'

# from sklearn.utils.estimator_checks import check_estimator
#
# pf = ProximityForest(seed=5, verbose=True, dms=('dtw', 'lcss', 'msm'))
# x_train, y_train = load_gunpoint(return_X_y=True)
# x_test, y_test = load_gunpoint(split='TEST', return_X_y=True)
# pf.fit(x_train, y_train)
# predictions = pf.predict_proba(x_test)
#
# score = accuracy_score(y_test, predictions)
# print(score)

# nb = GaussianNB()
# iris = datasets.load_iris()
# nb.fit(iris.data, iris.target)
# predictions = nb.predict(iris.data)
# cheating really
# score = accuracy_score(iris.target, predictions)
# print(score)

# check_estimator(pf)

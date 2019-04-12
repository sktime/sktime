# author: George Oastler (g.oastler@uea.ac.uk, linkedin.com/goastler)
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

from datasets import load_gunpoint

class ProximityForest:
    'Proximity Forest - a distance-based time-series classifier. https://arxiv.org/pdf/1808.10594.pdf'

    _seed_key = 'seed'
    _verbose_key = 'verbose'
    _distance_measures_key = 'dms'
    _default_distance_measures = ('dtw', 'lcss', 'msm')

    def __init__(self, **parameters):
        self.__parameters = parameters.copy()
        self.__parameters.setdefault(self._seed_key, 0)
        self.__parameters.setdefault(self._verbose_key, True)
        self.__parameters.setdefault(self._distance_measures_key, self._default_distance_measures)
        if self.is_verbose():
            print(self, self.get_params())

    def is_verbose(self):
        return self.__parameters.get(self._verbose_key)

    def get_seed(self):
        return self.__parameters.get(self._seed_key)

    def get_distance_measures(self):
        return self.__parameters.get(self._distance_measures_key)

    def fit(self, x, y):
        # check x and y have correct shapes
        x, y = check_X_y(x, y)
        self._classes = unique_labels(y)
        self._x = x
        self._y = y

    def predict(self, x):
        return

    def predict_proba(self, x):
        return

    def get_params(self, deep=True):
        return self.__parameters.copy()  # todo deep copy, test

    def set_params(self, **parameters):
        self.__parameters.update(parameters)  # todo test

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

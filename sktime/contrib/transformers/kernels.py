import numpy as np
import pyximport
import sklearn.model_selection
from scipy import spatial, stats
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import check_random_state

import sktime.datasets
import sktime.utils.dataset_properties
from sktime.classifiers.base import BaseClassifier
from sktime.transformers.base import BaseTransformer
from sktime.utils.data_container import tabularise

pyximport.install()
from sktime.distances.elastic_cython import (
    wdtw_distance, ddtw_distance, wddtw_distance, msm_distance, lcss_distance,
    erp_distance, dtw_distance,
    twe_distance,
    )
from sktime.pipeline import Pipeline
import pandas as pd


class PandasToNumpy(BaseTransformer):

    def __init__(self,
                 cls = None,
                 unpack_train = True,
                 unpack_test = True):
        self.cls = cls
        self.unpack_train = unpack_train
        self.unpack_test = unpack_test

    def transform(self, X, y = None):
        if self.unpack_train and isinstance(X, pd.DataFrame):
            X = tabularise(X, return_array = True)
        return X


def unpack_series_row(ts):
    if isinstance(ts, pd.Series):
        ts = ts.values
    return ts


def unpack_series(ts):
    ts = np.reshape(ts, (ts.shape[0], 1))
    return ts


class RbfKernel(BaseEstimator, TransformerMixin):

    def __init__(self,
                 sigma = 1.0):
        self.sigma = sigma

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        kernel = np.exp(-(X ** 2) / (self.sigma ** 2))
        return kernel


class PairwiseKernel(BaseEstimator, TransformerMixin):

    def __init__(self):
        if type(self) == PairwiseKernel:
            raise NotImplementedError("abstract")
        self.X_train = None

    def _distance(self, s1, s2):
        raise NotImplementedError("abstract")

    def fit(self, X, y = None):
        self.X_train = X
        return self

    def transform(self, X, y = None):
        kernel = cdist(X, self.X_train, metric = self._distance)
        return kernel


class TriKernel(PairwiseKernel):  # rbf
    def __init__(self):
        super().__init__()

    def _distance(self, s1, s2):
        dist = spatial.distance.cosine(s1, s2)
        return dist


class PolyKernel(PairwiseKernel):  # rbf
    def __init__(self, degree = 4):
        super().__init__()
        self.degree = degree

    def _distance(self, s1, s2):
        coef1 = np.polynomial.polynomial.polyfit(range(0, len(s1)), s1, self.degree)
        coef2 = np.polynomial.polynomial.polyfit(range(0, len(s2)), s2, self.degree)
        dist = np.linalg.norm(coef1 - coef2)
        return dist


class Kl2Kernel(PairwiseKernel):  # rbf
    def __init__(self, degree = 4):
        super().__init__()
        self.degree = degree

    def _distance(self, s1, s2):  # todo unpack missing might cause issues
        coef1, residuals1, rank, singular_values, rcond = np.polyfit(range(len(s1)), s1, full = True, deg = self.degree)
        coef2, residuals2, rank, singular_values, rcond = np.polyfit(range(len(s2)), s2, full = True, deg = self.degree)
        means1 = np.polyval(np.polyfit(s1, range(len(s1)), deg = self.degree), range(len(s1)))
        means2 = np.polyval(np.polyfit(s2, range(len(s2)), deg = self.degree), range(len(s2)))
        samples1 = np.random.normal(0, np.sqrt(residuals1), len(means1))
        samples1 = samples1 + means1
        samples2 = np.random.normal(0, np.sqrt(residuals2), len(means2))
        samples2 = samples2 + means2
        dist = stats.entropy(samples1, samples2)
        return dist


class HellKernel(PairwiseKernel):  # rbf
    def __init__(self, degree = 4):
        super().__init__()
        self.degree = degree

    def _distance(self, s1, s2):
        coef1, residuals1, rank, singular_values, rcond = np.polyfit(range(len(s1)), s1, full = True, deg = self.degree)
        coef2, residuals2, rank, singular_values, rcond = np.polyfit(range(len(s2)), s2, full = True, deg = self.degree)
        means1 = np.polyval(np.polyfit(s1, range(len(s1)), deg = self.degree), range(len(s1)))
        means2 = np.polyval(np.polyfit(s2, range(len(s2)), deg = self.degree), range(len(s2)))
        samples1 = np.random.normal(0, np.sqrt(residuals1), len(means1))
        samples1 = samples1 + means1
        min1 = np.min(samples1)
        if min1 < 0:
            samples1 = np.add(samples1, -min1)
        samples2 = np.random.normal(0, np.sqrt(residuals2), len(means2))
        samples2 = samples2 + means2
        min2 = np.min(samples2)
        if min2 < 0:
            samples2 = np.add(samples2, -min2)
        sqrt1 = np.sqrt(samples1)
        sqrt2 = np.sqrt(samples2)
        sqrt_tot = np.sum((sqrt1 - sqrt2) ** 2)
        dist = np.sqrt(sqrt_tot) / np.sqrt(2)
        return dist


class NegToZero(BaseTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, X, y = None):
        kernel = np.where(X > 0, X, 0)
        return kernel


class NegToAbs(BaseTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, X, y = None):
        kernel = np.where(X > 0, X, np.abs(X))
        return kernel


class NegToMin(BaseTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, X, y = None):
        min = np.min(X)
        result = X - min
        return result


class EigKernel(BaseTransformer):
    def __init__(self, transformer = NegToZero()):
        super().__init__()
        self.transformer = transformer
        self.X_train_ = None

    def transform(self, X, y = None):
        if X.shape[0] != X.shape[1]:
            eigen_values, eigen_vectors = np.linalg.eig(self.X_train_)
        else:
            eigen_values, eigen_vectors = np.linalg.eig(X)
        eigen_values = np.real(eigen_values)
        eigen_vectors = np.real(eigen_vectors)
        # ------ mod eig values
        if len(eigen_values[eigen_values < 0]) > 0:
            eigen_values = self.transformer.transform(eigen_values)
        diag = np.diag(eigen_values)
        inv_eigen_vectors = np.linalg.pinv(eigen_vectors)
        regularized = np.matmul(np.matmul(eigen_vectors, diag), inv_eigen_vectors)
        if X.shape[0] != X.shape[1]:
            P = np.matmul(regularized, np.linalg.pinv(self.X_train_))
            return np.matmul(X, P)
        else:
            return regularized
        # old version
        # if X.shape[0]==X.shape[1]:
        #     eigen_values, eigen_vectors = np.linalg.eig(X)
        #     eigen_values = np.real(eigen_values)
        #     eigen_vectors = np.real(eigen_vectors)
        #     # ------ mod eig values
        #     if len(eigen_values[eigen_values < 0]) > 0:
        #         eigen_values = self.transform_eigen_values(eigen_values)
        #     diag = np.diag(eigen_values)
        #     inv_eigen_vectors = np.linalg.pinv(eigen_vectors)
        #     result = np.matmul(np.matmul(eigen_vectors, diag), inv_eigen_vectors)
        #     return result
        # if X.shape[0]!=X.shape[1]:
        #     eigen_values, eigen_vectors = np.linalg.eig(self.X_train_)
        #     eigen_values = np.real(eigen_values)
        #     eigen_vectors = np.real(eigen_vectors)
        #     # ------ mod eig values
        #     if len(eigen_values[eigen_values < 0]) > 0:
        #         eigen_values = self.transform_eigen_values(eigen_values)
        #     diag = np.diag(eigen_values)
        #     inv_eigen_vectors = np.linalg.pinv(eigen_vectors)
        #     regularized = np.matmul(np.matmul(eigen_vectors, diag), inv_eigen_vectors)
        #     P = np.matmul(regularized,np.linalg.pinv(self.X_train_))
        #     return np.matmul(X,P)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class ParametersFromDatasetWrapper(BaseEstimator):
    def __init__(self, cv, parameters_getter):
        self.cv = cv
        self.parameters_getter = parameters_getter
        self.parameters = None

    def fit(self, X, y):
        current_parameters = {}
        if not (isinstance(self.cv, sklearn.model_selection.GridSearchCV) or isinstance(self.cv,
                                                                                        sklearn.model_selection.RandomizedSearchCV)):
            raise Exception("cv is not a tuner")
        if isinstance(self.cv, sklearn.model_selection.GridSearchCV):
            current_parameters = self.cv.param_grid
        if isinstance(self.cv, sklearn.model_selection.RandomizedSearchCV):
            current_parameters = self.cv.param_distributions
        if not callable(self.parameters_getter):
            raise Exception("distance measure parameters getter not callable")
        self.parameters = self.parameters_getter(X, y)
        all_params = {**current_parameters, **self.parameters}
        if isinstance(self.cv, sklearn.model_selection.GridSearchCV):
            self.cv.param_grid = all_params
        if isinstance(self.cv, sklearn.model_selection.RandomizedSearchCV):
            self.cv.param_distributions = all_params
        self.cv.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.cv.best_estimator_.predict_proba(X)

    def predict(self, X):
        return self.cv.best_estimator_.predict(X)


class LcssKernel(PairwiseKernel):

    def __init__(self, dim_to_use = 0, delta = 1, epsilon = 1):
        super().__init__()
        self.dim_to_use = dim_to_use
        self.delta = delta
        self.epsilon = epsilon

    def _distance(self, s1, s2):
        s1 = np.reshape(s1, (s1.shape[0], 1))
        s2 = np.reshape(s2, (s2.shape[0], 1))
        return lcss_distance(s1, s2, delta = self.delta, dim_to_use = self.dim_to_use,
                             epsilon = self.epsilon)


def lcss_parameter_space_getter(X, y):
    """
    generate the lcss distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    stdp = sktime.utils.dataset_properties.stdp(X)
    instance_length = sktime.utils.dataset_properties.max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    n_dimensions = 1  # todo use other dimensions
    return {
            'dim_to_use': stats.randint(low = 0, high = n_dimensions),
            'epsilon'   : stats.uniform(0.2 * stdp, stdp - 0.2 * stdp),
            'delta'     : stats.randint(low = 0, high = max_raw_warping_window +
                                                        1)  # scipy stats randint
            # is exclusive on the max value, hence + 1
            }


class DtwKernel(PairwiseKernel):

    def __init__(self, w = 0):
        super().__init__()
        self.w = w

    def _distance(self, s1, s2):
        s1 = np.reshape(s1, (s1.shape[0], 1))
        s2 = np.reshape(s2, (s2.shape[0], 1))
        return dtw_distance(s1, s2, w = self.w)


def dtw_parameter_space_getter(X, y):
    """
    generate the dtw distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
            'w': stats.uniform(0, 0.25)
            }


class MsmKernel(PairwiseKernel):

    def __init__(self, dim_to_use = 0, c = 1):
        super().__init__()
        self.dim_to_use = dim_to_use
        self.c = c

    def _distance(self, s1, s2):
        s1 = np.reshape(s1, (s1.shape[0], 1))
        s2 = np.reshape(s2, (s2.shape[0], 1))
        return msm_distance(s1, s2, c = self.c, dim_to_use = self.dim_to_use)


def msm_parameter_space_getter(X, y):
    """
    generate the msm distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    n_dimensions = 1  # todo use other dimensions
    return {
            'dim_to_use': stats.randint(low = 0, high = n_dimensions),
            'c'         : [0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325,
                           0.03625, 0.04, 0.04375, 0.0475, 0.05125,
                           0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775,
                           0.08125, 0.085, 0.08875, 0.0925, 0.09625,
                           0.1, 0.136, 0.172, 0.208,
                           0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496,
                           0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748,
                           0.784, 0.82, 0.856,
                           0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8,
                           3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68,
                           6.04, 6.4, 6.76, 7.12,
                           7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2,
                           20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46,
                           49.6, 53.2, 56.8, 60.4,
                           64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4,
                           100]
            }


class ErpKernel(PairwiseKernel):

    def __init__(self, dim_to_use = 0, g = 1, band_size = 1):
        super().__init__()
        self.dim_to_use = dim_to_use
        self.g = g
        self.band_size = band_size

    def _distance(self, s1, s2):
        s1 = np.reshape(s1, (s1.shape[0], 1))
        s2 = np.reshape(s2, (s2.shape[0], 1))
        return erp_distance(s1, s2, g = self.g, dim_to_use = self.dim_to_use,
                            epsilon = self.band_size)


def erp_parameter_space_getter(X, y):
    """
    generate the erp distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    stdp = sktime.utils.dataset_properties.stdp(X)
    instance_length = sktime.utils.dataset_properties.max_instance_length(X)  # todo should this use the max instance
    # length for unequal length dataset instances?
    max_raw_warping_window = np.floor((instance_length + 1) / 4)
    n_dimensions = 1  # todo use other dimensions
    return {
            'dim_to_use': stats.randint(low = 0, high = n_dimensions),
            'g'         : stats.uniform(0.2 * stdp, 0.8 * stdp - 0.2 * stdp),
            'band_size' : stats.randint(low = 0, high = max_raw_warping_window + 1)
            # scipy stats randint is exclusive on the max value, hence + 1
            }


class TwedKernel(PairwiseKernel):

    def __init__(self, penalty = 1, stiffness = 1):
        super().__init__()
        self.penalty = penalty
        self.stiffness = stiffness

    def _distance(self, s1, s2):
        s1 = np.reshape(s1, (s1.shape[0], 1))
        s2 = np.reshape(s2, (s2.shape[0], 1))
        return twe_distance(s1, s2, penalty = self.penalty, stiffness = self.stiffness)


def twe_parameter_space_getter(X, y):
    """
    generate the twe distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
            'penalty'  : [0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667,
                          0.077777778, 0.088888889, 0.1],
            'stiffness': [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            }


class WdtwKernel(PairwiseKernel):

    def __init__(self, g = 1):
        super().__init__()
        self.g = g

    def _distance(self, s1, s2):
        s1 = np.reshape(s1, (s1.shape[0], 1))
        s2 = np.reshape(s2, (s2.shape[0], 1))
        return wdtw_distance(s1, s2, g = self.g)


def wdtw_parameter_space_getter(X, y):
    """
    generate the wdtw distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
            'g': stats.uniform(0,
                               1)
            }


def ed_parameter_space_getter(X, y):
    """
    generate the ed distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
            'w': [0]
            }


def full_dtw_parameter_space_getter(X, y):
    """
    generate the full dtw distance measure
    :param X: dataset to derive parameter ranges from
    :return: distance measure and parameter range dictionary
    """
    return {
            'w': [1]
            }


def find_params_using(params_getter):
    def f(X, y):
        return find_params(X, y, params_getter)
    return f


def find_params(X, y, func):
    params = func(X, y)
    result = {}
    for k, v in params.items():
        result["d__" + k] = v
    return result


# if __name__ == "__main__":
#
#     pipe = Pipeline([
#             ('pd_to_np', PandasToNumpy()),
#             ('d', DtwKernel()),
#             ('rbf', RbfKernel()),
#             ('svm', SVC(probability = True, kernel = 'precomputed')),
#             ])
#     cv_params = {
#             'rbf__sigma': stats.expon(scale = .1),
#             'svm__C'    : stats.expon(scale = 100)
#             }
#     model = RandomizedSearchCV(pipe,
#                                cv_params,
#                                cv = 5,
#                                n_jobs = 1,
#                                n_iter = 10,
#                                verbose = 1,
#                                random_state = 0,
#                                )
#     cls = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
#     X_train, y_train = sktime.datasets.load_gunpoint(split = 'TRAIN', return_X_y = True)
#     X_test, y_test = sktime.datasets.load_gunpoint(split = 'TEST', return_X_y = True)
#     cls.fit(X_train, y_train)
#     pred_probas = cls.predict_proba(X_test)
#     print(pred_probas)


class InvertKernel(BaseTransformer):

    def __init__(self):
        super().__init__()

    def transform(self, X, y = None):
        X = X + 1
        ones = np.ones(X.shape)
        X = ones / X
        return X

    def fit(self, X, y = None, **fit_params):
        return self

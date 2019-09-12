import numpy as np
import sklearn.model_selection
import sktime.datasets
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from scipy import spatial

import sktime.utils.dataset_properties
from sktime.utils.data_container import tabularise

from sktime.transformers.base import BaseTransformer
from sktime.classifiers.base import BaseClassifier
import pyximport;

pyximport.install()
from sktime.distances.elastic_cython import (
    wdtw_distance, ddtw_distance, wddtw_distance, msm_distance, lcss_distance,
    erp_distance, dtw_distance,
    twe_distance,
    )
from sktime.classifiers.distance_based.proximity_forest import (
    dtw_distance_measure_getter, wdtw_distance_measure_getter,
    msm_distance_measure_getter, erp_distance_measure_getter, twe_distance_measure_getter,
    euclidean_distance_measure_getter,
    cython_wrapper,
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


# Kernel for triangle similarity
def triangle_similarity_pairs(s1, s2, sigma):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = spatial.distance.cosine(s1, s2)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def triangle_kernel(X, Y, sigma):
    M = cdist(X, Y, metric = triangle_similarity_pairs, sigma = sigma)
    return M


# Kernel for polynomial similarity
def polynomial_similarity_pairs(s1, s2, sigma, degree):
    s1 = unpack_series_row(s1)
    s2 = unpack_series_row(s2)
    coef1 = np.polynomial.polynomial.polyfit(range(0, len(s1)), s1, degree)
    coef2 = np.polynomial.polynomial.polyfit(range(0, len(s2)), s2, degree)
    dist = np.linalg.norm(coef1 - coef2)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def polynomial_kernel(X, Y, sigma, degree):
    M = cdist(X, Y, metric = polynomial_similarity_pairs, sigma = sigma, degree = degree)
    return M


def KL2_similarity_pairs(s1, s2, sigma, degree):
    s1 = unpack_series_row(s1)
    s2 = unpack_series_row(s2)
    coef1, residuals1, rank, singular_values, rcond = np.polyfit(range(len(s1)), s1, full = True, deg = degree)
    coef2, residuals2, rank, singular_values, rcond = np.polyfit(range(len(s2)), s2, full = True, deg = degree)
    means1 = np.polyval(np.polyfit(s1, range(len(s1)), deg = degree), range(len(s1)))
    means2 = np.polyval(np.polyfit(s2, range(len(s2)), deg = degree), range(len(s2)))
    samples1 = np.random.normal(0, np.sqrt(residuals1), len(means1))
    samples1 = samples1 + means1
    samples2 = np.random.normal(0, np.sqrt(residuals2), len(means2))
    samples2 = samples2 + means2
    dist = stats.entropy(samples1, samples2)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def KL2_kernel(X, Y, sigma, degree):
    M = cdist(X, Y, metric = KL2_similarity_pairs, sigma = sigma, degree = degree)
    return M


def Hell_similarity_pairs(s1, s2, sigma, degree):
    s1 = unpack_series_row(s1)
    s2 = unpack_series_row(s2)
    coef1, residuals1, rank, singular_values, rcond = np.polyfit(range(len(s1)), s1, full = True, deg = degree)
    coef2, residuals2, rank, singular_values, rcond = np.polyfit(range(len(s2)), s2, full = True, deg = degree)
    means1 = np.polyval(np.polyfit(s1, range(len(s1)), deg = degree), range(len(s1)))
    means2 = np.polyval(np.polyfit(s2, range(len(s2)), deg = degree), range(len(s2)))
    samples1 = np.random.normal(0, np.sqrt(residuals1), len(means1))
    samples1 = samples1 + means1
    samples2 = np.random.normal(0, np.sqrt(residuals2), len(means2))
    samples2 = samples2 + means2
    _SQRT2 = np.sqrt(2)
    dist = np.sqrt(np.sum((np.sqrt(samples1) - np.sqrt(samples2)) ** 2)) / _SQRT2
    return np.exp(-(dist ** 2) / (sigma ** 2))


def Hell_kernel(X, Y, sigma, degree):
    M = cdist(X, Y, metric = Hell_similarity_pairs, sigma = sigma, degree = degree)
    return M


# Kernels for wdtw distance
def wdtw_pairs(s1, s2, sigma, g):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = wdtw_distance(s1, s2, g)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def wdtw_kernel(X, Y, sigma, g):
    M = cdist(X, Y, metric = wdtw_pairs, sigma = sigma, g = g)
    return M


# Kernels for ddtw distance
def ddtw_pairs(s1, s2, sigma, w):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = ddtw_distance(s1, s2, w)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def ddtw_kernel(X, Y, sigma, w):
    M = cdist(X, Y, metric = ddtw_pairs, sigma = sigma, w = w)
    return M


# Kernels for wddtw distance
def wddtw_pairs(s1, s2, sigma, g):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = wddtw_distance(s1, s2, g)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def wddtw_kernel(X, Y, sigma, g):
    M = cdist(X, Y, metric = wddtw_pairs, sigma = sigma, g = g)
    return M


# Kernels for msm distance
def msm_pairs(s1, s2, sigma, c, dim_to_use):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = msm_distance(s1, s2, c, dim_to_use)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def msm_kernel(X, Y, sigma, c, dim_to_use):
    M = cdist(X, Y, metric = msm_pairs, sigma = sigma, c = c, dim_to_use = dim_to_use)
    return M


# Kernels for lcss distance
def lcss_pairs(s1, s2, sigma, delta, epsilon, dim_to_use):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = lcss_distance(s1, s2, delta, epsilon, dim_to_use)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def lcss_kernel(X, Y, sigma, delta, epsilon, dim_to_use):
    M = cdist(X, Y, metric = lcss_pairs, sigma = sigma, delta = delta, epsilon = epsilon, dim_to_use = dim_to_use)
    return M


# Kernels for erp distance
def erp_pairs(s1, s2, sigma, band_size, g, dim_to_use):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = erp_distance(s1, s2, band_size, g, dim_to_use)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def erp_kernel(X, Y, sigma, band_size, g, dim_to_use):
    M = cdist(X, Y, metric = erp_pairs, sigma = sigma, band_size = band_size, g = g, dim_to_use = dim_to_use)
    return M


# Kernels for twe distance
def twe_pairs(s1, s2, sigma, penalty, stiffness):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = twe_distance(s1, s2, penalty, stiffness)
    return np.exp(-(dist ** 2) / (sigma ** 2))


def twe_kernel(X, Y, sigma, penalty, stiffness):
    M = cdist(X, Y, metric = twe_pairs, sigma = sigma, penalty = penalty, stiffness = stiffness)
    return M


class DTWKernel(BaseEstimator, TransformerMixin):

    # Kernels for dtw distance
    def dtw_pairs(self, s1, s2, w):
        dist = dtw_distance(s1, s2, w)
        return dist

    def dtw_kernel(self, X, Y):
        M = cdist(X, Y, metric = self.dtw_pairs)
        return M

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        kernel = self.dtw_kernel(X, X)
        return kernel


class RBFKernel(BaseEstimator, TransformerMixin):

    def __init__(self,
                 sigma = 1.0):
        self.sigma = sigma

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        kernel = np.exp(-(X ** 2) / (self.sigma ** 2))
        return kernel


class DistanceMeasureKernel(BaseEstimator, TransformerMixin):

    def __init__(self):
        if type(self) == DistanceMeasureKernel:
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

class LcssKernel(DistanceMeasureKernel):

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

class ParameterFromDatasetWrapper(BaseEstimator):
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


def tp(distance_measure):
    """
    wrap a distance measure in cython convertion (to 1 column per dimension format)
    :param distance_measure: distance measure to wrap
    :return: a distance measure which automatically formats data for cython distance measures
    """
    def distance(instance_a, instance_b, **params):
        instance_a = np.transpose(instance_a)
        instance_b = np.transpose(instance_b)
        return distance_measure(instance_a, instance_b, **params)

    return distance

def lcss_distance_measure_getter(X):
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
        'distance_measure': [tp(lcss_distance)],
        'dim_to_use': stats.randint(low=0, high=n_dimensions),
        'epsilon': stats.uniform(0.2 * stdp, stdp - 0.2 * stdp),
        'delta': stats.randint(low=0, high=max_raw_warping_window +
                                           1)  # scipy stats randint
        # is exclusive on the max value, hence + 1
    }

def abc(X, y):
    params = lcss_distance_measure_getter(X)
    del params['distance_measure']
    result = {}
    for k, v in params.items():
        result["d__" + k] = v
    return result

    # result = params
    # distance_measure = params['distance_measure']
    # del params['distance_measure']
    # result = explode(result)
    # wrapped = {'d__distance_measure_parameters': [result], 'd__distance_measure': distance_measure}
    # return wrapped


if __name__ == "__main__":

    pipe = Pipeline([
            ('pd_to_np', PandasToNumpy()),
            ('d', LcssKernel()),
            ('rbf', RBFKernel()),
            ('svm', SVC(probability = True, kernel = 'precomputed')),
            ])
    cv_params = {
            'rbf__sigma': stats.expon(scale = .1),
            'svm__C'    : stats.expon(scale = 100)
            }
    model = RandomizedSearchCV(pipe,
                               cv_params,
                               cv = 5,
                               n_jobs = 1,
                               n_iter = 10,
                               verbose = 1,
                               random_state = 0,
                               )
    wrapped = ParameterFromDatasetWrapper(model, abc)
    X_train, y_train = sktime.datasets.load_gunpoint(split = 'TRAIN', return_X_y = True)
    X_test, y_test = sktime.datasets.load_gunpoint(split = 'TEST', return_X_y = True)
    wrapped.fit(X_train, y_train)
    pred_probas = wrapped.predict_proba(X_test)
    print(pred_probas)





















class Kernel(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sigma = 1.0,
                 distance_measure = None,
                 **distance_measure_parameters,
                 ):
        self.sigma = sigma
        self.distance_measure_parameters = distance_measure_parameters
        self.distance_measure = distance_measure

    def distance(self, s1, s2):
        s1 = unpack_series(s1)
        s2 = unpack_series(s2)
        dist = twe_distance(s1, s2, **self.distance_measure_parameters)
        return np.exp(-(dist ** 2) / (self.sigma ** 2))

    def transform(self, X, y = None):
        kernel = cdist(X, X, metric = self.distance)
        return kernel


class TriKernel(BaseTransformer):
    def __init__(self, sigma = 1.0):
        super(TriKernel, self).__init__()
        self.sigma = sigma
        self.X_train_ = None

    def transform(self, X, y = None):
        return triangle_kernel(X, self.X_train_, sigma = self.sigma)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class PolyKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, degree = 4):
        super(PolyKernel, self).__init__()
        self.sigma = sigma
        self.degree = degree
        self.X_train_ = None

    def transform(self, X, y = None):
        return polynomial_kernel(X, self.X_train_, sigma = self.sigma, degree = self.degree)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class KL2Kernel(BaseTransformer):
    def __init__(self, sigma = 1.0, degree = 4):
        super(KL2Kernel, self).__init__()
        self.sigma = sigma
        self.degree = degree
        self.X_train_ = None

    def transform(self, X, y = None):
        return KL2_kernel(X, self.X_train_, sigma = self.sigma, degree = self.degree)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class HellKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, degree = 4):
        super(HellKernel, self).__init__()
        self.sigma = sigma
        self.degree = degree
        self.X_train_ = None

    def transform(self, X, y = None):
        return Hell_kernel(X, self.X_train_, sigma = self.sigma, degree = self.degree)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class DtwKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, w = 0):
        super(DtwKernel, self).__init__()
        self.sigma = sigma
        self.w = w
        self.X_train_ = None

    def transform(self, X, y = None):
        return dtw_kernel(X, self.X_train_, sigma = self.sigma, w = self.w)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class EigKernel(BaseTransformer):
    def __init__(self, transform_eigen_values = None):
        super().__init__()
        self.transform_eigen_values = transform_eigen_values
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
            eigen_values = self.transform_eigen_values(eigen_values)
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


class NegEigToZero(EigKernel):
    def __init__(self):
        super().__init__(transform_eigen_values = lambda a: np.where(a > 0, a, 0))


class NegEigToAbs(EigKernel):
    def __init__(self):
        super().__init__(transform_eigen_values = lambda a: np.where(a > 0, a, np.abs(a)))


class NegEigToMin(EigKernel):
    def __init__(self):
        super().__init__(transform_eigen_values = self.convert)

    def convert(self, eigen_values):
        min = np.min(eigen_values)
        result = eigen_values - min
        return result


class EdKernel(BaseTransformer):
    def __init__(self, sigma = 1.0):
        super(EdKernel, self).__init__()
        self.sigma = sigma
        self.X_train_ = None

    def transform(self, X, y = None):
        return dtw_kernel(X, self.X_train_, sigma = self.sigma, w = 0)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class FullDtwKernel(BaseTransformer):
    def __init__(self, sigma = 1.0):
        super().__init__()
        self.sigma = sigma
        self.X_train_ = None

    def transform(self, X, y = None):
        return dtw_kernel(X, self.X_train_, sigma = self.sigma, w = 1)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class WdtwKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, g = 0):
        super(WdtwKernel, self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X, y = None):
        return wdtw_kernel(X, self.X_train_, sigma = self.sigma, g = self.g)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


# Class for ddtw distance kernel
class DdtwKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, w = 0):
        super(DdtwKernel, self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X, y = None):
        return ddtw_kernel(X, self.X_train_, sigma = self.sigma, w = self.w)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


# Class for ddtw distance kernel
class FullDdtwKernel(BaseTransformer):
    def __init__(self, sigma = 1.0):
        super(FullDdtwKernel, self).__init__()
        self.sigma = sigma

    def transform(self, X, y = None):
        return ddtw_kernel(X, self.X_train_, sigma = self.sigma)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


# Class for wddtw distance kernel
class WddtwKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, g = 0):
        super(WddtwKernel, self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X, y = None):
        return wddtw_kernel(X, self.X_train_, sigma = self.sigma, g = self.g)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


# Class for msm distance kernel
class MsmKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, c = 0, dim_to_use = 0):
        super(MsmKernel, self).__init__()
        self.sigma = sigma
        self.c = c
        self.dim_to_use = dim_to_use

    def transform(self, X, y = None):
        return msm_kernel(X, self.X_train_, sigma = self.sigma, c = self.c, dim_to_use = self.dim_to_use)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


# Class for lcss distance kernel
class LcssKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, delta = 1, epsilon = 0, dim_to_use = 0):
        super(LcssKernel, self).__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta
        self.dim_to_use = dim_to_use

    def transform(self, X, y = None):
        return lcss_kernel(X, self.X_train_, sigma = self.sigma, delta = self.delta, epsilon = self.epsilon,
                           dim_to_use = self.dim_to_use)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


# Class for erp distance kernel
class ErpKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, band_size = 5, g = 0.5, dim_to_use = 0):
        super(ErpKernel, self).__init__()
        self.sigma = sigma
        self.band_size = band_size
        self.g = g
        self.dim_to_use = dim_to_use

    def transform(self, X, y = None):
        return erp_kernel(X, self.X_train_, sigma = self.sigma, band_size = self.band_size, g = self.g,
                          dim_to_use = self.dim_to_use)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


# Class for twe distance kernel
class TweKernel(BaseTransformer):
    def __init__(self, sigma = 1.0, penalty = 0, stiffness = 1):
        super(TweKernel, self).__init__()
        self.sigma = sigma
        self.penalty = penalty
        self.stiffness = stiffness

    def transform(self, X, y = None):
        return twe_kernel(X, self.X_train_, sigma = self.sigma, penalty = self.penalty, stiffness = self.stiffness)

    def fit(self, X, y = None, **fit_params):
        self.X_train_ = X
        return self


class EigDtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
                 n_iter = 10,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('eig', NegEigToZero()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class TriSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 10,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', TriKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class PolySvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 10,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', PolyKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {
                'dk__degree' : stats.randint(low = 1, high = 10),
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class KL2Svm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 10,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', KL2Kernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {
                'dk__degree' : stats.randint(low = 1, high = 10),
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class HellSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 10,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', HellKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {
                'dk__degree' : stats.randint(low = 1, high = 10),
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FullDtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {
                'dk__w'      : [1],
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FullDdtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('der', DerivativeSlopeTransformer()),
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {
                'dk__w'      : [1],
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DtwKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FullDtwKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {
                'dk__w'      : [1],
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FullDdtwKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('der', DerivativeSlopeTransformer()),
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {
                'dk__w'      : [1],
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class EdKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.euclidean_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', EdKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class WdtwKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.wdtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', WdtwKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LcssKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.lcss_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', LcssKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MsmKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.msm_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', MsmKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ErpKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.erp_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', ErpKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class TweKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.twe_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', TweKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DdtwKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('der', DerivativeSlopeTransformer()),
                ('conv', PandasToNumpy()),
                ('dk', DtwKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class WddtwKnn(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = proximity.wdtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('der', DerivativeSlopeTransformer()),
                ('conv', PandasToNumpy()),
                ('dk', WdtwKernel()),
                ('inv', InvertKernel()),
                ('cls', KNeighborsClassifier(n_neighbors = 1)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'cls__metric': ['precomputed'],
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


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


class WdtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = wdtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', WdtwKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class EdSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = euclidean_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', EdKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class WddtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = wdtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('der', DerivativeSlopeTransformer()),
                ('conv', PandasToNumpy()),
                ('dk', WdtwKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DdtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = dtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('der', DerivativeSlopeTransformer()),
                ('conv', PandasToNumpy()),
                ('dk', DdtwKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MsmSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = msm_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', MsmKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LcssSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = lcss_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', LcssKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ErpSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = erp_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', ErpKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class TweSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = 1,
                 n_iter = 100,
                 label_encoder = None,
                 ):
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.label_encoder = label_encoder
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        self.random_state = check_random_state(self.random_state)
        distance_measure_space = twe_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
                ('conv', PandasToNumpy()),
                ('dk', TweKernel()),
                ('svm', SVC(probability = True)),
                ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
                **cv_params,
                'dk__sigma'  : stats.expon(scale = .1),
                'svm__kernel': ['precomputed'],
                'svm__C'     : stats.expon(scale = 100)
                }
        self.model = RandomizedSearchCV(pipe,
                                        cv_params,
                                        cv = 5,
                                        n_jobs = self.n_jobs,
                                        n_iter = self.n_iter,
                                        verbose = self.verbosity,
                                        random_state = self.random_state,
                                        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

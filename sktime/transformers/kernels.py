import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sktime.classifiers.base import BaseClassifier
from sktime.classifiers.proximity import dtw_distance_measure_getter, wdtw_distance_measure_getter, \
    msm_distance_measure_getter, lcss_distance_measure_getter, erp_distance_measure_getter, twe_distance_measure_getter
from sktime.distances.elastic_cython import wdtw_distance, ddtw_distance, wddtw_distance, msm_distance, lcss_distance, \
    erp_distance, dtw_distance, twe_distance
from sktime.model_selection import GridSearchCV
from sktime.pipeline import Pipeline
from sktime.transformers.pandas_to_numpy import PandasToNumpy
import pandas as pd

def unpack_series(ts):
    if isinstance(ts, pd.Series): ts = ts.values
    ts = np.reshape(ts, (ts.shape[0], 1))
    return ts

#Kernels for dtw distance
def dtw_pairs(s1,s2,sigma,w):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = dtw_distance(s1, s2, w)
    return np.exp(-(dist**2) / (sigma**2))


def dtw_kernel(X,Y,sigma,w):
    M=cdist(X,Y,metric=dtw_pairs,sigma=sigma,w=w)
    return M

#Kernels for wdtw distance
def wdtw_pairs(s1,s2,sigma,g):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = wdtw_distance(s1, s2, g)
    return np.exp(-(dist**2) / (sigma**2))


def wdtw_kernel(X,Y,sigma,g):
    M=cdist(X,Y,metric=wdtw_pairs,sigma=sigma,g=g)
    return M


#Kernels for ddtw distance
def ddtw_pairs(s1,s2,sigma,w):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = ddtw_distance(s1, s2, w)
    return np.exp(-(dist**2) / (sigma**2))


def ddtw_kernel(X,Y,sigma,w):
    M=cdist(X,Y,metric=ddtw_pairs,sigma=sigma,w=w)
    return M


#Kernels for wddtw distance
def wddtw_pairs(s1,s2,sigma,g):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = wddtw_distance(s1, s2, g)
    return np.exp(-(dist**2) / (sigma**2))


def wddtw_kernel(X,Y,sigma,g):
    M=cdist(X,Y,metric=wddtw_pairs,sigma=sigma,g=g)
    return M


#Kernels for msm distance
def msm_pairs(s1,s2,sigma,c):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = msm_distance(s1, s2,c)
    return np.exp(-(dist**2) / (sigma**2))


def msm_kernel(X,Y,sigma,c):
    M=cdist(X,Y,metric=msm_pairs,sigma=sigma,c=c)
    return M


#Kernels for lcss distance
def lcss_pairs(s1,s2,sigma, delta, epsilon):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = lcss_distance(s1, s2,delta, epsilon)
    return np.exp(-(dist**2) / (sigma**2))


def lcss_kernel(X,Y,sigma,delta, epsilon):
    M=cdist(X,Y,metric=lcss_pairs,sigma=sigma, delta=delta, epsilon=epsilon)
    return M


#Kernels for erp distance
def erp_pairs(s1,s2,sigma, band_size, g):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = erp_distance(s1, s2,band_size, g)
    return np.exp(-(dist**2) / (sigma**2))


def erp_kernel(X,Y,sigma, band_size, g):
    M=cdist(X,Y,metric=erp_pairs,sigma=sigma,band_size=band_size, g=g)
    return M

#Kernels for twe distance
def twe_pairs(s1, s2, sigma, penalty, stiffness):
    s1 = unpack_series(s1)
    s2 = unpack_series(s2)
    dist = twe_distance(s1, s2,penalty, stiffness)
    return np.exp(-(dist**2) / (sigma**2))

def twe_kernel(X, Y, sigma, penalty, stiffness):
    M=cdist(X, Y, metric=twe_pairs, sigma=sigma, penalty=penalty, stiffness=stiffness)
    return M


class Kernel(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sigma=1.0,
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
        return np.exp(-(dist**2) / (self.sigma ** 2))


    def transform(self, X, y=None):
        kernel = cdist(X, X, metric=self.distance)
        return kernel

class DtwKernel(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1.0, w=0, label_encoder = None):
        super(DtwKernel, self).__init__()
        self.sigma = sigma
        self.w = w
        self.X_train_ = None

    def transform(self, X, y=None):
        return dtw_kernel(X, self.X_train_, sigma=self.sigma, w=self.w)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


class WdtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(WdtwKernel,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X, y=None):
        return wdtw_kernel(X, self.X_train_, sigma=self.sigma, g=self.g)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self



#Class for ddtw distance kernel
class DdtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(DdtwKernel,self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X, y=None):
        return ddtw_kernel(X, self.X_train_, sigma=self.sigma, w=self.w)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self



#Class for wddtw distance kernel
class WddtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(WddtwKernel,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X, y=None):
        return wddtw_kernel(X, self.X_train_, sigma=self.sigma, g=self.g)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


#Class for msm distance kernel
class MsmKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, c=0):
        super(MsmKernel,self).__init__()
        self.sigma = sigma
        self.c = c

    def transform(self, X, y=None):
        return msm_kernel(X, self.X_train_, sigma=self.sigma, c=self.c)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


#Class for lcss distance kernel
class LcssKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, delta= 1, epsilon=0):
        super(LcssKernel,self).__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

    def transform(self, X, y=None):
        return lcss_kernel(X, self.X_train_, sigma=self.sigma, delta= self.delta, epsilon=self.epsilon)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


#Class for erp distance kernel
class ErpKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, band_size=5,g=0.5):
        super(ErpKernel,self).__init__()
        self.sigma = sigma
        self.band_size = band_size
        self.g = g

    def transform(self, X, y=None):
        return erp_kernel(X, self.X_train_, sigma=self.sigma, band_size= self.band_size, g=self.g)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


#Class for twe distance kernel
class TweKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, penalty=0,stiffness=1):
        super(TweKernel,self).__init__()
        self.sigma = sigma
        self.penalty = penalty
        self.stiffness = stiffness

    def transform(self, X, y=None):
        return twe_kernel(X, self.X_train_, sigma=self.sigma, penalty= self.penalty, stiffness=self.stiffness)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self



class DtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
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
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)









class WdtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
                 n_iter = 5,
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
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)




class DdtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
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
        distance_measure_space = ddtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
            ('conv', PandasToNumpy()),
            ('dk', DdtwKernel()),
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)






class WddtwSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
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
        distance_measure_space = wddtw_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
            ('conv', PandasToNumpy()),
            ('dk', WddtwKernel()),
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)





class MsmSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
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
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)






class LcssSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
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
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)







class ErpSvm(BaseClassifier):

    def __init__(self,
                 random_state = None,
                 verbosity = 0,
                 n_jobs = -1,
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
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)



class TweSvm(BaseClassifier):

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
        distance_measure_space = twe_distance_measure_getter(X)
        del distance_measure_space['distance_measure']
        pipe = Pipeline([
            ('conv', PandasToNumpy()),
            ('dk', TweKernel()),
            ('svm', SVC(probability=True)),
        ])
        cv_params = {}
        for k, v in distance_measure_space.items():
            cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'dk__sigma': stats.expon(scale=.1),
            'svm__kernel': ['precomputed'],
            'svm__C': stats.expon(scale=100)
        }
        self.model = RandomizedSearchCV(pipe,
                                    cv_params,
                                    cv=5,
                                    n_jobs=self.n_jobs,
                                    n_iter=self.n_iter,
                                    verbose=self.verbosity,
                                    random_state=self.random_state,
                                    )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


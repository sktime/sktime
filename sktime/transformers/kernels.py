import numpy as np
from  scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from sktime.classifiers.base import BaseClassifier
from sktime.classifiers.proximity import dtw_distance_measure_getter
from sklearn.metrics import accuracy_score
from sktime.distances.elastic_cython import wdtw_distance, ddtw_distance, wddtw_distance, msm_distance, lcss_distance, \
    erp_distance, dtw_distance, twe_distance
from sktime.model_selection import GridSearchCV
from sktime.pipeline import Pipeline
from sktime.transformers.pandas_to_numpy import PandasToNumpy
from sktime.utils.load_data import load_ts
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


class DtwKernel(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(DtwKernel, self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X, y=None):
        return dtw_kernel(X, X, sigma=self.sigma, w=self.w)


class WdtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(WdtwKernel,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X):
        return wdtw_kernel(X, X, sigma=self.sigma, g=self.g)

#Class for ddtw distance kernel
class DdtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(DdtwKernel,self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X):
        return ddtw_kernel(X, X, sigma=self.sigma, w=self.w)

#Class for wddtw distance kernel
class WddtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(WddtwKernel,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X):
        return wddtw_kernel(X, X, sigma=self.sigma, g=self.g)


#Class for msm distance kernel
class MsmKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, c=0):
        super(MsmKernel,self).__init__()
        self.sigma = sigma
        self.c = c

    def transform(self, X):
        return msm_kernel(X, X, sigma=self.sigma, c=self.c)

#Class for lcss distance kernel
class LcssKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, delta= 1, epsilon=0):
        super(LcssKernel,self).__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

    def transform(self, X):
        return lcss_kernel(X, X, sigma=self.sigma, delta= self.delta, epsilon=self.epsilon)


#Class for erp distance kernel
class ErpKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, band_size=5,g=0.5):
        super(ErpKernel,self).__init__()
        self.sigma = sigma
        self.band_size = band_size
        self.g = g

    def transform(self, X):
        return erp_kernel(X, X, sigma=self.sigma, band_size= self.band_size, g=self.g)

#Class for twe distance kernel
class TweKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, penalty=0,stiffness=1):
        super(TweKernel,self).__init__()
        self.sigma = sigma
        self.penalty = penalty
        self.stiffness = stiffness

    def transform(self, X):
        return twe_kernel(X, X, sigma=self.sigma, penalty= self.penalty, stiffness=self.stiffness)

class DistanceMeasureSvm(BaseClassifier):

    def __init__(self,
                 distance_measure_getter = None,
                 random_state = None,
                 ):
        self.distance_measure_getter = distance_measure_getter
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.random_state = check_random_state(self.random_state)
        distance_measure = self.distance_measure_getter(X)
        pipe = Pipeline([
            ('conv', PandasToNumpy()),
            ('dk', DtwKernel()),
            ('svm', SVC()),
        ])
        cv_params = {}
        for k, v in distance_measure.items(): cv_params['dk__' + k] = v
        cv_params = {
            **cv_params,
            'svm__kernel': ['precomputed'],
            'svm__C': [1]
        }
        self.model = RandomizedSearchCV(pipe,
                                   cv_params,
                                   scoring=accuracy_score,
                                   n_jobs=1,
                                   verbose=0,
                                   random_state=self.random_state,
                                   )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)
def DtwSvm():

    # dtw kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', DtwKernel()),
        ('svm', SVC()),
    ])

    def predict(self, X):
        return self.model.predict(X)


    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model

# #wdtw kernel parameter estimation
#     pipe = Pipeline([
#         ('dk', distancekernel_wdtw()),
#         ('svm', SVC()),
#     ])
#
#     # cv_params = dict([
#     #     ('dk__sigma', [0.01,0.1,1,10,100]),
#     #     ('dk__g', [0.01,0.1,0,10,100]),
#     #     ('svm__kernel', ['precomputed']),
#     #     ('svm__C', [0.01,0.1,1,10,100])
#     # ])
#
#     # To test if it works
#     cv_params = dict([
#         ('dk__sigma', [0.01]),
#         ('dk__g', [0.01]),
#         ('svm__kernel', ['precomputed']),
#         ('svm__C', [0.01])
#     ])
#
#     model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_test_wdtw = accuracy_score(y_test, y_pred)
#     print("Test accuracy wdtw: {}".format(acc_test_wdtw))
#     print("Best params:")
#     print(model.best_params_)


def WdtwSvm():
#wdtw kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', WdtwKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__g', [0.01,0.1,0,10,100]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])

    # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.01]),
        ('dk__g', [0.01]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])

    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model



def DdtwSvm():
#ddtw kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', DdtwKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__w', [-1,0.01,0.1,0.2,0.4]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])

    # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.01]),
        ('dk__w', [-1]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])


    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model


def WddtwSmv():
#wddtw kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', WdtwKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__g', [0.01,0.1,0,10,100]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])

    # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.01]),
        ('dk__g', [0.01]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])



    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model




def MsmSvm():
#msm kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', MsmKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__c', [0.01, 0.1, 1, 10, 100]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])

    # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.01]),
        ('dk__c', [0.01]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])



    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model



def LcssSvm():

    # lcss kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', LcssKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__delta', [0.1,1,10,100,500]),
    #     ('dk__epsilon', [0.01,0.1,0.2,0.4]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])

    # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.01]),
        ('dk__delta', [0.1]),
        ('dk__epsilon', [0.01]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])

    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model




def ErpSvm():
#erp kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', ErpKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__band_size', [0.001,0.01,0.1,0.2,0.4]),
    #     ('dk__g', [0.01,0.1,0,10,100]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])


    # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.01]),
        ('dk__band_size', [0.01]),
        ('dk__g', [0.01]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])


    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model





def TweSvm():
#twe kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', TweKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__penalty', [0.001,0.01,0.1,0.2,0.4]),
    #     ('dk__stiffness', [0.01,0.1,0,10,100]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])


    # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.1]),
        ('dk__penalty', [1]),
        ('dk__stiffness', [0.01]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])


    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    return model





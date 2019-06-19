import numpy as np
from  scipy.spatial.distance import cdist



#Kernels for dtw distance
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sktime.distances.elastic_cython import wdtw_distance, ddtw_distance, wddtw_distance, msm_distance, lcss_distance, \
    erp_distance, dtw_distance

from sktime.transformers.base import BaseTransformer

from sktime.model_selection import GridSearchCV
from sktime.pipeline import Pipeline

from sktime.classifiers.base import BaseClassifier
from sktime.utils.load_data import load_ts
from sktime.utils.transformations import tabularise
import pandas as pd



#Kernels for dtw distance
def dtw_pairs(s1,s2,sigma,w):
    if isinstance(s1, pd.Series): s1 = s1.values
    if isinstance(s2, pd.Series): s2 = s2.values
    s1 = np.reshape(s1, (s1.shape[0], 1))
    s2 = np.reshape(s2, (s2.shape[0], 1))
    dist = dtw_distance(s1, s2, w)
    return np.exp(-(dist**2) / (sigma**2))


def dtw_kernel(X,Y,sigma,w):
    M=cdist(X,Y,metric=dtw_pairs,sigma=sigma,w=w)
    return M

#Kernels for wdtw distance
def wdtw_pairs(s1,s2,sigma,g):
    dist = wdtw_distance(s1, s2, g)
    return np.exp(-(dist**2) / (sigma**2))


def wdtw_kernel(X,Y,sigma,g):
    M=cdist(X,Y,metric=wdtw_pairs,sigma=sigma,g=g)
    return M


#Kernels for ddtw distance
def ddtw_pairs(s1,s2,sigma,w):
    dist = ddtw_distance(s1, s2, w)
    return np.exp(-(dist**2) / (sigma**2))


def ddtw_kernel(X,Y,sigma,w):
    M=cdist(X,Y,metric=ddtw_pairs,sigma=sigma,w=w)
    return M


#Kernels for wddtw distance
def wddtw_pairs(s1,s2,sigma,g):
    dist = wddtw_distance(s1, s2, g)
    return np.exp(-(dist**2) / (sigma**2))


def wddtw_kernel(X,Y,sigma,g):
    M=cdist(X,Y,metric=wddtw_pairs,sigma=sigma,g=g)
    return M


#Kernels for msm distance
def msm_pairs(s1,s2,sigma,c):
    dist = msm_distance(s1, s2,c)
    return np.exp(-(dist**2) / (sigma**2))


def msm_kernel(X,Y,sigma,c):
    M=cdist(X,Y,metric=msm_pairs,sigma=sigma,c=c)
    return M


#Kernels for lcss distance
def lcss_pairs(s1,s2,sigma, delta, epsilon):
    dist = lcss_distance(s1, s2,delta, epsilon)
    return np.exp(-(dist**2) / (sigma**2))


def lcss_kernel(X,Y,sigma,delta, epsilon):
    M=cdist(X,Y,metric=lcss_pairs,sigma=sigma, delta=delta, epsilon=epsilon)
    return M


#Kernels for erp distance
def erp_pairs(s1,s2,sigma, band_size, g):
    dist = erp_distance(s1, s2,band_size, g)
    return np.exp(-(dist**2) / (sigma**2))


def erp_kernel(X,Y,sigma, band_size, g):
    M=cdist(X,Y,metric=erp_pairs,sigma=sigma,band_size=band_size, g=g)
    return M


def distance_kernel(distance_measure, **kwargs):
    sigma = kwargs['sigma']


    def distance(a, b, **kwargs):
        dist = distance_measure(a, b, **kwargs)
        return np.exp(-(dist**2) / sigma**2)


    def build_kernel(X, Y):
        kernel = cdist(X, Y, metric=distance)
        return kernel
#Kernels for twe distance
def twe_pairs(s1, s2, sigma, penalty, stiffness):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = twe_distance(s1, s2,penalty, stiffness)
    return np.exp(-(dist**2) / (sigma**2))

    return build_kernel

def twe_kernel(X, Y, sigma, penalty, stiffness):
    M=cdist(X, Y, metric=twe_pairs, sigma=sigma, penalty=penalty, stiffness=stiffness)
    return M

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

#Class for dtw distance kernel

from sktime.transformers.pandas_to_numpy import PandasToNumpy
from sktime.utils.load_data import load_ts


class DtwKernel(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(DtwKernel, self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X, y=None):
        return dtw_kernel(X, self.X_train_, sigma=self.sigma, w=self.w)

    def fit(self, X, y=None, **fit_params):
        return self




class WdtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(WdtwKernel,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X):
        return wdtw_kernel(X, self.X_train_, sigma=self.sigma, g=self.g)

    def fit(self, X, y=None, **fit_params):
        return self





#Class for ddtw distance kernel
class DdtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(DdtwKernel,self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X):
        return ddtw_kernel(X, self.X_train_, sigma=self.sigma, w=self.w)

    def fit(self, X, y=None, **fit_params):
        return self






#Class for wddtw distance kernel
class WddtwKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(WddtwKernel,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X):
        return wddtw_kernel(X, self.X_train_, sigma=self.sigma, g=self.g)

    def fit(self, X, y=None, **fit_params):
        return self



#Class for msm distance kernel
class MsmKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, c=0):
        super(MsmKernel,self).__init__()
        self.sigma = sigma
        self.c = c

    def transform(self, X):
        return msm_kernel(X, self.X_train_, sigma=self.sigma, c=self.c)

    def fit(self, X, y=None, **fit_params):
        return self




#Class for lcss distance kernel
class LcssKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, delta= 1, epsilon=0):
        super(LcssKernel,self).__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

    def transform(self, X):
        return lcss_kernel(X, self.X_train_, sigma=self.sigma, delta= self.delta, epsilon=self.epsilon)

    def fit(self, X, y=None, **fit_params):
        return self



#Class for erp distance kernel
class ErpKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, band_size=5,g=0.5):
        super(ErpKernel,self).__init__()
        self.sigma = sigma
        self.band_size = band_size
        self.g = g

    def transform(self, X):
        return erp_kernel(X, self.X_train_, sigma=self.sigma, band_size= self.band_size, g=self.g)

    def fit(self, X, y=None, **fit_params):
        return self




#Class for twe distance kernel
class TweKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, penalty=0,stiffness=1):
        super(TweKernel,self).__init__()
        self.sigma = sigma
        self.penalty = penalty
        self.stiffness = stiffness

    def transform(self, X):
        return twe_kernel(X, self.X_train_, sigma=self.sigma, penalty= self.penalty, stiffness=self.stiffness)

    def fit(self, X, y=None, **fit_params):
        return self





if __name__ == "__main__":
    datasets_dir_path = '/scratch/data/Univariate2018'
    dataset_name = 'GunPoint'
    format = '.ts'
    X_train, y_train = load_ts(datasets_dir_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN' + format)
    X_test, y_test = load_ts(datasets_dir_path + '/' + dataset_name + '/' + dataset_name + '_TEST' + format)



#dtw kernel parameter estimation
    pipe = Pipeline([
        ('conv', PandasToNumpy()),
        ('dk', DtwKernel()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__w', [-1,0.01,0.1,0.2,0.4]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])

    # # To test if it works
    cv_params = dict([
        ('dk__sigma', [0.1]),
        ('dk__w', [1]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [1])
    ])

    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_dtw = accuracy_score(y_test, y_pred)
    print("Test accuracy dtw: {}".format(acc_test_dtw))
    print("Best params:")
    print(model.best_params_)





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





# #ddtw kernel parameter estimation
#     pipe = Pipeline([
#         ('dk', distancekernel_ddtw()),
#         ('svm', SVC()),
#     ])
#
#     # cv_params = dict([
#     #     ('dk__sigma', [0.01,0.1,1,10,100]),
#     #     ('dk__w', [-1,0.01,0.1,0.2,0.4]),
#     #     ('svm__kernel', ['precomputed']),
#     #     ('svm__C', [0.01,0.1,1,10,100])
#     # ])
#
#     # To test if it works
#     cv_params = dict([
#         ('dk__sigma', [0.01]),
#         ('dk__w', [-1]),
#         ('svm__kernel', ['precomputed']),
#         ('svm__C', [0.01])
#     ])
#
#
#     model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_test_ddtw = accuracy_score(y_test, y_pred)
#     print("Test accuracy ddtw: {}".format(acc_test_ddtw))
#     print("Best params:")
#     print(model.best_params_)
#
#

#
# #wddtw kernel parameter estimation
#     pipe = Pipeline([
#         ('dk', distancekernel_wddtw()),
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
#
#
#     model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_test_wddtw = accuracy_score(y_test, y_pred)
#     print("Test accuracy wddtw: {}".format(acc_test_wddtw))
#     print("Best params:")
#     print(model.best_params_)
#
#
#
#
# #msm kernel parameter estimation
#     pipe = Pipeline([
#         ('dk', distancekernel_msm()),
#         ('svm', SVC()),
#     ])
#
#     # cv_params = dict([
#     #     ('dk__sigma', [0.01,0.1,1,10,100]),
#     #     ('dk__c', [0.01, 0.1, 1, 10, 100]),
#     #     ('svm__kernel', ['precomputed']),
#     #     ('svm__C', [0.01,0.1,1,10,100])
#     # ])
#
#     # To test if it works
#     cv_params = dict([
#         ('dk__sigma', [0.01]),
#         ('dk__c', [0.01]),
#         ('svm__kernel', ['precomputed']),
#         ('svm__C', [0.01])
#     ])
#
#
#
#     model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_test_msm = accuracy_score(y_test, y_pred)
#     print("Test accuracy msm: {}".format(acc_test_msm))
#     print("Best params:")
#     print(model.best_params_)
#
#
#
#
#
#
# #lcss kernel parameter estimation
#     pipe = Pipeline([
#         ('dk', distancekernel_lcss()),
#         ('svm', SVC()),
#     ])
#
#     # cv_params = dict([
#     #     ('dk__sigma', [0.01,0.1,1,10,100]),
#     #     ('dk__delta', [0.1,1,10,100,500]),
#     #     ('dk__epsilon', [0.01,0.1,0.2,0.4]),
#     #     ('svm__kernel', ['precomputed']),
#     #     ('svm__C', [0.01,0.1,1,10,100])
#     # ])
#
#
#     #To test if it works
#     cv_params = dict([
#         ('dk__sigma', [0.01]),
#         ('dk__delta', [0.1]),
#         ('dk__epsilon', [0.01]),
#         ('svm__kernel', ['precomputed']),
#         ('svm__C', [0.01])
#     ])
#
#
#
#
#     model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_test_lcss = accuracy_score(y_test, y_pred)
#     print("Test accuracy lcss: {}".format(acc_test_lcss))
#     print("Best params:")
#     print(model.best_params_)
#
#
#
#
#
# #erp kernel parameter estimation
#     pipe = Pipeline([
#         ('dk', distancekernel_erp()),
#         ('svm', SVC()),
#     ])
#
#     # cv_params = dict([
#     #     ('dk__sigma', [0.01,0.1,1,10,100]),
#     #     ('dk__band_size', [0.001,0.01,0.1,0.2,0.4]),
#     #     ('dk__g', [0.01,0.1,0,10,100]),
#     #     ('svm__kernel', ['precomputed']),
#     #     ('svm__C', [0.01,0.1,1,10,100])
#     # ])
#
#
#     # To test if it works
#     cv_params = dict([
#         ('dk__sigma', [0.01]),
#         ('dk__band_size', [0.01]),
#         ('dk__g', [0.01]),
#         ('svm__kernel', ['precomputed']),
#         ('svm__C', [0.01])
#     ])
#
#
#     model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_test_erp = accuracy_score(y_test, y_pred)
#     print("Test accuracy erp: {}".format(acc_test_erp))
#     print("Best params:")
#     print(model.best_params_)
#
#
#
#
#
#
#
#
#
#
# #twe kernel parameter estimation
#     pipe = Pipeline([
#         ('dk', distancekernel_twe()),
#         ('svm', SVC()),
#     ])
#
#     # cv_params = dict([
#     #     ('dk__sigma', [0.01,0.1,1,10,100]),
#     #     ('dk__penalty', [0.001,0.01,0.1,0.2,0.4]),
#     #     ('dk__stiffness', [0.01,0.1,0,10,100]),
#     #     ('svm__kernel', ['precomputed']),
#     #     ('svm__C', [0.01,0.1,1,10,100])
#     # ])
#
#
#     # To test if it works
#     cv_params = dict([
#         ('dk__sigma', [0.1]),
#         ('dk__penalty', [1]),
#         ('dk__stiffness', [0.01]),
#         ('svm__kernel', ['precomputed']),
#         ('svm__C', [0.01])
#     ])
#
#
#     model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc_test_twe = accuracy_score(y_test, y_pred)
#     print("Test accuracy twe: {}".format(acc_test_twe))
#     print("Best params:")
#     print(model.best_params_)
#



np.savetxt('GunPoint.out', [acc_test_dtw],fmt='%2f')


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

class PandaUnpacker(BaseClassifier):

    def __init__(self,
                 cls = None,
                 unpack_train = True,
                 unpack_test = True):
        self.cls = cls
        self.unpack_train = unpack_train
        self.unpack_test = unpack_test

    def fit(self, X, y):
        if self.unpack_train and isinstance(X, pd.DataFrame): X = tabularise(X, return_array=True)
        self.cls.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.unpack_test and isinstance(X, pd.DataFrame): X = tabularise(X, return_array=True)
        return self.cls.predict_proba(X)

    def predict(self, X):
        if self.unpack_test and isinstance(X, pd.DataFrame): X = tabularise(X, return_array=True)
        return self.cls.predict(X)


class DtwKernel(BaseTransformer):

    def __init__(self,
                 sigma = 0.01,
                 w = -1,
                 dim = 0,
                 ):
        self.sigma = sigma
        self.w = w
        self.dim = dim

    def transform(self, X, Z): #y=None):
        if isinstance(X, pd.DataFrame): X = X.to_numpy()
        if isinstance(Z, pd.DataFrame): Z = Z.to_numpy()
        M = cdist(X, Z, metric=self.dtw_pairs)
        return M

    def fit(self, X, y=None
            , **fit_params
            ):
        self.sigma = fit_params.get('sigma', self.sigma)
        self.w = fit_params.get('w', self.w)
        self.dim = fit_params.get('dim', self.dim)
        return self

    def dtw_pairs(self, s1,s2):
        if isinstance(s1, pd.Series): s1 = s1.values[self.dim]
        if isinstance(s2, pd.Series): s2 = s2.values[self.dim]
        s1 = np.reshape(s1, (s1.shape[0], 1))
        s2 = np.reshape(s2, (s2.shape[0], 1))
        dist = dtw_distance(s1, s2, self.w)
        return np.exp(-(dist**2) / (self.sigma**2))


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
def GDS_twe_pairs(s1,s2,sigma, penalty, stiffness):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    dist = twe_distance(s1, s2,penalty, stiffness)
    return np.exp(-(dist**2) / (sigma**2))

    return build_kernel

def GDS_twe_matrix(X,Y,sigma, penalty, stiffness):
    M=cdist(X,Y,metric=GDS_twe_pairs,sigma=sigma,penalty=penalty, stiffness=stiffness)
    return M

def dtw_svm(**gs_params):
    model = PandaUnpacker(
            SVC(probability=True, kernel=DtwKernel(dim=0, sigma = 0.1, w = -1).transform)
            , unpack_train=True, unpack_test=True)

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__w', [-1,0.01,0.1,0.2,0.4]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])

    cv_params = [{
        'kernel': [DtwKernel().transform,
        {
            'sigma': [0.01],
            'w': [-1],
            'dim': [0],
        }],
        'cls__kernel': ['precomputed'],
        'cls__C': [0.01],
        'cls__probability': [True],
    }]

    cv_params= [{'kernel': [DtwKernel().transform], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    gs = GridSearchCV(model, cv_params, **gs_params)
    return gs

class DtwSvm(BaseClassifier):

    def __init__(self,
                 w = -1,
                 sigma = 0.01,
                 dim = 0,
                 random_state = None
                 ):
        self.w = w
        self.random_state = random_state
        self.dim = dim
        self.sigma = sigma
        self.cls_ = None

    def fit(self, X, y):
        self.random_state = check_random_state(self.random_state)
        self.cls_ = PandaUnpacker(SVC(
            kernel = DtwKernel(w=self.w,
                              dim=self.dim,
                              sigma=self.sigma).transform,
                       probability=True))
        self.cls_.fit(X, y)
        return self

    def predict(self, X):
        return self.cls_.predict(X)

    def predict_proba(self, X):
        return self.cls_.predict_proba(X)








def distance_matrix(distance_measure, **kwargs):
    sigma = kwargs['sigma']


if __name__ == '__main__':
    datasets_dir_path = '/scratch/data/Univariate2018'
    dataset_name = 'GunPoint'
    format = '.ts'
    trainX, trainY = load_ts(datasets_dir_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN' + format)
    testX, testY = load_ts(datasets_dir_path + '/' + dataset_name + '/' + dataset_name + '_TEST' + format)
    kernel_transformer = DtwKernel()
    # cls = \
    #     PandaUnpacker(
    #         SVC(probability=True, kernel=DtwKernel(dim=0, sigma = 0.1, w = -1).transform)
    #         , train=True, test=True)
    trainX = tabularise(trainX, return_array=True)
    testX = tabularise(testX, return_array=True)
    cls = SVC(
        kernel=DtwKernel().transform,
        probability=True,
        random_state=1,
    )
    # cls = DtwSvm(
    #     # probability=True,
    #     random_state=1,
    # )
    params = [{
        # 'sigma': [0.01],
        # 'w': [-1],
        # 'dim': [0],
        'C': [0.1, 0.5, 1]
    }]
    cls = PandaUnpacker(GridSearchCV(cls, params, cv=3, verbose=1, n_jobs=1))
    cls.fit(trainX, trainY)
    class_labels = np.unique(np.concatenate([np.array(testY), np.array(trainY)]))
    # cls = dtw_svm(cv=2, verbose=1, n_jobs=1)
    # cls.fit(trainX, trainY)
    predictions = cls.predict_proba(testX)
    acc = accuracy_score(testY, class_labels[np.argmax(predictions, axis = 1)])
    print(acc)
    # debug=True
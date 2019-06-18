from sktime.kernels.base import GDS_wdtw_matrix
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np


#Class for wdtw distance kernel
class distancekernel_wdtw(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(distancekernel_wdtw,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X):
        return GDS_wdtw_matrix(X, self.X_train_, sigma=self.sigma, g=self.g)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self






if __name__ == "__main__":
    X_train, y_train, X_test, y_test = UCR_UEA_datasets(use_cache=True).load_dataset("GunPoint")
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
    y_train = np.ravel(y_train)






#wdtw kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_wdtw()),
        ('svm', SVC()),
    ])

    cv_params = dict([
        ('dk__sigma', [0.01,0.1,1,10,100]),
        ('dk__g', [0.01,0.1,0,10,100]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01,0.1,1,10,100])
    ])



    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_wdtw = accuracy_score(y_test, y_pred)
    print("Test accuracy wdtw: {}".format(acc_test_wdtw))
    print("Best params:")
    print(model.best_params_)




np.savetxt('GunPoint_wdtw.out', [acc_test_wdtw],fmt='%2f')


from sktime.kernels.base import dtw_kernel
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

#Class for dtw distance kernel
class distancekernel_dtw(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(distancekernel_dtw,self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X):
        return dtw_kernel(X, self.X_train_, sigma=self.sigma, w=self.w)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self



if __name__ == "__main__":
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("GunPoint")
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
    y_train = np.ravel(y_train)



#dtw kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_dtw()),
        ('svm', SVC()),
    ])

    cv_params = dict([
        ('dk__sigma', [0.01,0.1,1,10,100]),
        ('dk__w', [-1,0.01,0.1,0.2,0.4]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01,0.1,1,10,100])
    ])



    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_dtw = accuracy_score(y_test, y_pred)
    print("Test accuracy dtw: {}".format(acc_test_dtw))
    print("Best params:")
    print(model.best_params_)





np.savetxt('GunPoint_dtw.out', [acc_test_dtw],fmt='%2f')


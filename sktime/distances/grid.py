from sktime.kernels.base import GDS_dtw_matrix, GDS_wdtw_matrix,GDS_ddtw_matrix,GDS_wddtw_matrix,GDS_msm_matrix,GDS_lcss_matrix,GDS_erp_matrix
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline



#Class for dtw distance kernel
class distancekernel_dtw(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(distancekernel_dtw,self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X):
        return GDS_dtw_matrix(X, self.X_train_, sigma=self.sigma, w=self.w)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self



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





#Class for ddtw distance kernel
class distancekernel_ddtw(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, w=0):
        super(distancekernel_ddtw,self).__init__()
        self.sigma = sigma
        self.w = w

    def transform(self, X):
        return GDS_ddtw_matrix(X, self.X_train_, sigma=self.sigma, w=self.w)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self








#Class for wddtw distance kernel
class distancekernel_wddtw(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, g=0):
        super(distancekernel_wddtw,self).__init__()
        self.sigma = sigma
        self.g = g

    def transform(self, X):
        return GDS_wddtw_matrix(X, self.X_train_, sigma=self.sigma, g=self.g)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self



#Class for wddtw distance kernel
class distancekernel_msm(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, c=0):
        super(distancekernel_msm,self).__init__()
        self.sigma = sigma
        self.c = c

    def transform(self, X):
        return GDS_msm_matrix(X, self.X_train_, sigma=self.sigma, c=self.c)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self




#Class for lcss distance kernel
class distancekernel_lcss(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, delta= 1, epsilon=0):
        super(distancekernel_lcss,self).__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

    def transform(self, X):
        return GDS_lcss_matrix(X, self.X_train_, sigma=self.sigma, delta= self.delta, epsilon=self.epsilon)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self



#Class for erp distance kernel
class distancekernel_erp(BaseEstimator,TransformerMixin):
    def __init__(self, sigma=1.0, band_size=5,g=0.5):
        super(distancekernel_erp,self).__init__()
        self.sigma = sigma
        self.band_size = band_size
        self.g = g

    def transform(self, X):
        return GDS_erp_matrix(X, self.X_train_, sigma=self.sigma, band_size= self.band_size, g=self.g)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self














if __name__ == "__main__":
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("GunPoint")
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
    y_train = y_train.reshape(-1,1)







#dtw kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_dtw()),
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_dtw = accuracy_score(y_test, y_pred)
    print("Test accuracy dtw: {}".format(acc_test_dtw))
    print("Best params:")
    print(model.best_params_)





#wdtw kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_wdtw()),
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_wdtw = accuracy_score(y_test, y_pred)
    print("Test accuracy wdtw: {}".format(acc_test_wdtw))
    print("Best params:")
    print(model.best_params_)





#ddtw kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_ddtw()),
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_ddtw = accuracy_score(y_test, y_pred)
    print("Test accuracy ddtw: {}".format(acc_test_ddtw))
    print("Best params:")
    print(model.best_params_)




#wddtw kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_wddtw()),
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_wddtw = accuracy_score(y_test, y_pred)
    print("Test accuracy wddtw: {}".format(acc_test_wddtw))
    print("Best params:")
    print(model.best_params_)




#msm kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_msm()),
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_msm = accuracy_score(y_test, y_pred)
    print("Test accuracy msm: {}".format(acc_test_msm))
    print("Best params:")
    print(model.best_params_)






#lcss kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_lcss()),
        ('svm', SVC()),
    ])

    # cv_params = dict([
    #     ('dk__sigma', [0.01,0.1,1,10,100]),
    #     ('dk__delta', [0.1,1,10,100,500]),
    #     ('dk__epsilon', [0.01,0.1,0.2,0.4]),
    #     ('svm__kernel', ['precomputed']),
    #     ('svm__C', [0.01,0.1,1,10,100])
    # ])


    #To test if it works
    cv_params = dict([
        ('dk__sigma', [0.01]),
        ('dk__delta', [0.1]),
        ('dk__epsilon', [0.01]),
        ('svm__kernel', ['precomputed']),
        ('svm__C', [0.01])
    ])




    model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_lcss = accuracy_score(y_test, y_pred)
    print("Test accuracy lcss: {}".format(acc_test_lcss))
    print("Best params:")
    print(model.best_params_)





#erp kernel parameter estimation
    pipe = Pipeline([
        ('dk', distancekernel_erp()),
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_test_erp = accuracy_score(y_test, y_pred)
    print("Test accuracy erp: {}".format(acc_test_erp))
    print("Best params:")
    print(model.best_params_)

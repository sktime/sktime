#if __name__ == "__main__":
#    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape((1, 5))
#    b = np.array([3.0,4.0,3.0,6.0,1.0]).reshape((1, 5))
#    dist = GDS(a, b, w=5)
#   print(dist)


# clf1 = KNeighborsClassifier(n_neighbors=1, metric=GDS_pairs)
# clf1.fit(X_train2,y_train)
# print("Correct classification rate:", clf1.score(X_test2, y_test))




# class distancekernel2(BaseEstimator,TransformerMixin):
#     def __init__(self, **kwargs):
#         super(distancekernel2,self).__init__()
#         self.kwargs = kwargs
#
#
#     def transform(self, X):
#         return GDS_matrix2(X, self.X_train_, kwargs=self.kwargs)
#
#     def fit(self, X, y=None, **fit_params):
#         self.X_train_ = X
#         return self

from astropy.io import ascii
ascii.write([acc_test_dtw, acc_test_wdtw], 'GunPoint.dat', names=['dtw', 'wdtw'])

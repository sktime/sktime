import numpy as np

from sklearn import svm
from sktime.distances.kernel_functions import GDS_matrix
from tslearn.datasets import UCR_UEA_datasets


X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("FaceFour")
X_train2= X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test2= X_test.reshape(X_test.shape[0],X_test.shape[1])

clf = svm.SVC(kernel=GDS_matrix)
clf.fit(X_train2, y_train)
print(clf.predict(X_test2))
print(y_test)
print("Correct classification rate:", clf.score(X_test2, y_test))
from sklearn import svm
from sktime.kernels.base import GDS_matrix
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("GunPoint")
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])

    clf = svm.SVC(kernel=GDS_matrix)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, predictions, normalize = True)

    print('Accuracy: ' + str(acc))



from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_unit_test
from sklearn.metrics import accuracy_score


X_train, y_train = load_unit_test(split="train", return_X_y=True)
X_test, y_test = load_unit_test(split="test", return_X_y=True)


clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance="dtw")


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy with KNN (DTW): {accuracy * 100:.2f}%")

sample_pred = clf.predict(X_test.iloc[[0]])
print(f"Predicted class for the first test sample: {sample_pred[0]}")

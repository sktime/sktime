
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.datasets import load_italy_power_demand
from sklearn.model_selection import train_test_split


def test_stc_with_pd():
    X, y = load_italy_power_demand(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    c = ShapeletTransformClassifier()
    c.fit(X_train, y_train)


def test_stc_with_np():
    X, y = load_italy_power_demand(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    c = ShapeletTransformClassifier()
    c.fit(X_train, y_train.to_numpy())

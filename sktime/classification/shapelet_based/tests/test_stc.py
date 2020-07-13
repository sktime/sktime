
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sktime.classification.shapelet_based import ShapeletTransformClassifier
# from sktime.datasets import load_italy_power_demand


# def test_stc_with_pd():
#     random_state = np.random.RandomState(1)

#     X, y = load_italy_power_demand(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, random_state=random_state)

#     c = ShapeletTransformClassifier(1.0, random_state=random_state)
#     c.fit(X_train, y_train)

#     preds = c.predict(X_test)

#     np.testing.assert_equal((preds == y_test).sum(), 184)


# def test_stc_with_np():
#     random_state = np.random.RandomState(1)
#     X, y = load_italy_power_demand(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, random_state=random_state)

#     c = ShapeletTransformClassifier(1.0, random_state=random_state)
#     c.fit(X_train, y_train.to_numpy())

#     preds = c.predict(X_test)

#     np.testing.assert_equal((preds == y_test).sum(), 184)

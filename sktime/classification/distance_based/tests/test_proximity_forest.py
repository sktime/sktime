import numpy as np
from numpy import testing
from sktime.classification.distance_based._proximity_forest import \
    ProximityForest
from sktime.classification.distance_based._proximity_forest import \
    ProximityStump
from sktime.classification.distance_based._proximity_forest import \
    ProximityTree
from sktime.classification.distance_based._proximity_forest import \
    best_of_n_stumps
from sktime.datasets import load_gunpoint


def run_classifier_on_dataset(classifier, dataset_loader,
                              expected_predict_probas,
                              expected_predictions):
    X_train, y_train = dataset_loader(split='train', return_X_y=True)
    X_test, y_test = dataset_loader(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)
    # print(indices)
    X_train = X_train.iloc[indices]
    y_train = y_train[indices]
    X_test = X_test.iloc[indices]
    classifier.fit(X_train, y_train)
    predict_probas = classifier.predict_proba(X_test)
    # print_array(predict_probas)
    testing.assert_array_equal(predict_probas, expected_predict_probas)
    predictions = classifier.predict(X_test)
    # print(predictions)
    testing.assert_array_equal(predictions, expected_predictions)


def test_proximity_stump_on_gunpoint():
    classifier = ProximityStump(random_state=0)
    run_classifier_on_dataset(classifier, load_gunpoint,
                              stump_gunpoint_predict_probas,
                              stump_gunpoint_predictions)


def test_proximity_tree_on_gunpoint():
    classifier = ProximityTree(random_state=0, find_stump=best_of_n_stumps(5))
    run_classifier_on_dataset(classifier, load_gunpoint,
                              tree_gunpoint_predict_probas,
                              tree_gunpoint_predictions)


def test_proximity_forest_on_gunpoint():
    classifier = ProximityForest(random_state=0, n_estimators=3,
                                 find_stump=best_of_n_stumps(1))
    run_classifier_on_dataset(classifier, load_gunpoint,
                              forest_gunpoint_predict_probas,
                              forest_gunpoint_predictions)


stump_gunpoint_predict_probas = np.array([
    [0.44120236006662833, 0.5587976399333716, ],
    [0.6400296376908556, 0.35997036230914436, ],
    [0.6160567045660461, 0.3839432954339539, ],
    [0.6012284005076122, 0.39877159949238794, ],
    [0.588033086192038, 0.4119669138079619, ],
    [0.6173396452244585, 0.3826603547755414, ],
    [0.47698382946587947, 0.5230161705341205, ],
    [0.7405781022127333, 0.25942189778726665, ],
    [0.893422555759702, 0.10657744424029808, ],
    [0.5917564022219167, 0.4082435977780834, ],
])
stump_gunpoint_predictions = np.array(
    ['2', '1', '1', '1', '1', '1', '2', '1', '1', '1'])
tree_gunpoint_predict_probas = np.array([
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
])
tree_gunpoint_predictions = np.array(
    ['2', '1', '1', '2', '2', '1', '1', '1', '1', '2'])
forest_gunpoint_predict_probas = np.array([
    [0.6666666666666666, 0.3333333333333333, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
    [0.0, 1.0, ],
    [0.6666666666666666, 0.3333333333333333, ],
    [0.0, 1.0, ],
    [0.6666666666666666, 0.3333333333333333, ],
    [0.6666666666666666, 0.3333333333333333, ],
    [0.0, 1.0, ],
])
forest_gunpoint_predictions = np.array(
    ['1', '1', '1', '2', '2', '1', '2', '1', '1', '2'])

# code to generate predictions below:

# def print_array(array):
#     print('[')
#     for sub_array in array:
#         print('[', end='')
#         for value in sub_array:
#             print(value.astype(str), end='')
#             print(',', end='')
#         print('],')
#     print(']')
#
# if __name__ == "__main__":
# #    change below to prox stump / tree / forest as required
# #     classifier = ProximityForest(verbosity=1, random_state = 0,
# n_estimators = 3, find_stump = best_of_n_stumps(1))
#     classifier = ProximityTree(verbosity=1, random_state = 0, find_stump =
#     best_of_n_stumps(5))
#     # classifier = ProximityStump(verbosity=1, random_state = 0)
#     X_train, y_train = load_gunpoint(split = 'TRAIN', return_X_y = True)
#     indices = np.random.RandomState(0).permutation(10)
#     print(indices)
#     X_train = X_train.iloc[indices]
#     y_train = y_train[indices]
#     X_test, y_test = load_gunpoint(split = 'TEST', return_X_y = True)
#     X_test = X_test.iloc[indices]
#     y_test = y_test[indices]
#     classifier.fit(X_train, y_train)
#     predict_probas = classifier.predict_proba(X_test)
#     print_array(predict_probas)
#     predictions = classifier.predict(X_test)
#     print(predictions)

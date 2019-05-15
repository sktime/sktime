import sktime.datasets

import sktime.classifiers.proximity

if __name__ == "__main__":
    # """
    # Example simple usage, with arguments input via script or hard coded for testing
    # """
    # print('playgrounding...')
    # pf = sktime.classifiers.proximity.ProximityForest(random_state = 0, num_trees = 4, num_stump_evaluations = 1)
    #
    # X_train, y_train = sktime.datasets.load_gunpoint(split = 'TRAIN', return_X_y = True)
    # X_test, y_test = sktime.datasets.load_gunpoint(split = 'TEST', return_X_y = True)
    #
    # pf.fit(X_train, y_train)
    # score = pf.score(X_test, y_test)
    # print(score)

    # code to get predictions / predict_probas
    classifier = sktime.classifiers.proximity.ProximityStump(debug = True, random_state = 0,
                                                              # num_trees = 10, num_stump_evaluations = 5
                                                              )
    X_train, y_train = sktime.datasets.load_gunpoint(split = 'TRAIN', return_X_y = True)
    X_test, y_test = sktime.datasets.load_gunpoint(split = 'TEST', return_X_y = True)
    classifier.fit(X_train, y_train)
    predict_probas = classifier.predict_proba(X_test)
    print(predict_probas)
    predictions = classifier.predict(X_test)
    print(predictions)



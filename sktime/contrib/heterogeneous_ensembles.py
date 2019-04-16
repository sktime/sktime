import numpy as np
import time
from sklearn.model_selection import GridSearchCV, cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import class_distribution


# Author:   Jason Lines <j.lines@uea.ac.uk>

class HeterogeneousEnsemble:
    def __init__(self,
                 classifiers,
                 classifier_names=None,
                 classifier_param_grids=None,
                 cv_folds = 5,
                 alpha=4,
                 random_state=0,
                 verbose=0):

        if classifier_param_grids is not None and len(classifier_param_grids) != len(classifiers):
            raise ValueError("If paramaters are to be optimised for any classifiers please ensure that "
                             "len(classifier_param_grids)==len(classifiers) - specify None for any classifiers not "
                             "requiring parameter optimisation. Expected len(classifier_param_grids): "+
                             str(len(classifiers))+", found: "+str(len(classifier_param_grids))+".")

        if len(classifier_names) != len(classifiers):
            raise ValueError("Incorrect number of classifier names. Found "+str(len(classifier_names))
                             + " names and " + str(len(classifiers))+" classifiers.")

        self.classifiers = classifiers
        self.classifier_param_grids = classifier_param_grids
        self.classifier_names = classifier_names
        self.cv_folds = cv_folds
        self.alpha = alpha
        self.random_state=random_state
        self.verbose = verbose

        self.classes_ = None
        self.train_accs_by_classifier = None
        self.training_preds = None
        self.training_probas = None
        self.constituent_build_times = None

    def add_classifier(self, classifier, param_grid=None, classifier_name=None):
        self.classifiers.append(classifier)
        self.classifier_param_grids.append(param_grid)
        self.classifier_names.append(classifier_name)

    def fit(self, X, y):

        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        self.constituent_build_times = np.zeros(len(self.classifiers))
        self.train_accs_by_classifier = np.zeros(len(self.classifiers))
        self.training_preds = np.empty((len(self.classifiers), len(y)),dtype=type(y[0]))
        self.training_probas = np.empty((len(self.classifiers), len(y), len(self.classes_)))

        # build each classifier
        for c_id in range(len(self.classifiers)):
            if self.verbose > 0:
                print("Building "+self.classifier_names[c_id])
            start_time = time.time()
            self.classifiers[c_id].random_state=self.random_state
            if self.classifier_param_grids is None or self.classifier_param_grids[c_id] is None:
                pass
            else:
                grid = GridSearchCV(estimator=self.classifiers[c_id], param_grid=self.classifier_param_grids[c_id],
                                    scoring='accuracy', cv=self.param_cv_folds, verbose=self.verbose)
                self.classifiers[c_id] = grid.fit(X, y).best_estimator_

            self.training_probas[c_id] = cross_val_predict(self.classifiers[c_id], X=X, y=y, cv=self.cv_folds, method='predict_proba')

            end_time = time.time()

            self.training_preds[c_id] = np.array([self.classes_[np.argmax(x)] for x in self.training_probas[c_id]])
            self.train_accs_by_classifier[c_id] = accuracy_score(y, self.training_preds[c_id])
            self.constituent_build_times[c_id] = end_time-start_time
            self.classifiers[c_id].fit(train_x, train_y)

    def predict(self, X):
        probas = self.predict_proba(X)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds

    def predict_proba(self, X):

        output_probas = []
        train_sum = 0

        for c in range(0, len(self.classifiers)):
            this_train_acc = self.train_accs_by_classifier[c]

            # get the probas for this classifier, raise to the pwower of alpha, and multiply by confidence
            this_probas = np.multiply(this_train_acc, np.power(self.classifiers[c].predict_proba(X), self.alpha))
            output_probas.append(this_probas)
            train_sum += this_train_acc

        # sum up all of the weigthed probas
        output_probas = np.sum(output_probas, axis=0)

        sums_by_case = np.sum(output_probas, axis=1)
        sums_by_case = np.repeat([sums_by_case], len(output_probas[0]), axis=0)
        return np.divide(output_probas, sums_by_case.T)

    def preds_and_probas(self, X):
        probas = self.predict_proba(X) # does derivative transform within (if required)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds, probas


if __name__ == "__main__":
    # to suppress verbose/forced sklearn FutureWarnings
    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn

    # imports req. for testing only - can safely delete later
    from sklearn.neighbors.classification import KNeighborsClassifier as KNN
    from sklearn.ensemble.forest import RandomForestClassifier as RandFor
    from sktime.datasets import load_gunpoint

    # load train and test data
    train_x, train_y = load_gunpoint(split='TRAIN', return_X_y=True)
    test_x, test_y = load_gunpoint(split='TEST', return_X_y=True)

    # convert to sklearn expected format
    train_x = np.array([x for x in train_x.iloc[:, 0]])
    test_x = np.array([x for x in test_x.iloc[:, 0]])

    # classifiers to use
    classifiers = [KNN(metric='euclidean', algorithm='brute'), RandFor()]

    # params to evaluate in training for each classifier, indexed by classifier from 0 to num_classifiers-1
    param_grids = [
        {'n_neighbors': [1, 5]},
        {'n_estimators': [10, 50]}
    ]

    # pass classifiers and param grids into a new ensemble classifier
    he = HeterogeneousEnsemble(classifiers, param_grids)

    # fits the ensemble (training all constiuents, evaluating parameters on train data where param grids are passed)
    he.fit(train_x, train_y)

    # get the predictions and probas
    preds, probas = he.preds_and_probas(test_x)

    # print the test classification accuracy
    acc = accuracy_score(test_y, preds)
    print("Acc: "+str(acc))




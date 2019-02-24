from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
from sktime.pipeline import TSPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from numba import jit
import os
import numpy as np
import pandas as pd
import time


@jit  # simple but effective optimisation
def time_series_slope(y):
    n = y.shape[0]
    if n < 2:
        return 0
    else:
        x = np.arange(n) + 1
        x_mu = x.mean()
        return (((x * y).mean() - x_mu * y.mean())
                / ((x ** 2).mean() - x_mu ** 2))


def load_data(file_path):
    with open(file_path) as f:
        for line in f:
            if line.strip():
                if "@data" in line.lower():
                    data_started = True
                    break

        df = pd.read_csv(f, delimiter=',', header=None)
        y = df.pop(df.shape[1] - 1)
        X = pd.DataFrame([[row] for _, row in df.iterrows()])  # transform into nested pandas dataframe
    return X, y


data_path = os.path.abspath('/Users/mloning/Documents/Research/python_methods/sktime/data/Downloads')


# read in list of smaller time-series classification datasets
with open('pigs.txt', 'r') as f:
    pigs = [line.strip('\n') for line in f.readlines()]

features = [np.mean, np.std, time_series_slope]
steps = [('transform', RandomIntervalFeatureExtractor(n_intervals='sqrt', features=features)),
         ('clf', DecisionTreeClassifier())]
base_estimator = TSPipeline(steps)

n_pigs = len(pigs)
results = np.zeros((n_pigs, 3))
for i, pig in enumerate(pigs):
    print(f'{i + 1}/{n_pigs}')
    #  load data
    train_file = os.path.join(data_path, f'{pig}/{pig}_TRAIN.arff')
    X_train, y_train = load_data(train_file)
    test_file = os.path.join(data_path, f'{pig}/{pig}_TEST.arff')
    X_test, y_test = load_data(test_file)

    #  create classifier
    clf = TimeSeriesForestClassifier(base_estimator=base_estimator,
                                     criterion='entropy',
                                     n_estimators=1,
                                     bootstrap=False,
                                     oob_score=False,
                                     n_jobs=1)
    # fit
    s = time.time()
    clf.fit(X_train, y_train)
    results[i, 0] = time.time() - s

    #  predict
    s = time.time()
    y_pred = clf.predict(X_test)
    results[i, 1] = time.time() - s

    #  score
    results[i, 2] = accuracy_score(y_test, y_pred)

results = pd.DataFrame(results, columns=['fit_time', 'predict_time', 'accuracy'], index=pigs)
results.to_csv('tsforest_pigs_results.csv')
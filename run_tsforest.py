#!/usr/bin/env python3

import time
import os
import glob
from xpandas.data_container import XDataFrame, XSeries
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sktime.tsforest import TimeSeriesForestClassifier
from sklearn.metrics import accuracy_score
import re


data_path = os.path.abspath('/Users/mloning/Documents/Research/python_methods/sktime/data/Downloads')
files = glob.glob(f'{data_path}/*.zip')

n_files = len(files)
scores = {}

for i, file in enumerate(files):
    dataset = file.split('/')[-1].split('.')[0]
    print(f'{i + 1}/{n_files}', dataset)

    # load data
    def read_data(file):
        data = file.readlines()
        rows = [row.decode('utf-8').strip().split(',') for row in data]
        df = pd.DataFrame(rows, dtype=np.float)
        y = df.pop(0)
        # transform into nested pandas dataframe
        X = pd.DataFrame(XDataFrame(
            XSeries([pd.Series(row, dtype=np.float) for _, row in df.iterrows()])))
        return X, y

    zipfile = ZipFile(file)
    r = re.compile('(?!.*__MACOS*)(.*TRAIN.txt)')
    train_file_list = list(filter(r.match, zipfile.namelist()))
    if not len(train_file_list) == 0:
        train_file = list(filter(r.match, zipfile.namelist()))[0]
        X_train, y_train = read_data(zipfile.open(train_file))

        r = re.compile('(?!.*__MACOS*)(.*TEST.txt)')
        test_file = list(filter(r.match, zipfile.namelist()))[0]
        X_test, y_test = read_data(zipfile.open(test_file))
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # classification
        clf = TimeSeriesForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
        start = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end = time.time()
        print('Wall time (sec):', end - start)
        score = accuracy_score(y_test, y_pred)
        print('Score:', score)
        scores[dataset] = scores
    else:
        print('No txt files')

out = pd.DataFrame(scores)
out.to_csv('tsforest_scores.csv')


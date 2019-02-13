import os
import glob
from xpandas.data_container import XDataFrame, XSeries
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sktime.tsforest import TimeSeriesForestClassifier
from sklearn.metrics import accuracy_score

data_path = os.path.abspath('/Users/mloning/Desktop/Downloads')
files = glob.glob(f'{data_path}/*.zip')
# files = ['/Users/mloning/Desktop/Downloads/GunPoint.zip',
#          '/Users/mloning/Desktop/Downloads/ECGFiveDays.zip']
n_files = len(files)
scores = {}
for i, file in enumerate(files):
    # load data
    def read_data(file):
        data = file.readlines()
        rows = [row.decode('utf-8').strip().split(',') for row in data]
        df = pd.DataFrame(rows, dtype=np.float)
        y = df.pop(0)
        # transform into nested pandas dataframe
        X = pd.DataFrame(XDataFrame(
            XSeries([pd.Series(row, dtype=np.float) for _, row in df.iterrows()])))
        return X, y

    zipfile = ZipFile(file)
    dataset = file.split('/')[-1].split('.')[0]
    print(f'{i + 1}/{n_files}', dataset)

    train_file = zipfile.open(f'{dataset}/{dataset}_TRAIN.txt')
    X_train, y_train = read_data(train_file)

    test_file = zipfile.open(f'{dataset}/{dataset}_TEST.txt')
    X_test, y_test = read_data(test_file)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # classification
    clf = TimeSeriesForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores[dataset] = accuracy_score(y_test, y_pred)
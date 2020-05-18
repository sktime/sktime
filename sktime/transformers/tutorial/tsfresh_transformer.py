import numpy as np
import pandas as pd

from sktime.transformers.summarise import TsFreshTransfomer
from sktime.datasets import load_gunpoint
from sktime.utils.time_series import time_series_slope


# print("Training data\n"X_train.head())


X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)


# Convert to tsfresh required format
X_time_series = get_time_series_container(X_train)
y_time_series = get_formatted_predictions(y_train)

# Just testing univariate for now with default rc params
tf = TsFreshTransfomer(column_id='series_id',column_sort='series_time')

X_extraced = tf.transform(X_time_series)



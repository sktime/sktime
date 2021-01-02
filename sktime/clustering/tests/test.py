# -*- coding: utf-8 -*-
import os
import sktime
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.clustering.tests.test_sklearn_utils import (
    test_convert_df_to_learn_format,
    test_create_sklearn_k_means,
)

# print("=============== START ================")
DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
train_x, train_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)

# Utils testing
test_convert_df_to_learn_format(train_x)
test_create_sklearn_k_means(train_x, train_y)

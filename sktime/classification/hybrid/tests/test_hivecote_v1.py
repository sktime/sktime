# -*- coding: utf-8 -*-
"""HIVE-COTE v1 test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.hybrid import HIVECOTEV1
from sktime.datasets import load_unit_test


def test_hivecote_v1_on_unit_test_data():
    """Test of HIVECOTEV1 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train HIVE-COTE v1
    hc1 = HIVECOTEV1(
        random_state=0,
        stc_params={
            "estimator": RandomForestClassifier(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
        },
        tsf_params={"n_estimators": 10},
        rise_params={"n_estimators": 10},
        cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
    )
    hc1.fit(X_train, y_train)

    # assert probabilities are the same
    probas = hc1.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, hivecote_v1_unit_test_probas, decimal=2)


hivecote_v1_unit_test_probas = np.array(
    [
        [
            0.08232436967368748,
            0.9176756303263125,
        ],
        [
            0.5161621848368437,
            0.48383781516315627,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.925,
            0.075,
        ],
        [
            0.8261138340619067,
            0.17388616593809328,
        ],
        [
            0.9676756303263125,
            0.03232436967368746,
        ],
        [
            0.7869430829690466,
            0.2130569170309533,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.7661621848368437,
            0.23383781516315624,
        ],
        [
            0.95,
            0.05000000000000001,
        ],
    ]
)

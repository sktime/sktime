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
        stc_params={
            "estimator": RandomForestClassifier(n_estimators=3),
            "n_shapelet_samples": 50,
            "max_shapelets": 5,
            "batch_size": 10,
        },
        tsf_params={"n_estimators": 3},
        rise_params={"n_estimators": 3},
        cboss_params={"n_parameter_samples": 5, "max_ensemble_size": 3},
        random_state=0,
    )
    hc1.fit(X_train, y_train)

    # assert probabilities are the same
    probas = hc1.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, hivecote_v1_unit_test_probas, decimal=2)


hivecote_v1_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [0.5524211502822163, 0.4475788497177836],
        [0.0, 1.0],
        [0.8284767137362622, 0.17152328626373778],
        [0.883912529989341, 0.11608747001065901],
        [0.9746366325363073, 0.025363367463692665],
        [0.718066236959776, 0.28193376304022405],
        [0.0, 1.0],
        [0.7911461963597987, 0.20885380364020145],
        [0.7167389773758789, 0.283261022624121],
    ]
)

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for (dunder) composition functionality attached to the base class."""

import numpy as np

from sktime.classification.compose import MultiplexClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.classification.model_selection import TSCGridSearchCV
from sktime.datasets import load_unit_test


def test_multiplex_classifier():
    """Test the MultiplexClassifier with a grid search."""
    # Load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test =load_unit_test(split="test", return_X_y=True)

    estimators_list = [
        ("SimpleRNNClassifier", DummyClassifier()),
        ("Arsenal", DummyClassifier()),
        ("ContractableBOSS", DummyClassifier()),
    ]

    param_grid = {"selected_classifier": ["SimpleRNNClassifier", "Arsenal"]}

    multiplexer = MultiplexClassifier(classifiers=estimators_list)

    parameter_tuning = TSCGridSearchCV(
        multiplexer,
        param_grid,
        error_score=np.nan, # "raise" np.nan
        cv=3,
        scoring="accuracy",  # accuracy balanced_accuracy
        verbose=3,
        refit=True,
        n_jobs=1,
    )
    parameter_tuning.fit_predict(X_train, y_train)

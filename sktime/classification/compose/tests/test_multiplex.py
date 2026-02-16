# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for (dunder) composition functionality attached to the base class."""

import numpy as np
import pytest

from sktime.classification.compose import MultiplexClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.classification.feature_based import SummaryClassifier
from sktime.classification.model_selection import TSCGridSearchCV
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.classification"]),
    reason="run test only if classification or distances code has changed",
)
def test_multiplex_classifier():
    """Test the MultiplexClassifier with a grid search.

    Failure case of #8547 as it includes a classifier with
    capability:multithreading tag.
    """
    # Load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)

    estimators_list = [
        ("SimpleRNNClassifier", DummyClassifier()),
        ("Arsenal", SummaryClassifier()),
        ("ContractableBOSS", DummyClassifier()),
    ]

    param_grid = {"selected_classifier": ["SimpleRNNClassifier", "Arsenal"]}

    multiplexer = MultiplexClassifier(classifiers=estimators_list)

    parameter_tuning = TSCGridSearchCV(
        multiplexer,
        param_grid,
        error_score=np.nan,  # "raise" np.nan
        cv=3,
        scoring="accuracy",  # accuracy balanced_accuracy
        verbose=3,
        refit=True,
        n_jobs=1,
    )
    parameter_tuning.fit_predict(X_train, y_train)

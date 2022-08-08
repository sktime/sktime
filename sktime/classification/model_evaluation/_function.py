# -*- coding: utf-8 -*-
from typing import Optional

import pandas as pd
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.model_selection import BaseShuffleSplit, cross_validate

from sktime.classification.base import BaseClassifier


def evaluate_classification(
    classifier: BaseClassifier,
    X,
    y,
    scoring=None,
    cv: Optional[BaseShuffleSplit] = None,
    return_data=True,
):
    """Evaluate classifier using sklearn cross-validation.

    Parameters
    ----------
    classifier : Any sktime classifier
    X : Nested Univariate Pandas DataFrame
        All training data
    y : np.array
        Np.array of labels.
    scoring : name of classification metrics, default="accuracy_score."
        For a complete list of acceptable metrics, see
        https://scikit-learn.org/stable/modules/model_evaluation.html
    cv : Cross-validation splitter
        Result of cv-splitters in sklearn.model_selection
    return_data : bool, default=False
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.

    Returns
    -------
    pd.DataFrame
        DataFrame that contains several columns with information regarding each
        fold of the classifier.

    >>> from sktime.classification.model_evaluation import evaluate_classification
    >>> from sktime.datasets import load_arrow_head
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sklearn.model_selection import ShuffleSplit
    >>> import pandas as pd
    >>> import numpy as np

    >>> arrow_train_X, arrow_train_y = load_arrow_head(split="train",
    ... return_type="nested_univ")
    >>> arrow_test_X, arrow_test_y = load_arrow_head(split="test",
    ... return_type="nested_univ")
    >>> # Merge train and test set for cv
    >>> arrow_X = pd.concat([arrow_train_X, arrow_test_X], axis=0)
    >>> arrow_X = arrow_X.reset_index().drop(columns=["index"])
    >>> arrow_y = np.concatenate([arrow_train_y, arrow_test_y], axis=0)

    >>> cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    >>> classifier = RocketClassifier()
    >>> result = evaluate_classification(classifier=classifier,
    ... X=arrow_X, y=arrow_y, cv=cv)

    """
    # Set metrics
    list_of_scorer = get_scorer_names()  # Double check and remove this
    if scoring is None:
        scoring = "accuracy"
    if scoring not in list_of_scorer:
        print("Metrics is not valid.")  # noqa
        print("See sklearn.metrics.get_scorer_names()")  # noqa
        print("for acceptable classsification metrics.")  # noqa
    scoring_method = get_scorer(scoring)
    scores = cross_validate(
        classifier, X=X, y=y, cv=cv, n_jobs=-1, scoring=scoring_method
    )
    results = {
        "score_name": [scoring] * len(scores["test_score"]),
        "fit_time": scores["fit_time"],
        "pred_time": scores["score_time"],
        "score": scores["test_score"],
    }

    results = pd.DataFrame(results)
    return results

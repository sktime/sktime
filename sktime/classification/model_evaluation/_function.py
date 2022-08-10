# -*- coding: utf-8 -*-
import sys
from typing import Optional

import pandas as pd
from sklearn.metrics import _scorer, get_scorer
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
    """

    def get_scorer_names():
        """Get the names of all available scorers.

        Only Activate when python version is less than 3.8

        Returns
        -------
        list of str
            Names of all available scorers.
        """
        return sorted(_SCORERS.keys())

    # Set metrics
    if sys.version_info.major == 3 and sys.version_info.minor < 8:
        _SCORERS = _scorer.SCORERS
        list_of_scorer = get_scorer_names()
    else:
        _SCORERS = _scorer._SCORERS
        list_of_scorer = get_scorer_names()

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

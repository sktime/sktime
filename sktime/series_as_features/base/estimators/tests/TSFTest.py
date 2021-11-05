import unittest


def test_tsf():
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    from sktime.classification.compose import ComposableTimeSeriesForestClassifier
    from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
    from sktime.datasets import load_arrow_head
    from sktime.utils.slope_and_trend import _slope

    x, y = load_arrow_head(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    steps = [
        (
            "extract",
            RandomIntervalFeatureExtractor(n_intervals="sqrt", features=[np.mean, np.std, _slope])
        ),
        ("clf", DecisionTreeClassifier())
    ]

    time_series_tree = Pipeline(steps)

    # Fit and evaluate single time series tree
    time_series_tree.fit(x_train, y_train)
    time_series_tree.score(x_test, y_test)

    tsf = ComposableTimeSeriesForestClassifier(
        estimator=time_series_tree,
        n_estimators=100,
        # criterion="entropy", # TODO - Check if this can be removed permanently.
        bootstrap=True,
        oob_score=True,
        random_state=1,
        n_jobs=-1
    )

    # Fit and obtain oob score
    tsf.fit(x_train, y_train)

    if tsf.oob_score:
        print(tsf.oob_score_)


if __name__ == '__main__':
    unittest.main()

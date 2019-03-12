from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
from sktime.datasets import load_gunpoint
import numpy as np
from sktime.pipeline import TSPipeline
from sktime.transformers.compose import RowwiseTransformer
from sklearn.tree import DecisionTreeClassifier
from sktime.transformers.series_to_series import RandomIntervalSegmenter
from sklearn.preprocessing import FunctionTransformer


def test_random_state():
    steps = [('transform', RandomIntervalFeatureExtractor(features=[np.mean])), ('clf', DecisionTreeClassifier())]
    pipe = TSPipeline(steps)

    # Check that pipe is initiated without random_state
    assert pipe.random_state is None
    assert pipe.get_params()['random_state'] is None

    # Check that all components are initiated without random_state
    for step in pipe.steps:
        assert step[1].random_state is None
        assert step[1].get_params()['random_state'] is None

    # Check that if random state is set, it's set to itself and all its random components
    rs = 1234
    pipe.set_params(**{'random_state': rs})

    assert pipe.random_state == rs
    assert pipe.get_params()['random_state'] == rs

    for step in pipe.steps:
        assert step[1].random_state == rs
        assert step[1].get_params()['random_state'] == rs

    # Check specific results
    X_train, y_train = load_gunpoint()
    X_test, y_test = load_gunpoint("TEST")

    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals='sqrt', check_input=False)),
        ('extract', RowwiseTransformer(FunctionTransformer(func=np.mean, validate=False))),
        ('clf', DecisionTreeClassifier())
    ]
    pipe = TSPipeline(steps, random_state=rs)
    pipe.fit(X_train, y_train)
    y_pred_first = pipe.predict(X_test)
    N_ITER = 10
    for _ in range(N_ITER):
        pipe = TSPipeline(steps, random_state=rs)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        np.testing.assert_array_equal(y_pred_first, y_pred)


def test_check_input():
    steps = [('transform', RandomIntervalFeatureExtractor(features=[np.mean]))]
    pipe = TSPipeline(steps)

    # Check that pipe is initiated without check_input set to True
    assert pipe.check_input is True
    assert pipe.get_params()['check_input'] is True

    # Check that all components are initiated with check_input set to True
    for step in pipe.steps:
        assert step[1].check_input is True
        assert step[1].get_params()['check_input'] is True

    # Check that if random state is set, it's set to itself and all its random components
    ci = False
    pipe.set_params(**{'check_input': ci})

    assert pipe.check_input == ci
    assert pipe.get_params()['check_input'] == ci

    for step in pipe.steps:
        assert step[1].check_input == ci
        assert step[1].get_params()['check_input'] == ci

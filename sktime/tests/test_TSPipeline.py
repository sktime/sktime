from sktime.pipeline import TSPipeline
from sktime.transformers.series_to_tabular import RandomIntervalFeatureExtractor
from sklearn.tree import DecisionTreeClassifier
import numpy as np


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

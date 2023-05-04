import pytest
from unittest import TestCase
from sktime.pipeline.pipeline import Pipeline


@pytest.mark.parametrize("steps", [
    [{}],
    [{}],
])
def test_add_steps(steps):
    pipeline = Pipeline()
    for step in steps:
        pipeline.add_step(**step)
    TestCase().assertListEqual(steps, pipeline.steps)


def test_transform():
    pass

def test_transform_not_available():
    pass

def test_predict():
    pass

def test_predict_not_available():
    pass

def test_predict_proba():
    pass

def test_predict_proba_not_available():
    pass

def test_predict_quantile():
    pass

def test_predict_quantile_not_available():
    pass



from unittest.mock import MagicMock

import pandas as pd
import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.pipeline.step import Step, StepResult
from sktime.transformations.series.exponent import ExponentTransformer


def test_get_result_none_predecessor():
    step = Step(None, "X", None, None, {})
    step.buffer = pd.DataFrame([1, 2, 3])
    result = step.get_result(True, "transform", ["transform"], {})

    pd.testing.assert_frame_equal(result.result, pd.DataFrame([1, 2, 3]))
    assert result.mode == ""


def test_get_results_predecessors():
    skobject_mock = MagicMock()
    skobject_mock.is_fitted = False
    predecessor_mock = MagicMock()
    # predecessor_mock.get_allowed_method.return_value = ["transform"]
    predecessor_mock.get_result.return_value = StepResult(
        pd.DataFrame([1, 2, 3]), mode=""
    )
    skobject_mock.transform.return_value = pd.DataFrame([2, 3, 4])

    step = Step(
        skobject_mock, "mock", {"X": [predecessor_mock]}, "transform", {"test": 24}
    )
    result = step.get_result(True, None, ["transform"], {"additional_kwarg": "42"})

    skobject_mock.fit.assert_called_once()
    assert "X" in skobject_mock.fit.call_args[1]
    pd.testing.assert_frame_equal(
        skobject_mock.fit.call_args[1]["X"], pd.DataFrame([1, 2, 3])
    )

    skobject_mock.transform.assert_called_once()

    pd.testing.assert_frame_equal(result.result, pd.DataFrame([2, 3, 4]))
    assert result.mode == ""


@pytest.mark.parametrize(
    "skobject,allowed_methods",
    [
        (None, ["transform"]),
        (ExponentTransformer(), ["transform", "inverse_transform"]),
        (
            NaiveForecaster(),
            ["predict", "predict_quantiles", "predict_residuals", "predict_interval"],
        ),
    ],
)
def test_get_allowed_methods(skobject, allowed_methods):
    step = Step(skobject, "mock", {"X": [MagicMock()]}, "transform", {"test": 24})
    result = step.get_allowed_method()

    assert set(result) == set(allowed_methods)

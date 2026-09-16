# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseObject universal base class that require sktime or sklearn imports."""

__author__ = ["fkiraly"]

import pytest

from sktime.base._base import _BaseObject, BaseObject


def test_get_fitted_params_sklearn():
    """Tests fitted parameter retrieval with sklearn components.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on obj sktime component returns expected nested params
    """
    from sktime.datasets import load_airline
    from sktime.forecasting.trend import TrendForecaster

    y = load_airline()
    f = TrendForecaster().fit(y)

    params = f.get_fitted_params()

    assert "regressor__coef" in params.keys()
    assert "regressor" in params.keys()
    assert "regressor__intercept" in params.keys()


def test_get_fitted_params_sklearn_nested():
    """Tests fitted parameter retrieval with sklearn components.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on obj sktime component returns expected nested params
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from sktime.datasets import load_airline
    from sktime.forecasting.trend import TrendForecaster

    y = load_airline()
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    f = TrendForecaster(pipe)
    f.fit(y)

    params = f.get_fitted_params()

    assert "regressor" in params.keys()
    assert "regressor__n_features_in" in params.keys()


def test_clone_nested_sklearn():
    """Tests nested set_params of with sklearn components has no side effects."""
    from sklearn.ensemble import GradientBoostingRegressor

    from sktime.forecasting.compose import make_reduction

    sklearn_model = GradientBoostingRegressor(random_state=5, learning_rate=0.02)
    original_model = make_reduction(sklearn_model)
    copy_model = original_model.clone()
    copy_model.set_params(estimator__random_state=42, estimator__learning_rate=0.01)

    # failure condition, see issue #4704: the setting of the copy also sets the orig
    assert original_model.get_params()["estimator__random_state"] == 5


class NoneTagTestClass(_BaseObject):
    """Class for testing tag_value_default when tag is None."""

    _tags = {
        "none_tag": None,
        "real_tag": "real_value",
    }


class NoneTagTestClass2(BaseObject):
    """Class for testing tag_value_default when tag is None."""

    _tags = {
        "none_tag": None,
        "real_tag": "real_value",
    }


@pytest.mark.parametrize("cls", [NoneTagTestClass, NoneTagTestClass2])
def test_get_class_tag_default_when_none(cls):
    """Test that tag_value_default is returned when tag value is None.

    Regression test for #10305: get_class_tag returns None even if
    tag_value_default is passed, when the tag is explicitly set to None
    in _tags (e.g. "python_version": None).
    """
    # When tag is None and a non-None default is provided, return the default
    result = cls.get_class_tag("none_tag", "fallback")
    assert result == "fallback", f"Expected 'fallback', got {result!r}"

    # When tag is None and no default is provided, return None
    result = cls.get_class_tag("none_tag")
    assert result is None, f"Expected None, got {result!r}"

    # When tag has a real value, return that value even if default is provided
    result = cls.get_class_tag("real_tag", "fallback")
    assert result == "real_value", f"Expected 'real_value', got {result!r}"

    # Non-existent tag with default returns the default
    result = cls.get_class_tag("nonexistent", "fallback")
    assert result == "fallback", f"Expected 'fallback', got {result!r}"

    # Non-existent tag without default returns None
    result = cls.get_class_tag("nonexistent")
    assert result is None, f"Expected None, got {result!r}"


@pytest.mark.parametrize("cls", [NoneTagTestClass, NoneTagTestClass2])
def test_get_tag_default_when_none(cls):
    """Test that tag_value_default is returned when tag value is None for get_tag.

    Same regression as #10305 but for the instance-level get_tag method.
    """
    obj = cls()

    # When tag is None and a non-None default is provided, return the default
    result = obj.get_tag("none_tag", "fallback")
    assert result == "fallback", f"Expected 'fallback', got {result!r}"

    # When tag is None and no default is provided, return None
    result = obj.get_tag("none_tag")
    assert result is None, f"Expected None, got {result!r}"

    # When tag has a real value, return that value even if default is provided
    result = obj.get_tag("real_tag", "fallback")
    assert result == "real_value", f"Expected 'real_value', got {result!r}"

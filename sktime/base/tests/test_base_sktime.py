# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseObject universal base class that require sktime or sklearn imports."""

__author__ = ["fkiraly"]


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


def test_load_warns_on_version_mismatch():
    """Tests that loading an object saved with a different sktime version warns.

    Raises
    ------
    AssertionError if a UserWarning mentioning the sktime version mismatch is
    not raised when deserializing an object saved with a different version.
    """
    import warnings

    from sktime.forecasting.naive import NaiveForecaster

    forecaster = NaiveForecaster()
    # simulate an object that was saved by an older sktime version
    forecaster.set_tags(sktime_version="0.0.0")
    cls, stored = forecaster.save()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls.load_from_serial(stored)

    version_warnings = [w for w in caught if "sktime version" in str(w.message)]
    assert len(version_warnings) == 1
    assert issubclass(version_warnings[0].category, UserWarning)


def test_load_no_warning_on_version_match():
    """Tests that no version-mismatch warning is raised when versions match."""
    import warnings

    from sktime.forecasting.naive import NaiveForecaster

    forecaster = NaiveForecaster()
    cls, stored = forecaster.save()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls.load_from_serial(stored)

    version_warnings = [w for w in caught if "sktime version" in str(w.message)]
    assert len(version_warnings) == 0

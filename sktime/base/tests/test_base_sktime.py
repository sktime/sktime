# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseObject universal base class that require sktime or sklearn imports."""

__author__ = ["fkiraly", "achieveordie"]


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


def test_get_fitted_params_on_sklearn_pipeline():
    """Test fitted parameter retrievel with sklearn Pipeline.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        internal call to `_gfp_sklearn_pipeline_plugin()` on a single sklearn.Pipeline
        should retrieve the components without any parent names.
    """
    from numpy import nan
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import make_reduction

    y = load_airline()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(missing_values=nan, strategy="mean")),
            ("regressor", LinearRegression(fit_intercept=True)),
        ]
    )

    forecaster = make_reduction(pipeline)
    forecaster.fit(y)

    params = forecaster.get_fitted_params()

    # check one parameter for each step in the pipeline
    # no parent included since there's only one Pipeline with zero-nesting.
    assert "imputer__statistics" in params.keys()
    assert "scaler__n_features_in" in params.keys()
    assert "regressor__intercept" in params.keys()


def test_get_fitted_params_on_sklearn_nested_pipelines():
    """Test fitted parameter retrievel with sklearn Pipeline.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        internal call to `_gfp_sklearn_pipeline_plugin()` on a nested sklearn.Pipeline
        should retrieve the inner pipeline's components.
    """
    from numpy import nan
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import make_reduction

    y = load_airline()

    inner_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    pipeline = Pipeline(
        [
            ("inner_pipeline", inner_pipeline),
            ("imputer", SimpleImputer(missing_values=nan, strategy="mean")),
            ("regressor", LinearRegression()),
        ]
    )

    forecaster = make_reduction(pipeline)
    forecaster.fit(y)

    params = forecaster.get_fitted_params()

    # inner components must include their parent's name i.e. `inner_pipeline__`
    assert "inner_pipeline__n_features_in" in params.keys()
    assert "inner_pipeline__scaler__mean" in params.keys()

    assert "imputer__statistics" in params.keys()


def test_get_fitted_params_on_multi_sklearn_nested_pipelines():
    """Test fitted parameter retrievel with sklearn Pipeline.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        internal call to `_gfp_sklearn_pipeline_plugin()` on multiple nested pipelines
        should retrieve their respective components with proper parent names.
    """
    from numpy import array_equal
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import make_reduction

    y = load_airline()

    p1_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    p2_pipeline = Pipeline([("scaler", MinMaxScaler())])

    pipeline = Pipeline(
        [
            ("p1_pipeline", p1_pipeline),
            ("p2_pipeline", p2_pipeline),
            ("regressor", LinearRegression()),
        ]
    )

    forecaster = make_reduction(pipeline)
    forecaster.fit(y)

    params = forecaster.get_fitted_params()

    p1_scaler_name = "p1_pipeline__scaler__scale"
    p2_scaler_name = "p2_pipeline__scaler__scale"

    for name in [p1_scaler_name, p2_scaler_name]:
        assert name in params.keys()

    # ensure their respective values reflect that they are different
    assert not array_equal(params[p1_scaler_name], params[p2_scaler_name])


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

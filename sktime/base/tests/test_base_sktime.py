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


def test_fit_in_sklearn_pipeline():
    """Tests that sktime estimators can be fitted as steps of an sklearn Pipeline.

    Failure case of bug #11408, where ``scikit-learn>=1.9`` meta-estimators, e.g.,
    ``Pipeline``, failed to fit ``sktime`` components, since ``reset`` in ``fit``
    deleted the ``_parent_callback_ctx`` attribute they set on the components.
    """
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    from sktime.classification.dummy import DummyClassifier
    from sktime.datasets import load_unit_test
    from sktime.transformations.reduce import Tabularizer

    X, y = load_unit_test(return_X_y=True)

    transformer_pipe = make_pipeline(Tabularizer(), DecisionTreeClassifier())
    classifier_pipe = Pipeline([("clf", DummyClassifier())])

    for pipe in [transformer_pipe, classifier_pipe]:
        pipe.fit(X, y)
        sktime_step = pipe.steps[0][1]

        assert sktime_step.is_fitted
        assert not hasattr(sktime_step, "_parent_callback_ctx")

    assert len(transformer_pipe.predict(X)) == len(y)

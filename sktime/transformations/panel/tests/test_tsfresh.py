"""Tests for TSFreshFeatureExtractor."""
__author__ = ["AyushmannSeth", "mloning"]

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from sktime.datasets import load_arrow_head
from sktime.datatypes import convert
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.utils._testing.panel import make_classification_problem


@pytest.mark.skipif(
    not run_test_for_class(TSFreshFeatureExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("default_fc_parameters", ["minimal"])
def test_tsfresh_extractor(default_fc_parameters):
    """Test that mean feature of TSFreshFeatureExtract is identical with sample mean."""
    X, _ = make_classification_problem()

    transformer = TSFreshFeatureExtractor(
        default_fc_parameters=default_fc_parameters, disable_progressbar=True
    )

    Xt = transformer.fit_transform(X)
    actual = Xt.filter(like="__mean", axis=1).values.ravel()
    converted = convert(X, from_type="nested_univ", to_type="pd-wide")
    expected = converted.mean(axis=1).values
    assert expected[0] == X.iloc[0, 0].mean()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(
    not run_test_for_class(TSFreshFeatureExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_docs_tsfresh_extractor():
    """Test whether doc example runs through."""
    X, _ = load_arrow_head(return_X_y=True)[:3]
    ts_eff = TSFreshFeatureExtractor(
        default_fc_parameters="efficient", disable_progressbar=True
    )
    ts_eff.fit_transform(X)
    features_to_calc = [
        "dim_0__quantile__q_0.6",
        "dim_0__longest_strike_above_mean",
        "dim_0__variance",
    ]
    ts_custom = TSFreshFeatureExtractor(
        kind_to_fc_parameters=features_to_calc, disable_progressbar=True
    )
    ts_custom.fit_transform(X)


@pytest.mark.skipif(
    not run_test_for_class(TSFreshFeatureExtractor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_kind_tsfresh_extractor():
    """Test extractor returns an array of expected num of cols."""
    X, y = load_arrow_head(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    features_to_calc = [
        "dim_0__quantile__q_0.6",
        "dim_0__longest_strike_above_mean",
        "dim_0__variance",
    ]
    ts_custom = TSFreshFeatureExtractor(
        kind_to_fc_parameters=features_to_calc, disable_progressbar=True
    )
    Xts_custom = ts_custom.fit_transform(X_train)
    assert Xts_custom.shape[1] == len(features_to_calc)

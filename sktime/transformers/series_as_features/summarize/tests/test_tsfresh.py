__author__ = ["Ayushmann Seth", "Markus Löning"]

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sktime.transformers.series_as_features.summarize import \
    TSFreshFeatureExtractor
from sktime.transformers.series_as_features.summarize import \
    TSFreshRelevantFeatureExtractor
from sktime.utils.data_container import tabularize
from sktime.utils.testing._series_as_features import \
    make_classification_problem


@pytest.mark.parametrize("default_fc_parameters", [
    "minimal"
])
@pytest.mark.parametrize("Transformer", [
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor
])
def test_tsfresh_extractor(Transformer, default_fc_parameters):
    X, y = make_classification_problem(random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    transformer = Transformer(
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=True
    )

    Xt = transformer.fit_transform(X_train, y_train)
    actual = Xt.filter(like="__mean", axis=1).values.ravel()
    expected = tabularize(X_train).mean(axis=1).values

    assert expected[0] == X_train.iloc[0, 0].mean()
    np.testing.assert_allclose(actual, expected)

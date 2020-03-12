__author__ = ["Ayushmann Seth", "Markus Löning"]

import pandas as pd
import pytest
from sktime.datasets import load_gunpoint, load_basic_motions
from sktime.transformers.summarise import TSFreshFeatureExtractor, TSFreshRelevantFeatureExtractor


@pytest.mark.parametrize("default_fc_parameters", ["minimal"])
@pytest.mark.parametrize("load_data", [load_gunpoint, load_basic_motions])
@pytest.mark.parametrize("Transformer", [TSFreshFeatureExtractor, TSFreshRelevantFeatureExtractor])
def test_tsfresh_extractor(Transformer, load_data, default_fc_parameters):
    X_train, y_train = load_data("TRAIN", return_X_y=True)
    X_test, y_test = load_data("TEST", return_X_y=True)

    X_train = X_train.iloc[:15, :]
    y_train = y_train[:15]

    t = Transformer(default_fc_parameters=default_fc_parameters, disable_progressbar=True)
    Xt = t.fit_transform(X_train, y_train)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == X_train.shape[0]

    X_test = X_test.iloc[:2, :]
    y_test = y_test[:2]

    Xt = t.transform(X_test, y_test)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == X_test.shape[0]


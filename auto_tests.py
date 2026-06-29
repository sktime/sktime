import pandas as pd
import numpy as np
from autoencoder import AutoEncoder


def test_autoencoder_fit_predict():
    X = pd.DataFrame(np.random.randn(100, 3))

    model = AutoEncoder(epoch_num=2, verbose=0)
    model.fit(X)

    preds = model.predict(X)

    assert len(preds) == len(X)
    assert set(preds.unique()).issubset({0, 1})

def test_autoencoder_scores():
    X = pd.DataFrame(np.random.randn(50, 2))

    model = AutoEncoder(epoch_num=2, verbose=0)
    model.fit(X)

    scores = model.predict_scores(X)

    assert len(scores) == len(X)
    assert scores.dtype != object

def test_autoencoder_series_input():
    X = pd.Series(np.random.randn(100))

    model = AutoEncoder(epoch_num=2, verbose=0)
    model.fit(X)

    preds = model.predict(X)

    assert len(preds) == len(X)

def test_threshold_exists():
    X = pd.DataFrame(np.random.randn(100, 2))

    model = AutoEncoder(epoch_num=2, verbose=0)
    model.fit(X)

    assert hasattr(model, "threshold_")

def test_random_state_reproducibility():
    X = pd.DataFrame(np.random.randn(100, 2))

    model1 = AutoEncoder(epoch_num=2, random_state=42, verbose=0)
    model2 = AutoEncoder(epoch_num=2, random_state=42, verbose=0)

    model1.fit(X)
    model2.fit(X)

    scores1 = model1.predict_scores(X)
    scores2 = model2.predict_scores(X)

    assert np.allclose(scores1, scores2, atol=1e-5)


test_autoencoder_fit_predict()
test_autoencoder_scores()
test_autoencoder_series_input()
test_threshold_exists()
test_random_state_reproducibility()

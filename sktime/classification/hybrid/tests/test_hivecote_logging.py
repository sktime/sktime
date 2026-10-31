import logging

import numpy as np
import pytest

from sktime.classification.hybrid import (
    HIVECOTEV1,
    HIVECOTEV2,
    _hivecote_v1,
    _hivecote_v2,
)


class _FakeComponent:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def _get_train_probs(self, X, y, **kwargs):
        return self.predict_proba(X)

    def predict_proba(self, X):
        return np.full((X.shape[0], len(self.classes_)), 1 / len(self.classes_))


@pytest.mark.parametrize(
    "classifier_cls, module, component_names",
    [
        (
            HIVECOTEV1,
            _hivecote_v1,
            [
                "ShapeletTransformClassifier",
                "TimeSeriesForestClassifier",
                "RandomIntervalSpectralEnsemble",
                "ContractableBOSS",
            ],
        ),
        (
            HIVECOTEV2,
            _hivecote_v2,
            [
                "ShapeletTransformClassifier",
                "DrCIF",
                "Arsenal",
                "TemporalDictionaryEnsemble",
            ],
        ),
    ],
)
def test_hivecote_verbose_logs_progress(
    classifier_cls, module, component_names, monkeypatch, caplog, capsys
):
    for component_name in component_names:
        monkeypatch.setattr(module, component_name, _FakeComponent)
    if classifier_cls is HIVECOTEV1:
        monkeypatch.setattr(
            module, "cross_val_predict", lambda *args, **kwargs: kwargs["y"]
        )

    X = np.zeros((4, 1, 5))
    y = np.array([0, 1, 0, 1])
    classifier = classifier_cls(verbose=1)

    with caplog.at_level(logging.INFO, logger=module.__name__):
        classifier.fit(X, y)

    assert "STC" in caplog.text
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "classifier_cls, module, component_names",
    [
        (
            HIVECOTEV1,
            _hivecote_v1,
            [
                "ShapeletTransformClassifier",
                "TimeSeriesForestClassifier",
                "RandomIntervalSpectralEnsemble",
                "ContractableBOSS",
            ],
        ),
        (
            HIVECOTEV2,
            _hivecote_v2,
            [
                "ShapeletTransformClassifier",
                "DrCIF",
                "Arsenal",
                "TemporalDictionaryEnsemble",
            ],
        ),
    ],
)
def test_hivecote_verbose_zero_is_quiet(
    classifier_cls, module, component_names, monkeypatch, caplog
):
    for component_name in component_names:
        monkeypatch.setattr(module, component_name, _FakeComponent)
    if classifier_cls is HIVECOTEV1:
        monkeypatch.setattr(
            module, "cross_val_predict", lambda *args, **kwargs: kwargs["y"]
        )

    X = np.zeros((4, 1, 5))
    y = np.array([0, 1, 0, 1])
    with caplog.at_level(logging.INFO):
        classifier_cls(verbose=0).fit(X, y)

    assert not caplog.records

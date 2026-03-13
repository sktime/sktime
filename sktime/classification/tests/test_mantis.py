"""Tests for MantisClassifier."""

import numpy as np
import pytest

pytest.importorskip("mantis_tsfm")

from sktime.classification.deep_learning import MantisClassifier


class TestMantisClassifier:

    def setup_method(self):
        np.random.seed(42)

        self.X_train = np.random.randn(10, 3, 50)
        self.y_train = np.random.randint(0, 2, 10)

        self.X_test = np.random.randn(5, 3, 50)

    def test_mantis_classifier_init(self):
        clf = MantisClassifier()

        assert clf.pretrained is True
        assert clf.device == "cpu"
        assert clf.n_epochs == 50
        assert clf.batch_size == 32
        assert clf.lr == 1e-4
        assert clf.verbose is False

    def test_mantis_classifier_custom_init(self):
        clf = MantisClassifier(
            pretrained=False,
            n_epochs=10,
            batch_size=16,
            lr=1e-3,
            verbose=True,
        )

        assert clf.pretrained is False
        assert clf.n_epochs == 10
        assert clf.batch_size == 16
        assert clf.lr == 1e-3

    def test_fit_predict(self):
        clf = MantisClassifier(n_epochs=1, pretrained=False)

        clf.fit(self.X_train, self.y_train)

        preds = clf.predict(self.X_test)

        assert len(preds) == len(self.X_test)

    def test_predict_proba(self):
        clf = MantisClassifier(n_epochs=1, pretrained=False)

        clf.fit(self.X_train, self.y_train)

        probs = clf.predict_proba(self.X_test)

        assert probs.shape[0] == len(self.X_test)
        assert probs.shape[1] == clf.n_classes_

        assert np.allclose(probs.sum(axis=1), 1)

    def test_unfitted_error(self):
        clf = MantisClassifier()

        with pytest.raises(RuntimeError):
            clf.predict(self.X_test)
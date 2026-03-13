"""Structural tests for MantisClassifier."""

import numpy as np
import pytest

from sktime.classification.deep_learning.mantis import MantisClassifier
from sktime.classification.base import BaseClassifier


class TestMantisClassifierStructure:

    def test_mantis_is_base_classifier(self):
        assert issubclass(MantisClassifier, BaseClassifier)

    def test_mantis_has_required_methods(self):
        for method in ["_fit", "_predict", "_predict_proba"]:
            assert hasattr(MantisClassifier, method)

    def test_mantis_tags_configured(self):
        tags = MantisClassifier._tags
        assert tags.get("capability:multivariate") is True
        assert tags.get("capability:predict_proba") is True
        assert tags.get("python_dependencies") == "mantis-tsfm"

    def test_mantis_docstring(self):
        assert "Mantis" in MantisClassifier.__doc__

    def test_mantis_default_parameters(self):
        clf = MantisClassifier()
        assert clf.n_epochs == 50
        assert clf.batch_size == 32

    def test_mantis_methods_callable(self):
        clf = MantisClassifier.__new__(MantisClassifier)
        assert callable(clf._fit)
        assert callable(clf._predict)
        assert callable(clf._predict_proba)

    def test_mantis_has_get_test_params(self):
        params = MantisClassifier.get_test_params()
        assert isinstance(params, dict)
        assert params["n_epochs"] <= 1
        assert params["pretrained"] is False

    def test_mantis_unfitted_predict_error(self):
        clf = MantisClassifier()
        X = np.random.randn(5, 3, 50)

        with pytest.raises(RuntimeError):
            clf._predict(X)
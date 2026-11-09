from unittest.mock import patch

import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.utils._estimator_html_repr import (
    _HTMLDocumentationLinkMixin,
    _object_html_repr,
)


@pytest.mark.parametrize("mock_version", ["0.3.0.dev0", "0.12.0"])
def test_doc_link_generator(mock_version):
    """Test the `_doc_link_generator` method for generating documentation links."""
    with patch("sktime.__version__", mock_version):
        expected_version = mock_version.split(".dev")[0]
        expected_path = "sktime.utils._HTMLDocumentationLinkMixin"
        expected_url = (
            f"https://www.sktime.net/en/v{expected_version}/api_reference/auto_generated/"
            f"{expected_path}.html"
        )
        assert _HTMLDocumentationLinkMixin._generate_doc_link() == expected_url


def test_get_doc_link():
    """Test `_get_doc_link` when the module does not match `_doc_link_module`."""
    with patch("sktime.__version__", "0.12.0"):
        expected_url = "https://www.sktime.net/en/v0.12.0/api_reference/auto_generated/sktime.utils._HTMLDocumentationLinkMixin.html"

        assert _HTMLDocumentationLinkMixin()._get_doc_link() == expected_url


def test_html_repr_includes_parameters_table():
    """Test HTML repr includes structured parameters table."""
    est = NaiveForecaster(strategy="mean", sp=12)
    html = _object_html_repr(est)
    assert "Parameters" in html
    assert "estimator-table" in html or "parameters-table" in html
    assert "strategy" in html
    assert "user-set" in html or "sp" in html


def test_html_repr_get_params_html():
    """Test _get_params_html returns ParamsDict with correct structure."""
    est = NaiveForecaster(strategy="mean", sp=12)
    params = est._get_params_html(deep=False, doc_link="")
    html = params._repr_html_inner()
    assert "Parameters" in html
    assert "strategy" in html
    assert "user-set" in html

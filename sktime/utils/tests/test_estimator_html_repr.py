from unittest.mock import patch

import pytest

from sktime.utils._estimator_html_repr import _HTMLDocumentationLinkMixin


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

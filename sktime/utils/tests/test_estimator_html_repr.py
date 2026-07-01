from unittest.mock import patch

import pytest
from skbase.base._pretty_printing._object_html_repr import (
    _VisualBlock as _SkbaseVisualBlock,
)

from sktime.base import BaseObject
from sktime.utils import _estimator_html_repr as html_repr
from sktime.utils._estimator_html_repr import (
    _HTMLDocumentationLinkMixin,
    _VisualBlock,
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
    """Test `_get_doc_link` for a sktime object."""
    with patch("sktime.__version__", "0.12.0"):
        expected_url = (
            "https://www.sktime.net/en/v0.12.0/api_reference/auto_generated/"
            "sktime.utils._HTMLDocumentationLinkMixin.html"
        )

        assert _HTMLDocumentationLinkMixin()._get_doc_link() == expected_url


def test_get_doc_link_empty_for_non_sktime_object():
    """Test `_get_doc_link` returns empty for objects outside the doc module."""

    class ExternalDocumentationLink(_HTMLDocumentationLinkMixin):
        pass

    ExternalDocumentationLink.__module__ = "external.module"

    assert ExternalDocumentationLink()._get_doc_link() == ""


def test_visual_block_reexports_skbase_visual_block():
    """Test that sktime's compatibility import points to skbase."""
    assert _VisualBlock is _SkbaseVisualBlock


def test_object_html_repr_delegates_to_skbase(monkeypatch):
    """Test that the sktime helper delegates rendering to skbase."""
    sentinel = object()

    def fake_object_html_repr(base_object):
        assert base_object is sentinel
        return "<div>delegated</div>"

    monkeypatch.setattr(html_repr, "_skbase_object_html_repr", fake_object_html_repr)

    assert html_repr._object_html_repr(sentinel) == "<div>delegated</div>"


def test_base_object_repr_html_uses_skbase_with_sktime_doc_links():
    """Test sktime objects use skbase repr and keep sktime documentation links."""

    class DocLinkedObject(BaseObject):
        """Object with a documented parameter.

        Parameters
        ----------
        alpha : int
            Documentation for alpha.
        """

        def __init__(self, alpha=1):
            self.alpha = alpha
            super().__init__()

    DocLinkedObject.__module__ = "sktime.utils._doc_module"
    DocLinkedObject.__qualname__ = "DocLinkedObject"

    with patch("sktime.__version__", "0.12.0"):
        html_output = DocLinkedObject(alpha=2)._repr_html_()

    assert "parameters-table" in html_output
    assert "Documentation for alpha." in html_output
    assert (
        "https://www.sktime.net/en/v0.12.0/api_reference/auto_generated/"
        "sktime.utils.DocLinkedObject.html"
    ) in html_output

from unittest.mock import patch

import pytest
from packaging.version import parse as parse_version

from sktime.utils._estimator_html_repr import _HTMLDocumentationLinkMixin


@pytest.mark.parametrize("mock_version", ["0.3.0.dev0", "0.12.0"])
def test_html_documentation_link_mixin_sktime(mock_version):
    """Check the behaviour of the `_HTMLDocumentationLinkMixin` class for sktime
    default.
    """

    # mock the `__version__` where the mixin is located
    with patch("sktime.__version__", mock_version):
        mixin = _HTMLDocumentationLinkMixin()

        assert mixin._doc_link_module == "sktime"
        sktime_version = parse_version(mock_version).base_version
        # we need to parse the version manually to be sure that this test is passing in
        # other branches than `main` (that is "dev").
        assert (
            mixin._doc_link_template == f"https://www.sktime.net/en/v{sktime_version}"
            "/api_reference/auto_generated/{reduced_path}.html"
        )
        assert (
            mixin._get_doc_link() == f"https://www.sktime.net/en/v{sktime_version}"
            "/api_reference/auto_generated/"
            "sktime.utils._HTMLDocumentationLinkMixin.html"
        )


def test_html_documentation_link_mixin_get_doc_link():
    """Check the behaviour of the `_get_doc_link` with various parameter."""
    mixin = _HTMLDocumentationLinkMixin()

    # if we set `_doc_link`, then we expect to infer a module and name for the estimator
    mixin._doc_link_module = "sktime"
    mixin._doc_link_template = "https://website.com/{reduced_path}.html"
    assert (
        mixin._get_doc_link() == "https://website.com/"
        "sktime.utils._HTMLDocumentationLinkMixin.html"
    )


def test_html_documentation_link_mixin_get_doc_link_out_of_library():
    """Check the behaviour of the `_get_doc_link` with various parameter."""
    mixin = _HTMLDocumentationLinkMixin()

    # if the `_doc_link_module` does not refer to the root module of the estimator
    # (here the mixin), then we should return an empty string.
    mixin._doc_link_module = "xxx"
    assert mixin._get_doc_link() == ""


def test_html_documentation_link_mixin_doc_link_url_param_generator():
    mixin = _HTMLDocumentationLinkMixin()
    # we can bypass the generation by providing our own callable
    mixin._doc_link_template = (
        "https://website.com/{my_own_variable}.{another_variable}.html"
    )

    def url_param_generator(estimator):
        return {
            "my_own_variable": "value_1",
            "another_variable": "value_2",
        }

    mixin._doc_link_url_param_generator = url_param_generator

    assert mixin._get_doc_link() == "https://website.com/value_1.value_2.html"

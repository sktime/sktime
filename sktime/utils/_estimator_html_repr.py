# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Compatibility helpers for BaseObject HTML representations."""

import importlib

from packaging.version import parse as parse_version
from skbase.base._pretty_printing._object_html_repr import (
    _object_html_repr,
    _VisualBlock,
)

__author__ = ["RNKuhns", "mateuszkasprowicz"]
__all__ = [
    "_HTMLDocumentationLinkMixin",
    "_VisualBlock",
    "_get_reduced_path",
    "_object_html_repr",
]


def _get_reduced_path(input_path_string):
    """Remove submodules starting with an underscore to get a reduced path string."""
    substrings = input_path_string.split(".")

    index_to_remove = None
    for i, substring in enumerate(substrings):
        if substring.startswith("_"):
            index_to_remove = i
            break

    if index_to_remove is not None:
        substrings = substrings[:index_to_remove] + substrings[-1:]

    result_string = ".".join(substrings)

    return result_string


class _HTMLDocumentationLinkMixin:
    """Mixin class allowing to generate a link to the API documentation.

    This mixin relies on the ``_doc_link_module`` attribute, corresponding to the
    root module for which documentation links should be generated.
    """

    _doc_link_module = "sktime"

    @classmethod
    def _generate_doc_link(cls):
        module = importlib.import_module(cls._doc_link_module)
        version = parse_version(module.__version__).base_version
        modpath = str(cls)[8:-2]
        path = _get_reduced_path(modpath)

        return (
            f"https://www.sktime.net/en/v{version}/api_reference/auto_generated/"
            f"{path}.html"
        )

    def _get_doc_link(self):
        """Generate a link to the API documentation for a given base object.

        For compatibility with sklearn it's an instance method.

        Returns
        -------
        url : str
            The URL to the API documentation for this estimator. If the estimator does
            not belong to module `_doc_link_module`, the empty string (i.e. `""`) is
            returned.
        """
        if self.__class__.__module__.split(".")[0] != self._doc_link_module:
            return ""

        return self.__class__._generate_doc_link()

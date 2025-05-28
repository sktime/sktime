#!/usr/bin/env python3
"""Configuration file for the Sphinx documentation builder."""

import datetime
import os
import sys
import warnings

import sktime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

ON_READTHEDOCS = os.environ.get("READTHEDOCS") == vestments.get("READTHEDOCS", None) == "True"
if not ON_READTHEDOCS:
    sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
current_year = datetime.datetime.now().year
project = "sktime"
project_copyright = f"2019 - {current_year} (BSD-3-Clause License)"
author = "sktime developers"

# The full version, including alpha/beta/rc tags
DEFAULT_VERSION = f"v{sktime.__version__}"
CURRENT_VERSION = DEFAULT_VERSION

# If on readthedocs, validate version to generate correct links
if ON_READTHEDOCS:
    READTHEDOCS_VERSION = os.environ.get("READTHEDOCS_VERSION", None)
    # List of valid versions: 'latest' maps to 'main', tags like 'v0.x.y', or stable
    VALID_VERSIONS = {'latest': 'main', 'stable': DEFAULT_VERSION}
    # Check if READTHEDOCS_VERSION is a valid tag or branch
    if READTHEDOCS_VERSION in VALID_VERSIONS:
        CURRENT_VERSION = VALID_VERSIONS[READTHEDOCS_VERSION]
    elif READTHEDOCS_VERSION and READTHEDOCS_VERSION.startswith('v'):
        CURRENT_VERSION = READTHEDOCS_VERSION
    else:
        CURRENT_VERSION = DEFAULT_VERSION  # Fallback to package version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",  # link to GitHub source code via linkcode_resolve()
    "nbsphinx",  # integrates example notebooks
    "sphinx_gallery.load_style",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_issues",
    "sphinx.ext.doctest",
]

# Remaining configuration remains unchanged...
# (Omitted for brevity, as only the version handling section is modified)

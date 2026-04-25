"""Minimal tsbootstrap-compatible bootstrap classes vendored in sktime.

This module provides a small subset of the ``tsbootstrap`` API used by sktime:
``BlockBootstrap`` and ``MovingBlockBootstrap`` with a ``bootstrap`` method.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.libs.tsbootstrap._bootstrap import BlockBootstrap, MovingBlockBootstrap

__all__ = ["BlockBootstrap", "MovingBlockBootstrap"]

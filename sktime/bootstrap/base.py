"""Base class for bootstrap algorithms.

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["sunkireddy-Barath"]
__all__ = ["BaseBootstrap"]

from sktime.base import BaseObject


class BaseBootstrap(BaseObject):
    """Base class for bootstrap resampling algorithms."""

    _tags = {
        "object_type": "bootstrap",
        "capability:bootstrap_index": False,
    }

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class template for data sets.

    class name: BaseDataset

Scitype defining methods:
    loading dataset              - load()
    loading object from dataset  - load(*args)

Inspection methods:
    hyper-parameter inspection   - get_params()
"""

__author__ = ["fkiraly"]

__all__ = ["BaseDataset"]

from sktime.base import BaseObject
from sktime.utils.validation._dependencies import _check_estimator_deps


class BaseDataset(BaseObject):
    """Base class for datasets."""

    # default tag values - these typically make the "safest" assumption
    _tags = {}

    def __init__(self):

        super().__init__()
        _check_estimator_deps(self)

    def __call__(self, *args, **kwargs):
        """Load the dataset, same as calling load."""
        return self.load(*args, **kwargs)

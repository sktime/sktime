"""Implements base class for defining performance metric in sktime."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["rnkuhns", "fkiraly"]
__all__ = ["BaseMetric"]

from sktime.base import BaseObject
from sktime.utils.dependencies import _check_estimator_deps


class BaseMetric(BaseObject):
    """Base class for defining metrics in sktime.

    Extends sktime BaseObject.
    """

    _tags = {
        "object_type": "metric",
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
    }

    def __init__(self):
        super().__init__()

        # this block has a double purpose:
        # - emit a warning if dependencies are not met, but allow instantiation
        # - if dependencies are met, call __post_init__ used by inheriting classes
        if _check_estimator_deps(self, severity="warning"):
            self.__post_init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * dynamic tag setting
        * any soft dependency imports in the constructor
        """
        pass

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : ground truth prediction target values
            type depending on the loss type, abstract descendants

        y_pred : predicted values
            type depending on the loss type, abstract descendants

        Returns
        -------
        loss : type depending on the loss type, abstract descendants
        """
        return self.evaluate(y_true, y_pred, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : ground truth prediction target values
            type depending on the loss type, abstract descendants

        y_pred : predicted values
            type depending on the loss type, abstract descendants

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        raise NotImplementedError("abstract method")

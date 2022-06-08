# -*- coding: utf-8 -*-
"""Implements base class for defining performance metric in sktime."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["rnkuhns", "fkiraly"]
__all__ = ["BaseMetric"]

from warnings import warn

from sktime.base import BaseObject


class BaseMetric(BaseObject):
    """Base class for defining metrics in sktime.

    Extends sktime BaseObject.
    """

    # todo: 0.13.0, remove the func/name args here and to all the metrics
    def __init__(
        self,
        func=None,
        name=None,
    ):

        if func is not None or name is not None:
            warn(
                "func and name parameters of BaseMetric are deprecated from 0.12.0"
                "and will be removed in 0.13.0",
                DeprecationWarning,
            )

        self.func = func
        self.name = name

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

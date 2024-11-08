"""Implements base class for customer sample weights of performance metric in sktime."""

__author__ = ["markussagen"]
__all__ = ["BaseSampleWeightGenerator", "check_sample_weight_generator"]

from sktime.base import BaseObject
from sktime.performance_metrics.forecasting.sample_weight._types import (
    check_sample_weight_generator,
)


class BaseSampleWeightGenerator(BaseObject):
    """Base class for defining sample weight generators in sktime.

    Extends sktime BaseObject.
    """

    _tags = {
        "object_type": "sample_weight",
        "authors": "markussagen",
        "maintainers": "markussagen",
    }

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred=None, **kwargs):
        """Generate sample weights for a given set of true and predicted values.

        Parameters
        ----------
        y_true : time series in sktime compatible data container format
            Ground truth (correct) target values.

        y_pred : time series in sktime compatible data container format, optional
            Predicted values to evaluate against ground truth.
            Must be of same format as y_true, same indices and columns if indexed.
            If None, only y_true is used for weight generation.

        **kwargs : dict
            Additional keyword arguments specific to the weight generation method.

        Returns
        -------
        sample_weight : 1D np.ndarray or None
            Sample weights for each time point.

            * If an array, must be 1D.
              If y_true and y_pred are a single time series,
              sample_weight must be of the same length as y_true.
              If the time series are panel or hierarchical, the length of all
              individual time series must be the same, and equal to the length
              of sample_weight, for all instances of time series passed.
            * If None, the time indices are considered equally weighted.
        """
        raise NotImplementedError("abstract method")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [{}]

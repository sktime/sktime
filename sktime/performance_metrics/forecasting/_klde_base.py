#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class for KL-divergence metrics assuming Laplace-distributed errors."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class _KLDivergenceLaplaceBase(BaseForecastingErrorMetric):
    """Base class for KL-DE1 and KL-DE2 metrics.

    Implements the shared Laplace KL-divergence loss:
    ``exp(-|e|/sigma) + |e|/sigma - 1``, averaged over time points.

    Subclasses must implement ``_compute_rolling_scale`` to provide the
    scale estimator (standard deviation for KL-DE1, MAD for KL-DE2).
    """

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
        eps=None,
        window=None,
    ):
        self.eps = eps
        self.window = window
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _compute_rolling_scale(self, y_true_vals, eps):
        """Compute rolling scale estimate for each time index.

        Must be implemented by subclasses.

        Parameters
        ----------
        y_true_vals : np.ndarray, shape (n,) or (n, p)
            True values as numpy array.
        eps : float
            Epsilon for clamping.

        Returns
        -------
        sigma : np.ndarray, same shape as y_true_vals
            Rolling scale estimate, clamped to at least eps.
        """
        raise NotImplementedError("Subclasses must implement _compute_rolling_scale")

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from
        evaluate_by_index

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.

        y_pred : pandas.DataFrame
            Predicted values to evaluate.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.
        """
        multioutput = self.multioutput

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        sigma = self._compute_rolling_scale(y_true.values, eps)

        abs_error = np.abs(y_true.values - y_pred.values)
        ratio = abs_error / sigma

        per_index = np.exp(-ratio) + ratio - 1.0

        raw_values = pd.DataFrame(
            per_index, index=y_true.index, columns=y_true.columns
        )
        raw_values = self._get_weighted_df(raw_values, **kwargs)

        return self._handle_multioutput(raw_values, multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test
            instance, i.e., ``MyClass(**params)`` or ``MyClass(**params[i])``
            creates a valid test instance.
            ``create_test_instance`` uses the first (or only) dictionary in
            ``params``
        """
        params1 = {}
        params2 = {"eps": 1e-6}
        params3 = {"window": 5}
        return [params1, params2, params3]

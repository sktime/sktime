"""Base class for probabilistic forecasters with vectorized predict_proba support.

This module provides a BaseProbaForecaster class that extends sktime's BaseForecaster
to support vectorized predict_proba for hierarchical/panel forecasting.

The standard BaseForecaster raises NotImplementedError when predict_proba is called
on vectorized (hierarchical) data. This class implements the missing vectorization
logic by following the same pattern used for other probabilistic methods like
predict_quantiles and predict_interval.
"""

__author__ = ["marrov"]
__all__ = ["BaseProbaForecaster"]

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.warnings import warn


class BaseProbaForecaster(BaseForecaster):
    """Base forecaster with vectorized predict_proba support.

    This class extends BaseForecaster to provide vectorized predict_proba
    functionality for hierarchical/panel forecasting, which is not implemented
    in the standard sktime BaseForecaster.

    The vectorization follows the same pattern as predict_quantiles and
    predict_interval, iterating over instances and concatenating the resulting
    distributions.
    """

    def predict_proba(self, fh=None, X=None, marginal=True):
        """Compute/return fully probabilistic forecasts.

        Extends the base predict_proba to support vectorized (hierarchical) data.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.

        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : skpro BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        if not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "fully probabilistic predictions. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )
        self.check_is_fitted()

        # predict_proba requires skpro to provide the distribution object returns
        msg = (
            "Forecasters' predict_proba requires "
            "skpro to be present in the python environment, "
            "for distribution objects to represent distributional forecasts. "
            "To silence this message, ensure skpro is installed in the environment "
            "when calling forecasters' predict_proba."
        )
        non_default_pred_proba = self._has_implementation_of("_predict_proba")
        skpro_present = _check_soft_dependencies("skpro", severity="none")

        if not non_default_pred_proba and not skpro_present:
            warn(msg, obj=self, stacklevel=2)

        # input checks and conversions

        # check fh and coerce to ForecastingHorizon, if not already passed in fit
        fh = self._check_fh(fh, pred_int=True)

        # check and convert X
        X_inner = self._check_X(X=X)

        # Handle vectorized case (hierarchical/panel data)
        if hasattr(self, "_is_vectorized") and self._is_vectorized:
            pred_dist = self._vectorize_predict_proba(
                fh=fh, X=X_inner, marginal=marginal
            )
        else:
            # Non-vectorized case: call the inner method directly
            try:
                pred_dist = self._predict_proba(fh=fh, X=X_inner, marginal=marginal)
            except ImportError as e:
                if non_default_pred_proba and not skpro_present:
                    raise ImportError(msg)
                else:
                    raise e

        return pred_dist

    def _vectorize_predict_proba(self, fh, X, marginal):
        """Vectorized predict_proba for hierarchical/panel data.

        This method iterates over instances in the vectorized data structure
        and concatenates the resulting distributions.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : inner mtype format
            Exogeneous time series, already converted to inner format.
        marginal : bool
            Whether returned distribution is marginal by time index.

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Concatenated predictive distribution for all instances.
        """
        # Get the vectorized data structure
        yvec = self._yvec

        # Get the list of forecasters (one per instance)
        forecasters = self.forecasters_

        # Prepare X for vectorization
        kwargs = {"args_rowvec": {"X": X}}

        # Collect distributions from each forecaster
        dist_list = yvec.vectorize_est(
            forecasters,
            method="_predict_proba",
            return_type="list",
            backend=self.get_config()["backend:parallel"],
            backend_params=self.get_config()["backend:parallel:params"],
            fh=fh,
            marginal=marginal,
            **kwargs,
        )

        # Concatenate distributions
        # The distributions need to be combined with proper indexing
        if len(dist_list) == 1:
            return dist_list[0]

        # Concatenate distributions with proper hierarchical indexing
        pred_dist = self._concat_distributions_hierarchical(dist_list, yvec)

        return pred_dist

    def _concat_distributions_hierarchical(self, dist_list, yvec):
        """Concatenate distributions from vectorized prediction.

        Parameters
        ----------
        dist_list : list of skpro BaseDistribution
            List of distributions from each instance.
        yvec : VectorizedDF
            The vectorized data structure with instance information.

        Returns
        -------
        concat_dist : skpro BaseDistribution
            Concatenated distribution with proper hierarchical index.
        """
        import inspect

        import numpy as np
        import pandas as pd

        if len(dist_list) == 0:
            raise ValueError("Cannot concatenate empty list of distributions")

        if len(dist_list) == 1:
            return dist_list[0]

        # Get the distribution class from the first distribution
        dist_class = type(dist_list[0])

        # Get the instance indices from the vectorized structure
        # yvec.get_iter_indices returns (row_idx, col_idx) where row_idx is a MultiIndex
        row_idx, _ = yvec.get_iter_indices()

        # Build the combined index
        # Each distribution has its own index (time points)
        # We need to add the instance levels to create a hierarchical index
        combined_indices = []
        for i, dist in enumerate(dist_list):
            dist_index = dist.index
            # Get the instance identifier for this distribution
            instance_idx = row_idx[i]  # This is a tuple like ('h0_0', 'h1_0')

            # Create MultiIndex with instance levels + time level
            if isinstance(instance_idx, tuple):
                # Multiple hierarchy levels
                new_tuples = [instance_idx + (t,) for t in dist_index]
            else:
                # Single hierarchy level
                new_tuples = [(instance_idx, t) for t in dist_index]

            # Get the index names
            if isinstance(row_idx, pd.MultiIndex):
                instance_names = list(row_idx.names)
            else:
                instance_names = [
                    row_idx.name if hasattr(row_idx, "name") else "level_0"
                ]

            time_name = dist_index.name if dist_index.name is not None else "time"

            new_index = pd.MultiIndex.from_tuples(
                new_tuples,
                names=instance_names + [time_name],
            )
            combined_indices.append(new_index)

        # Concatenate all indices
        full_index = combined_indices[0]
        for idx in combined_indices[1:]:
            full_index = full_index.append(idx)

        # Get parameter names from the distribution's signature
        sig = inspect.signature(dist_class.__init__)
        param_names = [
            p
            for p in sig.parameters.keys()
            if p not in ["self", "index", "columns", "args", "kwargs"]
        ]

        # Stack parameters from all distributions
        param_arrays = {}
        for param in param_names:
            vals = []
            has_param = False
            for dist in dist_list:
                if hasattr(dist, param):
                    val = getattr(dist, param)
                    if val is not None:
                        has_param = True
                        # Ensure val is 2D array with shape (n_samples, n_columns)
                        val_arr = np.atleast_2d(val)
                        # If val is 1D and was made 2D as (1, n), transpose to (n, 1)
                        if val_arr.shape[0] == 1 and len(dist.index) > 1:
                            val_arr = val_arr.T
                        vals.append(val_arr)
            if has_param and len(vals) == len(dist_list):
                # Concatenate values along axis 0 (samples)
                concatenated = np.vstack(vals)
                param_arrays[param] = concatenated

        # Get columns from first distribution
        columns = dist_list[0].columns

        if param_arrays:
            return dist_class(**param_arrays, index=full_index, columns=columns)
        else:
            # Fallback: return first distribution (shouldn't normally happen)
            return dist_list[0]

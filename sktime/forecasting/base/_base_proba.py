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

import pandas as pd

from sktime.forecasting.base._base import BaseForecaster
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
            Exogenous time series to use in prediction.
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

        # Non-vectorized path is identical to BaseForecaster behavior.
        if not getattr(self, "_is_vectorized", False):
            return super().predict_proba(fh=fh, X=X, marginal=marginal)

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

        # Vectorized case (hierarchical/panel data)
        try:
            pred_dist = self._vectorize_predict_proba(
                fh=fh,
                X=X_inner,
                marginal=marginal,
            )
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
            Exogenous time series, already converted to inner format.
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
        import numpy as np
        import pandas as pd
        from skpro.distributions import Empirical

        if len(dist_list) == 0:
            raise ValueError("Cannot concatenate empty list of distributions")

        if len(dist_list) == 1:
            return dist_list[0]

        dist_class = type(dist_list[0])
        if not all(type(dist) is dist_class for dist in dist_list):
            raise TypeError(
                "All distributions in `dist_list` must have the same type, "
                f"but found {[type(dist).__name__ for dist in dist_list]}."
            )

        # Get the instance indices from the vectorized structure
        # yvec.get_iter_indices returns (row_idx, col_idx) where row_idx is a MultiIndex
        row_idx, _ = yvec.get_iter_indices()
        if row_idx is None:
            row_idx = pd.RangeIndex(len(dist_list))

        # Empirical distributions are concatenated through sample frame ``spl``.
        if all(isinstance(dist, Empirical) for dist in dist_list):
            return self._concat_empirical_distributions(dist_list, row_idx)

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

        # Concatenate all index frames row-wise (MultiIndex.append is deprecated).
        full_index = pd.concat([idx.to_frame() for idx in combined_indices]).index

        # Get columns from first distribution
        columns = dist_list[0].columns
        n_cols = len(columns)

        first_params = dist_list[0].get_params(deep=False)
        param_names = [k for k in first_params.keys() if k not in ["index", "columns"]]

        if len(param_names) == 0:
            raise RuntimeError(
                "Unable to concatenate non-empirical distributions without explicit "
                f"parameters. Unsupported distribution type: {dist_class.__name__}."
            )

        param_arrays = {param: [] for param in param_names}

        for dist in dist_list:
            dist_params = dist.get_params(deep=False)
            dist_param_names = [
                k for k in dist_params.keys() if k not in ["index", "columns"]
            ]
            if set(dist_param_names) != set(param_names):
                raise RuntimeError(
                    "Cannot concatenate distributions with mismatched parameters. "
                    f"Expected {sorted(param_names)}, found {sorted(dist_param_names)}."
                )

            n_rows = len(dist.index)
            for param in param_names:
                param_arr = self._coerce_distribution_param(
                    value=dist_params[param],
                    n_rows=n_rows,
                    n_cols=n_cols,
                    param_name=param,
                    dist_name=dist_class.__name__,
                )
                param_arrays[param].append(param_arr)

        stacked_params = {k: np.vstack(v) for k, v in param_arrays.items()}

        try:
            return dist_class(**stacked_params, index=full_index, columns=columns)
        except Exception as exc:
            raise RuntimeError(
                "Failed to concatenate non-empirical distributions of type "
                f"{dist_class.__name__}."
            ) from exc

    def _coerce_distribution_param(self, value, n_rows, n_cols, param_name, dist_name):
        """Coerce a distribution parameter to 2D array shape ``(n_rows, n_cols)``."""
        import numpy as np

        arr = np.asarray(value)

        if arr.ndim == 0:
            return np.full((n_rows, n_cols), arr)

        if arr.ndim == 1:
            if n_cols == 1 and arr.shape[0] == n_rows:
                return arr.reshape(-1, 1)
            if n_rows == 1 and arr.shape[0] == n_cols:
                return arr.reshape(1, -1)
            raise RuntimeError(
                f"Unsupported shape {arr.shape} for parameter `{param_name}` in "
                f"{dist_name}; expected compatible with {(n_rows, n_cols)}."
            )

        if arr.ndim == 2:
            if arr.shape != (n_rows, n_cols):
                raise RuntimeError(
                    f"Unsupported shape {arr.shape} for parameter `{param_name}` in "
                    f"{dist_name}; expected {(n_rows, n_cols)}."
                )
            return arr

        raise RuntimeError(
            f"Unsupported parameter rank ({arr.ndim}) for `{param_name}` in "
            f"{dist_name}; expected scalar, 1D, or 2D array-like."
        )

    def _get_instance_names_from_row_idx(self, row_idx):
        """Get instance names from row_idx for MultiIndex construction.

        Parameters
        ----------
        row_idx : pd.Index or pd.MultiIndex
            The row index from which to extract names.

        Returns
        -------
        list
            List of instance names.
        """
        if isinstance(row_idx, pd.MultiIndex):
            return list(row_idx.names)
        else:
            return [row_idx.name if hasattr(row_idx, "name") else "level_0"]

    def _concat_empirical_distributions(self, dist_list, row_idx):
        """
        Concatenate empirical distributions from vectorized prediction.

        Parameters
        ----------
        dist_list : list of skpro BaseDistribution
            List of empirical distributions from each instance.
        row_idx : pd.Index or pd.MultiIndex
            The row index from the vectorized structure,
            used for constructing the combined index.

        Returns
        -------
        concat_dist : skpro BaseDistribution
            Concatenated empirical distribution with proper hierarchical index.
        """
        dist_class = type(dist_list[0])

        spl_dfs = []
        combined_indices = []

        instance_names = self._get_instance_names_from_row_idx(row_idx)

        for i, dist in enumerate(dist_list):
            spl = dist.spl
            instance_idx = row_idx[i]

            if isinstance(spl.index, pd.MultiIndex):
                sample_level = spl.index.get_level_values(0)
                time_level = spl.index.get_level_values(-1)

                if isinstance(instance_idx, tuple):
                    new_tuples = [
                        (s,) + instance_idx + (t,)
                        for s, t in zip(sample_level, time_level)
                    ]
                else:
                    new_tuples = [
                        (s, instance_idx, t) for s, t in zip(sample_level, time_level)
                    ]

                spl_sample_name = spl.index.names[0]
                spl_time_name = spl.index.names[-1]
                new_names = [spl_sample_name] + instance_names + [spl_time_name]

                new_spl_index = pd.MultiIndex.from_tuples(new_tuples, names=new_names)
            else:
                new_spl_index = spl.index

            new_spl = spl.copy()
            new_spl.index = new_spl_index
            spl_dfs.append(new_spl)

            dist_index = dist.index
            if isinstance(instance_idx, tuple):
                new_dist_tuples = [instance_idx + (t,) for t in dist_index]
            else:
                new_dist_tuples = [(instance_idx, t) for t in dist_index]

            time_name = dist_index.name if dist_index.name is not None else "time"

            new_dist_index = pd.MultiIndex.from_tuples(
                new_dist_tuples,
                names=instance_names + [time_name],
            )
            combined_indices.append(new_dist_index)

        combined_spl = pd.concat(spl_dfs, axis=0)

        # Concatenate indices using pd.concat (MultiIndex.append is deprecated)
        full_index = pd.concat([idx.to_frame() for idx in combined_indices]).index

        columns = dist_list[0].columns
        return dist_class(spl=combined_spl, index=full_index, columns=columns)

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ToTo forecaster."""

# This product includes software developed at Datadog, Copyright 2025 Datadog, Inc.

__author__ = ["JATAYU000", "Datadog"]
__all__ = ["TotoForecaster"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TotoForecaster(BaseForecaster):
    """
    Interface for Toto forecaster for zero-shot forecasting.

    Toto is a foundation model for multivariate time series forecasting with a focus on
    observability metrics. This model leverages innovative architectural designs to
    efficiently handle the high-dimensional, complex time series that are characteristic
    of observability data. Generate both point forecasts and uncertainty estimates using
    a Student-T mixture model. Support for variable prediction horizons and context
    lengths.

    Parameters
    ----------
    num_samples : int
        Number of samples for probabilistic forecasting
    samples_per_batch : int, optional (default=1)
        Control memory usage during inference
    prediction_type : string, optional (default='median')
        Type of prediction to generate ('mean' or 'median').
    scale_factor_exponent : int, optional (default=10)
        Exponent for the scale factor used in the model.
    stabilize_with_global : boolean, optional (default=True)
        Whether to stabilize the model with global context.
    use_memory_efficient_attention : boolean, optional (default=True)
        Whether to use memory-efficient attention mechanisms using Xformers.
    model_path : string, optional (default='Datadog/Toto-Open-Base-1.0')
        Path to the Toto huggingface model.
    device : string, optional (default=None)
        Specifies the device on which to run the model on ('cpu' or 'cuda').

    References
    ----------
    .. [1] https://github.com/DataDog/toto

    Examples
    --------
    >>> from sktime.forecasting.toto import TotoForecaster
    >>> model = TotoForecaster() # needs work not completed
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame"],
        "X_inner_mtype": "None",
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        # contribution and dependency tags
        "authors": ["JATAYU000", "Datadog"],
        # Datadog for Datadog/toto
        "maintainers": ["JATAYU000"],
        "python_version": ">= 3.10",
        "python_dependencies": ["torch>=2.5", "toto-ts>=0.1.3"],
    }

    def __init__(
        self,
        id_mask=None,
        padding_mask=None,
        num_samples: int = 1,
        samples_per_batch: int = 1,
        prediction_type: str = "median",
        scale_factor_exponent: int = 10,
        stabilize_with_global: bool = True,
        use_memory_efficient_attention: bool = False,
        model_path: str = "Datadog/Toto-Open-Base-1.0",
        device=None,
    ):
        self.model_path = model_path
        self.device = device
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.use_memory_efficient_attention = use_memory_efficient_attention
        if self.use_memory_efficient_attention:
            if _check_soft_dependencies("xformers", severity="warning"):
                self.set_tags(python_dependencies=["torch", "xformers", "accelerate"])
            else:
                raise ImportError(
                    """
                    xformers is required for memory efficient attention.
                    Refer to https://github.com/facebookresearch/xformers
                    """
                )
        self.stabilize_with_global = stabilize_with_global
        self.scale_factor_exponent = scale_factor_exponent
        self.prediction_type = prediction_type
        if prediction_type not in ["mean", "median"]:
            raise ValueError("prediction_type must be either 'mean' or 'median'")

        # Original Implementation
        self.id_mask = id_mask
        self.padding_mask = padding_mask

        super().__init__()

    def _get_toto_key(self):
        """Get a unique key for the Toto model based on configuration parameters.

        This key is used by the _multiton decorator to ensure only one instance
        of a model with specific parameters exists.

        Returns
        -------
        tuple
            Unique identifier for this model configuration
        """
        kwargs = self._get_toto_kwargs()
        key = {
            **kwargs,
            "device": self._device,
        }
        return str(sorted(key.items()))

    def _get_toto_kwargs(self):
        """Get keyword arguments for the Toto model.

        Returns
        -------
        dict
            Keyword arguments for the Toto model.
        """
        return {
            "pretrained_model_name_or_path": self.model_path,
            "use_memory_efficient_attention": self.use_memory_efficient_attention,
            "stabilize_with_global": self.stabilize_with_global,
            "scale_factor_exponent": self.scale_factor_exponent,
        }

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        if self.device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        self._forecaster = _CachedToToForecaster(
            key=self._get_toto_key(),
            toto_kwargs=self._get_toto_kwargs(),
            device=self._device,
        ).load_from_checkpoint()

        # self.get_series_params_from_y(y)

        # -- Original Implementation --
        self._input_series = torch.tensor(y.values.T, dtype=torch.float32).to(
            self._device
        )

        if self.id_mask is None:
            self._id_mask = torch.zeros_like(self._input_series).to(self._device)
        else:
            self._id_mask = self.id_mask

        if self.padding_mask is None:
            self._padding_mask = torch.full_like(
                self._input_series, True, dtype=torch.bool
            ).to(self._device)
        else:
            self._padding_mask = self.padding_mask
        # -- End of Original Implementation --

        self._y_columns = self._y.columns
        self._cutoff = self._y.index[-1]

        # current model does not use these two variable, might be needed in future.
        self.timestamp_seconds = torch.zeros_like(self._input_series)
        self.time_interval_seconds = torch.full(
            (self._input_series.shape[0],), 60 * 15, dtype=torch.float32
        ).to(self._device)

        self._series = MaskedTimeseries(
            series=self._input_series,
            padding_mask=self._padding_mask,
            id_mask=self._id_mask,
            timestamp_seconds=self.timestamp_seconds,
            time_interval_seconds=self.time_interval_seconds,
        )

        return self

    def get_series_params_from_y(self, y):
        """Extract series parameters from the input DataFrame.

        Extracts the input series, id mask, and padding mask from the
        input DataFrame if it has the correct MultiIndex structure.
        Takes default values if the input is a simple DataFrame with a PeriodIndex.
        Writes the extracted parameters to self.

        Returns
        -------
        None
        """
        import torch

        from sktime.datatypes import check_is_mtype

        # hier with indices id_mask,padding_mask,var_name,time_stamp
        if check_is_mtype(y, "pd_multiindex_hier") and len(y.index.names) == 4:
            # predict return a DataFrame
            self._y_metadata["mtype"] = "pd.DataFrame"
            id_mask_level, padding_level, var_level, time_level = y.index.names
            self._y = y.reset_index().pivot(
                index=time_level, columns=var_level, values=y.columns[0]
            )
            self._y = self._y.rename_axis(None, axis=1).rename_axis(time_level, axis=0)
            self._input_series = torch.tensor(self._y.values.T, dtype=torch.float32).to(
                self._device
            )
            n, d = self._input_series.shape
            self._id_mask = (
                torch.tensor(y.index.get_level_values(id_mask_level))
                .reshape(n, d)
                .to(self._device)
            )
            self._padding_mask = (
                torch.tensor(y.index.get_level_values(padding_level))
                .reshape(n, d)
                .to(self._device)
            )

        # dataframe where id_mask and padding_mask needs to be default.
        elif isinstance(y.index, pd.PeriodIndex):
            self._y = y
            self._input_series = torch.tensor(y.values.T, dtype=torch.float32).to(
                self._device
            )
            self._id_mask = torch.zeros_like(self._input_series).to(self._device)
            self._padding_mask = torch.full_like(
                self._input_series, True, dtype=torch.bool
            ).to(self._device)

        # unsupported input format
        else:
            raise ValueError(
                f"Unsupported input format with indicies {y.index.names}\n"
                f"TotoForecaster supports the following MultiIndex input structure\n"
                f"[id_mask, padding_mask, variable, time]"
            )

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        prediction_length = max(fh)

        self._forecast = self._forecaster.forecast(
            self._series,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
        )
        if self.prediction_type.lower() == "median":
            all_predictions = self._forecast.median.cpu().squeeze(0).numpy().T
        else:
            all_predictions = self._forecast.mean.cpu().squeeze(0).numpy().T

        pred_index = fh.to_absolute(self._cutoff)._values
        relative_indices = fh.to_relative(self._cutoff) - 1
        selected_predictions = all_predictions[relative_indices]

        y_pred = pd.DataFrame(
            selected_predictions, index=pred_index, columns=self._y_columns
        )
        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        import torch

        self._predict(fh)
        var_names = self._y_columns
        cols_idx = pd.MultiIndex.from_product([var_names, alpha])
        pred_index = fh.to_absolute(self._cutoff)._values
        relative_indices = fh.to_relative(self._cutoff) - 1

        pred_quantiles = pd.DataFrame(index=pred_index, columns=cols_idx)
        alpha_tensor = torch.tensor(alpha, device=self._device)

        quantiles = self._forecast.quantile(alpha_tensor)
        if quantiles.dim() > 3:
            quantile_values = quantiles.cpu().squeeze(1).numpy()
        else:
            quantile_values = quantiles.cpu().numpy()

        for i, var_name in enumerate(var_names):
            for j, a in enumerate(alpha):
                selected_quantiles = quantile_values[j, i, relative_indices]
                pred_quantiles[(var_name, a)] = selected_quantiles
        return pred_quantiles

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # needs work not completed
        test_params = [
            {"num_samples": 4, "samples_per_batch": 2, "prediction_type": "median"},
            {"num_samples": 4, "samples_per_batch": 4, "prediction_type": "mean"},
        ]

        return test_params


@_multiton
class _CachedToToForecaster:
    """Cached Toto forecaster.

    Toto is a zero-shot model and immutable, hence there will not be
    any side effects of sharing the same instance across multiple uses.
    This caching mechanism uses the _multiton decorator to ensure
    that models with the same configuration are reused, preventing
    duplicate models in memory when handling multivariate data.
    """

    def __init__(self, key, toto_kwargs, device):
        self.key = key
        self.toto_kwargs = toto_kwargs
        self.device = device
        self.forecaster = None

    def load_from_checkpoint(self):
        if self.forecaster is not None:
            return self.forecaster

        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        toto_model = Toto.from_pretrained(**self.toto_kwargs)
        toto_model.to(self.device)
        toto_model.compile()
        self.forecaster = TotoForecaster(toto_model.model)

        return self.forecaster

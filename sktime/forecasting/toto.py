# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ToTo forecaster."""


# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

__author__ = ["JATAYU000"]
__all__ = ["ToToForecaster"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class ToToForecaster(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    and so on
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
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
        # ----------------------------------------------------------------------------
        # packaging info - only required for sktime contribution or 3rd party packages
        # ----------------------------------------------------------------------------
        #
        # ownership and contribution tags
        # -------------------------------
        # an author is anyone with significant contribution to the code at some point
        "authors": ["JATAYU000", "DataDog"],
        "maintainers": [],
        # dependency tags: python version and soft dependencies
        # -----------------------------------------------------
        "python_version": None,
        "python_dependencies": ["torch", "toto-ts"],
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        num_samples: int = 1,
        samples_per_batch: int = 1,
        prediction_type: str = "median",
        use_memory_efficient_attention: bool = True,
        stabilize_with_global: bool = True,
        scale_factor_exponent: int = 10,
        model_path: str = "Datadog/Toto-Open-Base-1.0",
        device=None,
    ):
        if _check_soft_dependencies("toto-ts", severity="error"):
            from toto.model.toto import Toto

        if _check_soft_dependencies("torch", severity="error"):
            import torch

        self.model_path = model_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.stabilize_with_global = stabilize_with_global
        self.scale_factor_exponent = scale_factor_exponent
        if prediction_type.lower() not in ["mean", "median"]:
            raise ValueError("prediction_type must be either 'mean' or 'median'")
        else:
            self.prediction_type = prediction_type.lower()

        self.toto_model = Toto.from_pretrained(model_path)
        self.toto_model.to(self.device)
        self.toto_model.compile()
        super().__init__()

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
        if _check_soft_dependencies("toto-ts", severity="error"):
            from toto.data.util.dataset import MaskedTimeseries
            from toto.inference.forecaster import TotoForecaster

        if _check_soft_dependencies("torch", severity="error"):
            import torch

        if isinstance(y, pd.DataFrame):
            self._y = y
            self._input_series = torch.tensor(y.values.T, dtype=torch.float32).to(
                self.device
            )
            self.id_mask = torch.zeros_like(self._input_series).to(self.device)
            self.padding_mask = torch.full_like(
                self._input_series, True, dtype=torch.bool
            ).to(self.device)

        else:
            self._y = y.reset_index().pivot(
                index="time_stamp", columns="variable", values="value"
            )
            self._y = self._y.rename_axis(None, axis=1).rename_axis(
                "time_stamp", axis=0
            )
            self._input_series = torch.tensor(self._y.values.T, dtype=torch.float32).to(
                self.device
            )
            n, d = self._y.shape
            self.id_mask = torch.tensor(
                self._y.index.get_level_values("id_mask")
            ).reshape(n, d)
            self.padding_mask = torch.tensor(
                self._y.index.get_level_values("padding_mask")
            ).reshape(n, d)

        self._y_columns = self._y.columns
        self._cutoff = self._y.index[-1]

        # current model does not use these two variable, might be needed in future.
        self.timestamp_seconds = torch.zeros_like(self._input_series)
        self.time_interval_seconds = torch.full(
            (self._input_series.shape[0],), 60 * 15, dtype=torch.float32
        ).to(self.device)

        self._series = MaskedTimeseries(
            series=self._input_series,
            padding_mask=self.padding_mask,
            id_mask=self.id_mask,
            timestamp_seconds=self.timestamp_seconds,
            time_interval_seconds=self.time_interval_seconds,
        )
        self._forecaster = TotoForecaster(self.toto_model.model)

        return self

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
        if min(fh) < 1:
            raise ValueError("Forecasting horizon must contain strictly future steps.")
        prediction_length = max(fh)

        self._forecast = self._forecaster.forecast(
            self._series,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
        )
        if self.prediction_type == "median":
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
        if _check_soft_dependencies("torch", severity="error"):
            import torch

        self._predict(fh)
        var_names = self._y_columns
        cols_idx = pd.MultiIndex.from_product([var_names, alpha])
        pred_index = fh.to_absolute(self._cutoff)._values
        relative_indices = fh.to_relative(self._cutoff) - 1

        pred_quantiles = pd.DataFrame(index=pred_index, columns=cols_idx)
        alpha_tensor = torch.tensor(alpha, device=self.device)

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

    # todo: consider implementing this, optional
    # implement only if different from default:
    #   default retrieves all self attributes ending in "_"
    #   and returns them with keys that have the "_" removed
    # if not implementing, delete the method
    #   avoid overriding get_fitted_params
    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        # implement here
        #
        # when this function is reached, it is already guaranteed that self is fitted
        #   this does not need to be checked separately
        #
        # parameters of components should follow the sklearn convention:
        #   separate component name from parameter name by double-underscore
        #   e.g., componentname__paramname

    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of Time-MoE for forecasting."""

__author__ = ["Maple728"]


import torch
from transformers import AutoModelForCausalLM

from sktime.forecasting.base._base import BaseForecaster


class TimeMoE(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    context_length : int, optional (default=7)
        Length of context window for generating forecasts.
    prediction_length : int, optional (default=7)
        Length of the forecast window.
    test_size : int, optional (default=168)
        Size of test dataset.
    model_size : str, optional (default='50M')
        Size of the model to use ('50M' or '200M').
    device : str, optional (default='cpu')
        Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        context_length=7,
        prediction_length=7,
        test_size=168,
        model_size="50M",
        device="cpu",
    ):
        model = AutoModelForCausalLM.from_pretrained(
            f"Maple728/TimeMoE-{model_size}", device_map=device, trust_remote_code=True
        )
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.test_size = test_size
        self.model_size = model_size
        self.device = device
        self.model = model
      
    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : sktime time series object
            Time series to which to fit the forecaster.
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : sktime time series object, optional (default=None)
            Exogenous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        data = torch.tensor(y.values, dtype=torch.float32)

        all_predictions = []

        with torch.no_grad():
            for i in range(
                0, self.test_size - self.prediction_length + 1, self.prediction_length
            ):
                # Get sequence for current window
                start_idx = len(data) - self.test_size + i - self.context_length
                sequence = data[start_idx : start_idx + self.context_length]
                sequence = sequence.unsqueeze(0)  # Add batch dimension

                # Normalize sequence
                mean = sequence.mean(dim=-1, keepdim=True)
                std = sequence.std(dim=-1, keepdim=True)
                normalized_sequence = (sequence - mean) / std

                # Generate forecast
                output = self.model.generate(
                    normalized_sequence,
                    max_new_tokens=self.prediction_length,
                    # position_ids=torch.arange(sequence.size(-1), device=self.device).unsqueeze(0)  # Example for custom position IDs
                )

                # Denormalize predictions
                normed_preds = output[:, -self.prediction_length :]
                predictions = normed_preds * std + mean
                all_predictions.append(predictions.squeeze(0))

                self.all_predictions = all_predictions

    def _predict(self, fh, X):
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
        # model = self.model
        # device = self.device
        # prediction_length = self.prediction_length

        # outputs = model.generate(
        #     inputs=batch["inputs"].to(device).to(model.dtype),
        #     max_new_tokens=prediction_length,
        # )
        # preds = outputs[:, -prediction_length:]
        # labels = batch["labels"].to(device)
        # if len(preds.shape) > len(labels.shape):
        #     labels = labels[..., None]
        # return preds, labels

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        # _y = _y if self._global_forecasting else self._y

        # multi-index conversion goes here
        # if isinstance(_y.index, pd.MultiIndex):
        #     hist = _frame2numpy(_y)
        # else:
        #     hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (batch_size, n_timestamps, n_cols)

        # h = 7

        # timemoe_preds_50M = self._fit(

        #     # target_column='y',
        #     context_length=6*h,
        #     prediction_length=fh,
        #     test_size=168,
        #     device='cpu'
        # )
        # return timemoe_preds_50M
        return torch.cat(self.all_predictions).numpy()

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



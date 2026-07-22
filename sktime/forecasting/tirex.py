# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# Uses TiRex (https://github.com/NX-AI/tirex),
# licensed under the NXAI Community License.
# Copyright NXAI GmbH. All Rights Reserved.
# This interface wraps the public TiRex forecasting model via pip dependency,
# respecting the license and attribution terms in section 1.b of the license.
"""
Module implements TiRexForecaster, a zero shot time series forecasting model.

It wraps the TiRex foundation model. Link is "https://github.com/NX-AI/tirex".
This is use for with the sktime forecasting interface.
TiRex provides fast and proper forecasting for short and long horizons.
It does not require any training or data input.
"""

__author__ = ["sinemkilicdere", "martinloretzzz"]

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)


class TiRexForecaster(BaseFoundationForecaster):
    """Interface to the TiRex Zero-Shot Forecaster.

    This forecaster loads the TiRex model from the ``tirex-ts`` package when fit() is
    called. Instead of training, it takes the given data as context, and predict()
    uses that context to produce forecasts for the requested future time points.
    ``torch`` is required at runtime and a clear error is raised if unavailable.

    Parameters
    ----------
    model : str (default = "NX-AI/TiRex")
        "Model identifier to load via the vendored TiRex loader"
    device : {"cpu", "cuda", ...}, default="cpu"
        Compute device used by the underlying TiRex model.
    license_accepted : bool, default=False
        Whether the user accepts the license terms of TiRex.
        Must be set to True to use the model.

    Attributes
    ----------
    model_handle_ : ModelHandle
        Shared handle containing the loaded TiRex model.

    References
    ----------
    [1] TiRex: https://github.com/NX-AI/tirex

    Example
    ----------
    >>> # Minimal usage (doctest is skipped since it requires torch and model files)
    >>> import pandas as pd
    >>> from sktime.forecasting.tirex import TiRexForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = pd.Series([1, 2, 3, 4, 5])
    >>> fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    >>> f = TiRexForecaster()  # doctest: +SKIP
    >>> f.fit(y)               # doctest: +SKIP
    >>> y_pred = f.predict(fh) # doctest: +SKIP
    """

    _tags = {
        # packaging tags
        # --------------
        "maintainers": ["sinemkilicdere", "martinloretzzz"],
        "authors": [
            "martinloretzzz",
            "apointa",
            "superbock",
            "sinemkilicdere",
        ],
        "python_dependencies": ["torch", "tirex-ts"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        model="NX-AI/TiRex",
        device: str = "cpu",
        license_accepted: bool = False,
    ):
        self.model = model
        self.device = device
        self.license_accepted = license_accepted
        model_spec = FoundationModelSpec(
            model_path=model,
            device=device,
            load_extra_kwargs={"backend": "torch"},
        )
        super().__init__(model_spec=model_spec)

        if not self.license_accepted:
            raise ValueError(
                "Use of TiRexForecaster is subject to the license terms of TiRex, "
                "licensed to third party vendor NXAI GmbH. "
                "This license is not permissive, and differs from the sktime license. "
                "You must accept the license terms of TiRex to use TiRexForecaster. "
                "To accept the license, set the `license_accepted` parameter to True "
                "to confirm that you have read and accepted the license terms. "
                "To print and view the license for TiRex, "
                "call `TiRexForecaster.print_license()`"
            )

    @classmethod
    def print_license(self):
        """Print the license terms of TiRex."""
        import importlib

        dist = importlib.metadata.distribution("tirex-ts")
        license_text = dist.read_text("licenses/LICENSE")
        print(license_text)

    def _load_model(self):
        """Load the TiRex backend into the shared model cache."""
        from tirex import load_model

        model_spec = self.model_spec_
        model = load_model(
            model_spec.model_path,
            device=model_spec.device,
            **model_spec.load_extra_kwargs,
        )
        return ModelHandle(model=model)

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        """Forecast a complete future horizon with the TiRex backend."""
        import torch

        relative_fh = fh.to_relative(self.cutoff).to_pandas()
        if not all(value > 0 for value in relative_fh):
            pred_len = len(relative_fh)

        context = torch.as_tensor(
            context_y.iloc[:, 0].to_numpy()[None, :],
            dtype=torch.float32,
        )
        forecast = handle.model.forecast(
            context=context,
            prediction_length=pred_len,
        )

        if isinstance(forecast, (list, tuple)):
            forecast = forecast[1]
        if hasattr(forecast, "detach"):
            forecast = forecast.detach().cpu().numpy()

        values = forecast.reshape(-1)[:pred_len]
        return ForecastResult(mean=values.reshape(-1, 1))

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
            Parameters to create testing instances of the class.
        """
        params1 = {"model": "NX-AI/TiRex", "device": "cpu", "license_accepted": True}
        params2 = {"model": "NX-AI/TiRex", "device": "cpu", "license_accepted": True}
        return [params1, params2]

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
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton

if _check_soft_dependencies("torch", severity="none"):
    import torch
else:

    class torch:
        """Dummy class if torch is unavailable."""

        bfloat16 = None

        class Tensor:
            """Dummy class if torch is unavailable."""


def _tirex_cache_key(model: str, device: str) -> str:
    """Create a deterministic cache key for the TiRex model."""
    model_str = str(model)
    device_str = str(device)
    cache_key = "_".join([model_str, device_str])
    return cache_key


@_multiton
class _cached_TiRex:
    """Cached TiRex loader; ensures one memory instance per unique key."""

    def __init__(self, key: str, model: str, device: str):
        self.key = key
        self.model = model
        self.device = device
        self._obj = None

    def load(self):
        from tirex import load_model

        if self._obj is None:
            self._obj = load_model(self.model, device=self.device, backend="torch")
        return self._obj


class TiRexForecaster(BaseForecaster):
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
    model_ : object
        The loaded TiRex model instance (vendored implementation).

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
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
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

        self.model_ = None

        # leave this as is
        super().__init__()

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

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Loads and caches the underlying TiRex model instance (no training).

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
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
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : TiRexForecaster
            Fitted forecaster (with ``model_`` set).
        """
        key = _tirex_cache_key(self.model, self.device)
        self.model_ = _cached_TiRex(
            key=key, model=self.model, device=self.device
        ).load()
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Converts the observed series to a tensor context and delegates multi-step
        prediction to the TiRex model. The current implementation does not use ``X``.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_predict : sktime time series object
            Point forecasts, same type as seen in _fit (as in "y_inner_mtype" tag).
        """
        # implement here

        y = self._y
        context_values = y.to_numpy()[None, :]

        context_tensor = torch.as_tensor(context_values, dtype=torch.float32)

        predict_len = len(fh)

        forecast = self.model_.forecast(
            context=context_tensor, prediction_length=predict_len
        )

        if isinstance(forecast, (list, tuple)):
            forecast = forecast[1]

        if hasattr(forecast, "detach"):
            forecast = forecast.detach().cpu().numpy()

        yhat = forecast.reshape(-1)[: len(fh)]

        index = fh.to_absolute(self.cutoff).to_pandas()

        return pd.Series(
            yhat, index=index, name=(y.name if hasattr(y, "name") else None)
        )

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

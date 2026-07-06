# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# Uses DynaMix (https://github.com/DurstewitzLab/DynaMix-python),
# licensed under the GNU General Public License v3.0 (GPL-3.0).
# Pretrained weights are distributed at
# https://huggingface.co/DurstewitzLab/dynamix (CC-BY-4.0).
# The DynaMix inference code is vendored (partial fork) into
# ``sktime.libs.dynamix`` under its original GPL-3.0 license; see
# ``sktime/libs/dynamix/LICENSE``. DynaMix is copyleft software and its license
# differs from sktime's permissive BSD-3 license.
"""
Module implements DynaMixForecaster, a zero-shot time series forecasting model.

It wraps the DynaMix foundation model from DurstewitzLab. Link is
"https://github.com/DurstewitzLab/DynaMix-python". DynaMix is a mixture-of-experts
dynamical-systems foundation model that produces zero-shot, multivariate forecasts
which preserve long-term statistics. It requires no training or fine-tuning.
"""

__author__ = ["yash-sangwan"]

__all__ = ["DynaMixForecaster"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

torch = _safe_import("torch")


def _dynamix_cache_key(model: str, device: str) -> str:
    """Create a deterministic cache key for the DynaMix model."""
    model_str = str(model)
    device_str = str(device)
    cache_key = "_".join([model_str, device_str])
    return cache_key


@_multiton
class _cached_DynaMix:
    """Cached DynaMix loader; ensures one memory instance per unique key.

    DynaMix is a zero-shot, immutable model, so a single in-memory instance can
    be safely shared across all forecasters that use the same ``(model, device)``
    configuration. The weights themselves are downloaded once by the underlying
    ``load_hf_model`` call and reused from the local Hugging Face cache
    (``~/.cache/huggingface``) on subsequent loads.
    """

    def __init__(self, key: str, model: str, device: str):
        self.key = key
        self.model = model
        self.device = device
        self._obj = None

    def load(self):
        from sktime.libs.dynamix.utilities.utilities import load_hf_model

        if self._obj is None:
            obj = load_hf_model(self.model)
            # move the model to the requested device if it is a torch module
            if hasattr(obj, "to"):
                try:
                    obj = obj.to(self.device)
                except (RuntimeError, ValueError):
                    pass
            self._obj = obj
        return self._obj


class DynaMixForecaster(BaseForecaster):
    """Interface to the DynaMix Zero-Shot Forecaster by DurstewitzLab.

    DynaMix is a dynamical-systems foundation model based on a mixture-of-experts
    architecture in latent space. It produces zero-shot, multivariate forecasts
    that preserve both short-term trajectory accuracy and long-term statistics,
    and is efficient enough to run on CPU. This forecaster loads a pretrained
    DynaMix model from the ``dynamix`` package when ``fit()`` is called. Instead
    of training, it uses the given series as context, and ``predict()`` produces
    forecasts for the requested future time points.

    The DynaMix inference code is vendored into ``sktime.libs.dynamix`` as a
    partial fork under its original GPL-3.0 license, which is copyleft and
    differs from sktime's permissive BSD-3 license (see
    ``sktime/libs/dynamix/LICENSE``). ``torch`` is required at runtime.

    Parameters
    ----------
    model : str, default="dynamix-3d-alrnn-v1.0"
        Identifier of the pretrained DynaMix model to load via ``load_hf_model``
        from the vendored ``sktime.libs.dynamix`` package. Available models are
        published at https://huggingface.co/DurstewitzLab/dynamix.
    device : {"cpu", "cuda", ...}, default="cpu"
        Compute device used by the underlying DynaMix model.
    preprocessing_method : str, default="delay_embedding"
        Preprocessing method passed to the DynaMix forecaster, used to embed the
        observed series into the model's latent dynamical space.
    standardize : bool, default=True
        Whether the DynaMix forecaster standardizes the context series prior to
        forecasting.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the forecasts. DynaMix is stochastic and draws
        from the global ``torch`` RNG; if set, the RNG is seeded before each
        ``predict`` call so that repeated forecasts are reproducible (the model is
        derandomized). If ``None``, forecasts are non-deterministic across calls.
    license_accepted : bool, default=False
        Whether the user accepts the GPL-3.0 license terms of DynaMix.
        Must be set to ``True`` to use the model.

    Attributes
    ----------
    model_ : object
        The loaded DynaMix model instance.

    References
    ----------
    .. [1] DynaMix (Python): https://github.com/DurstewitzLab/DynaMix-python
    .. [2] Pretrained weights: https://huggingface.co/DurstewitzLab/dynamix

    Examples
    --------
    >>> # doctest is skipped since it requires torch and the GPL ``dynamix`` package
    >>> import pandas as pd
    >>> from sktime.forecasting.dynamix import DynaMixForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    >>> fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    >>> f = DynaMixForecaster(license_accepted=True)  # doctest: +SKIP
    >>> f.fit(y)                                       # doctest: +SKIP
    >>> y_pred = f.predict(fh)                         # doctest: +SKIP
    """

    _tags = {
        # packaging tags
        # --------------
        "authors": ["yash-sangwan"],
        "maintainers": ["yash-sangwan"],
        "python_dependencies": [
            "torch",
            "huggingface_hub",
            "safetensors",
            "statsmodels",
        ],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "property:randomness": "derandomized",
        "capability:random_state": True,
        # CI and test flags
        # -----------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.dynamix"],
    }

    def __init__(
        self,
        model="dynamix-3d-alrnn-v1.0",
        device: str = "cpu",
        preprocessing_method: str = "delay_embedding",
        standardize: bool = True,
        random_state=None,
        license_accepted: bool = False,
    ):
        self.model = model
        self.device = device
        self.preprocessing_method = preprocessing_method
        self.standardize = standardize
        self.random_state = random_state
        self.license_accepted = license_accepted

        self.model_ = None

        # leave this as is
        super().__init__()

        if not self.license_accepted:
            raise ValueError(
                "Use of DynaMixForecaster is subject to the GPL-3.0 license terms "
                "of DynaMix, developed by DurstewitzLab. "
                "This license is copyleft and not permissive, and differs from the "
                "permissive sktime BSD-3 license. "
                "You must accept the license terms of DynaMix to use "
                "DynaMixForecaster. "
                "To accept the license, set the `license_accepted` parameter to True "
                "to confirm that you have read and accepted the license terms. "
                "To print and view the license for DynaMix, "
                "call `DynaMixForecaster.print_license()`"
            )

    @classmethod
    def print_license(cls):
        """Print the license terms of DynaMix."""
        import importlib.metadata

        try:
            dist = importlib.metadata.distribution("dynamix")
            license_text = None
            # try the common locations for the GPL license file
            for candidate in ("LICENSE", "LICENSE.txt", "COPYING"):
                try:
                    license_text = dist.read_text(candidate)
                except (FileNotFoundError, OSError):
                    license_text = None
                if license_text is not None:
                    break
            if license_text is not None:
                print(license_text)
                return
        except importlib.metadata.PackageNotFoundError:
            pass

        print(
            "DynaMix is licensed under the GNU General Public License v3.0 "
            "(GPL-3.0).\n"
            "The full license text is available at "
            "https://www.gnu.org/licenses/gpl-3.0.txt\n"
            "See also https://github.com/DurstewitzLab/DynaMix-python."
        )

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Loads and caches the underlying DynaMix model instance (no training).

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : DynaMixForecaster
            Fitted forecaster (with ``model_`` set).
        """
        key = _dynamix_cache_key(self.model, self.device)
        self.model_ = _cached_DynaMix(
            key=key, model=self.model, device=self.device
        ).load()
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        Converts the observed series to a context tensor of shape
        ``(T_C, S, N)`` (context length, samples, variables) and delegates
        multi-step prediction to the DynaMix forecaster. ``X`` is not used.

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored).

        Returns
        -------
        y_predict : pd.DataFrame
            Point forecasts, same columns as the series seen in ``_fit``.
        """
        # alias the vendored forecaster to avoid the name collision with this class
        from sktime.libs.dynamix.model.forecaster import (
            DynaMixForecaster as _DynaMixModel,
        )

        y = self._y

        # build context of shape (T_C, S, N): context length, 1 sample, n variables
        context_values = y.to_numpy()
        context_values = context_values[:, None, :]
        context_tensor = torch.as_tensor(
            context_values, dtype=torch.float32, device=self.device
        )

        predict_len = len(fh)

        # DynaMix is stochastic; seed the global torch RNG for reproducibility
        # when a random_state is provided (derandomization).
        if self.random_state is not None:
            from sklearn.utils import check_random_state

            rng = check_random_state(self.random_state)
            torch.manual_seed(int(rng.randint(0, 2**31 - 1)))

        forecaster = _DynaMixModel(self.model_)
        with torch.no_grad():
            forecast = forecaster.forecast(
                context=context_tensor,
                horizon=predict_len,
                preprocessing_method=self.preprocessing_method,
                standardize=self.standardize,
            )

        # some interfaces return a (states, observations) tuple; take observations
        if isinstance(forecast, (list, tuple)):
            forecast = forecast[-1]

        if hasattr(forecast, "detach"):
            forecast = forecast.detach().cpu().numpy()

        n_cols = y.shape[1]
        # collapse any sample dimension and keep the last (T_C, N) -> (horizon, N)
        forecast = forecast.reshape(-1, n_cols)[:predict_len]

        index = fh.to_absolute(self.cutoff).to_pandas()

        return pd.DataFrame(forecast, index=index, columns=y.columns)

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
        params1 = {
            "model": "dynamix-3d-alrnn-v1.0",
            "device": "cpu",
            "random_state": 42,
            "license_accepted": True,
        }
        params2 = {
            "model": "dynamix-3d-alrnn-v1.0",
            "device": "cpu",
            "standardize": False,
            "random_state": 42,
            "license_accepted": True,
        }
        return [params1, params2]

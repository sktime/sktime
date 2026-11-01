# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface to PyPOTS (Partially Observed Time Series) imputation models."""

__author__ = ["Spinachboul", "jgyasu"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.dependencies import _check_estimator_deps

_PYPOTS_MODEL_MAP = {
    "Autoformer": "pypots.imputation.Autoformer",
    "BRITS": "pypots.imputation.BRITS",
    "CSAI": "pypots.imputation.CSAI",
    "CSDI": "pypots.imputation.CSDI",
    "Crossformer": "pypots.imputation.Crossformer",
    "DLinear": "pypots.imputation.DLinear",
    "ETSformer": "pypots.imputation.ETSformer",
    "FEDformer": "pypots.imputation.FEDformer",
    "FiLM": "pypots.imputation.FiLM",
    "FreTS": "pypots.imputation.FreTS",
    "GPT4TS": "pypots.imputation.GPT4TS",
    "GPVAE": "pypots.imputation.GPVAE",
    "GRUD": "pypots.imputation.GRUD",
    "ImputeFormer": "pypots.imputation.ImputeFormer",
    "Informer": "pypots.imputation.Informer",
    "Koopa": "pypots.imputation.Koopa",
    "LOCF": "pypots.imputation.LOCF",
    "Lerp": "pypots.imputation.Lerp",
    "MICN": "pypots.imputation.MICN",
    "MOMENT": "pypots.imputation.MOMENT",
    "MRNN": "pypots.imputation.MRNN",
    "Mean": "pypots.imputation.Mean",
    "Median": "pypots.imputation.Median",
    "ModernTCN": "pypots.imputation.ModernTCN",
    "NonstationaryTransformer": "pypots.imputation.NonstationaryTransformer",
    "PatchTST": "pypots.imputation.PatchTST",
    "Pyraformer": "pypots.imputation.Pyraformer",
    "Reformer": "pypots.imputation.Reformer",
    "RevIN_SCINet": "pypots.imputation.RevIN_SCINet",
    "SAITS": "pypots.imputation.SAITS",
    "SCINet": "pypots.imputation.SCINet",
    "SegRNN": "pypots.imputation.SegRNN",
    "StemGNN": "pypots.imputation.StemGNN",
    "TCN": "pypots.imputation.TCN",
    "TEFN": "pypots.imputation.TEFN",
    "TOTEM": "pypots.imputation.TOTEM",
    "TRMF": "pypots.imputation.TRMF",
    "TSLANet": "pypots.imputation.TSLANet",
    "TiDE": "pypots.imputation.TiDE",
    "TimeLLM": "pypots.imputation.TimeLLM",
    "TimeMixer": "pypots.imputation.TimeMixer",
    "TimeMixerPP": "pypots.imputation.TimeMixerPP",
    "TimesNet": "pypots.imputation.TimesNet",
    "Transformer": "pypots.imputation.Transformer",
    "USGAN": "pypots.imputation.USGAN",
}


class PyPOTSImputer(BaseTransformer):
    """Interface to PyPOTS (Partially Observed Time Series) imputation models.

    PyPOTS is a Python library for data mining on partially-observed time series [1]_.

    This transformer wraps multiple imputation models from PyPOTS, including
    deep learning models (SAITS, BRITS, etc.) and statistical models (Lerp, LOCF).

    Parameters
    ----------
    model : str, default="SAITS"
        The name of the PyPOTS model to use.
        Valid values are keys of _PYPOTS_MODEL_MAP, e.g., "SAITS", "BRITS", "Lerp", "LOCF".
    model_params : dict, default=None
        Parameters passed to the PyPOTS model constructor.
        Note: n_steps and n_features are automatically inferred from the data.
    device : str, torch.device, or list, default=None
        The device to run the model on. If None, PyPOTS default is used.
    val_set : dict or str, default=None
        Validation set for neural network training, passed to model.fit().

    Examples
    --------
    >>> from sktime.transformations.series.impute import PyPOTSImputer
    >>> from sktime.datasets import load_airline
    >>> import numpy as np
    >>> y = load_airline()
    >>> y.iloc[5:10] = np.nan
    >>> transformer = PyPOTSImputer(model="Lerp")
    >>> y_hat = transformer.fit_transform(y)

    References
    ----------
    .. [1] Wenjie Du. PyPOTS: A Python Toolbox for Data Mining on Partially-Observed
           Time Series. arXiv:2305.18811, 2023.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Spinachboul", "jgyasu"],
        "maintainers": ["Spinachboul", "jgyasu"],
        "python_dependencies": ["pypots"],
        "python_version": ">=3.8",
        # estimator tags
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:missing_values:removes": True,
        "fit_is_empty": False,
        # testing configuration
        # ---------------------
        "tests:vm": True,
    }

    def __init__(
        self,
        model="SAITS",
        model_params=None,
        device=None,
        val_set=None,
    ):
        self.model = model
        self.model_params = model_params
        self.device = device
        self.val_set = val_set

        super().__init__()

        if model not in _PYPOTS_MODEL_MAP:
            raise ValueError(
                f"Invalid model name: {model}. "
                f"Available models: {list(_PYPOTS_MODEL_MAP.keys())}"
            )

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit transform to
        y : ignored argument for interface compatibility

        Returns
        -------
        self: reference to self
        """
        _check_estimator_deps(self)

        model_path = _PYPOTS_MODEL_MAP[self.model]
        module_path, class_name = model_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)

        params = self.model_params.copy() if self.model_params else {}
        if self.device is not None:
            params["device"] = self.device

        # Infer n_steps and n_features if the model expects them
        import inspect

        sig = inspect.signature(model_class.__init__)
        if "n_steps" in sig.parameters:
            params["n_steps"] = X.shape[0]
        if "n_features" in sig.parameters:
            params["n_features"] = X.shape[1]

        self.model_ = model_class(**params)

        # PyPOTS expects 3D numpy array [n_samples, n_steps, n_features]
        # Since instancewise=True, X is a single instance
        X_3d = X.to_numpy()[np.newaxis, :, :]

        # Some models don't need fit (statistical models like Lerp, LOCF)
        # But we call it anyway as per PyPOTS API if it's not a no-op
        if self.model not in ["Lerp", "LOCF", "Mean", "Median"]:
            self.model_.fit(train_set={"X": X_3d}, val_set=self.val_set)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        transformed version of X
        """
        _check_estimator_deps(self)

        # PyPOTS expects 3D numpy array [n_samples, n_steps, n_features]
        X_3d = X.to_numpy()[np.newaxis, :, :]

        # results is a dict containing 'imputation' which is [n_samples, n_steps, n_features]
        results = self.model_.predict(test_set={"X": X_3d})
        imputed_X_3d = results["imputation"]

        # Convert back to 2D and then to DataFrame
        imputed_X_2d = imputed_X_3d[0]
        Xt = pd.DataFrame(imputed_X_2d, index=X.index, columns=X.columns)

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
        """
        return [
            {"model": "Lerp"},
            {"model": "LOCF"},
        ]

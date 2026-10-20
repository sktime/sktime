# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Timestral forecaster for ``sktime``.

Wraps Mistral AI Timestral foundation model via the upstream ``timestral`` package.
"""

__all__ = ["TimestralForecaster"]

from sktime.forecasting.base import BaseForecaster

_LICENSE_URL = "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE"


class TimestralForecaster(BaseForecaster):
    """Interface to Mistral AI Timestral zero-shot forecaster.

    Timestral is an open-weights time series foundation model family developed
    by Mistral AI, trained using Decoupled Horizon Forecasting (DHF). It
    supports native multivariate forecasting, past-only and known-future covariates,
    categorical covariates, missing values, and probabilistic forecasting with 21
    quantiles.

    Parameters
    ----------
    model_path : str, default="mistralai/Timestral-209M"
        Hugging Face repository id or local directory containing model weights and config.
    device : str or None, default=None
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
    max_context : int or None, default=None
        Maximum context window length. Defaults to the model trained context length
        (up to 32,768).
    symmetric_averaging : bool, default=True
        Whether to enable upstream test-time symmetric reflection augmentation.
    clip_nonnegative : bool, default=True
        Whether to clamp forecasts to 0 for series with strictly non-negative context.
    license_accepted : bool, default=False
        Must be ``True`` to use pretrained weights distributed under Mistral AI's
        Modified MIT license. Call ``TimestralForecaster.print_license()`` for details.
    ignore_deps : bool, default=False
        If ``True``, skip soft-dependency checks (for testing).

    References
    ----------
    .. [1] https://github.com/mistralai/timestral
    .. [2] https://mistral.ai/news/timestral/
    """

    _tags = {
        "category": ["deep_learning", "foundation_model"],
        "authors": ["hasanfaesal"],
        "maintainers": ["hasanfaesal"],
        "python_dependencies": ["timestral", "torch>=2.4.0"],
        "capability:multivariate": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "capability:categorical_in_X": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_timestral"],
    }

    def __init__(
        self,
        model_path: str = "mistralai/Timestral-209M",
        device: str | None = None,
        max_context: int | None = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
        license_accepted: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.max_context = max_context
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative
        self.license_accepted = license_accepted
        self.ignore_deps = ignore_deps

        self.forecaster = None

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters."""
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __getstate__(self):
        """Return state for pickling, excluding the unpickleable upstream model."""
        state = self.__dict__.copy()
        if hasattr(self, "forecaster"):
            state["forecaster"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    @classmethod
    def print_license(cls):
        """Print license information for Timestral pretrained weights."""
        print(
            "Timestral pretrained weights are distributed under the Mistral Modified "
            "MIT license, which requires a commercial license agreement if the global "
            "consolidated monthly revenue exceeds $20M for the preceding month."
        )
        print(f"Full license text: {_LICENSE_URL}")

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at specified horizon."""
        raise NotImplementedError("Work in progress.")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        params1 = {
            "model_path": "mistralai/Timestral-5M",
            "license_accepted": True,
            "ignore_deps": True,
        }
        params2 = {
            "model_path": "mistralai/Timestral-5M",
            "symmetric_averaging": False,
            "clip_nonnegative": False,
            "license_accepted": True,
            "ignore_deps": True,
        }
        return [params1, params2]

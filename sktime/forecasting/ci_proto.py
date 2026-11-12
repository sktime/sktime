# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Zero-shot forecasting with Continuous Intelligence's ci-proto."""

__all__ = ["CIProtoForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.singleton import _multiton


class CIProtoForecaster(BaseForecaster):
    """Interface to the ci-proto probabilistic forecasting model.

    The model forecasts all target columns jointly and returns nine quantiles.
    The 0.5 quantile supplies point forecasts. Covariate columns supplied only
    during fit are past covariates; columns supplied at fit and predict time are
    known future covariates. Future covariates must cover every step through the
    largest requested horizon, including unrequested intermediate steps.

    Parameters
    ----------
    model_path : str, default="ContinuousIntelligence/ci-proto"
        Hugging Face repository ID or local model bundle.
    revision : str or None, default=None
        Hugging Face commit SHA to pin the model. Applies to Hub models only.
    device : {"auto", "cpu", "cuda"}, default="auto"
        Inference device. Use ``"cpu"`` for portable inference.

    Notes
    -----
    ``predict_proba`` interpolates between the native quantiles and clamps
    requests outside the 0.1 to 0.9 range to the nearest available quantile.

    References
    ----------
    .. [1] https://github.com/continuousintelligence/ci

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.ci_proto import CIProtoForecaster
    >>> y = load_airline()
    >>> forecaster = CIProtoForecaster(device="cpu")  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    >>> q = forecaster.predict_quantiles(  # doctest: +SKIP
    ...     fh=[1, 2, 3], alpha=[0.1, 0.5, 0.9]
    ... )
    """

    _tags = {
        "category": ["deep_learning", "foundation_model"],
        "authors": ["KrishChadha001"],
        "maintainers": ["KrishChadha001"],
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": True,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "python_version": ">=3.11,<3.15",
        "python_dependencies": ["continuousintelligence", "skpro>=2.14"],
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_ci_proto"],
        "shorthands": {"ci_proto": "CIProtoForecaster()"},
    }

    def __init__(
        self,
        model_path="ContinuousIntelligence/ci-proto",
        revision=None,
        device="auto",
    ):
        self.model_path = model_path
        self.revision = revision
        self.device = device
        super().__init__()

    def _fit(self, y, X, fh):
        self._cur_y = y
        self._cur_X = X
        self.model_ = _CachedCIProto(
            key=(str(self.model_path), self.revision, self.device),
            model_path=self.model_path,
            revision=self.revision,
            device=self.device,
        ).load()
        return self

    def _forecast(self, fh, X):
        import torch

        steps = np.asarray(fh.to_relative(self.cutoff), dtype=int)
        if np.any(steps < 1):
            raise ValueError("ci-proto supports only out-of-sample horizons")
        horizon = int(steps.max())
        history = torch.as_tensor(self._cur_y.to_numpy().T, dtype=torch.float32)
        kwargs = {}

        if self._cur_X is not None:
            past_columns = []
            future_columns = []
            future_names = set() if X is None else set(X.columns)
            for column in self._cur_X.columns:
                (future_columns if column in future_names else past_columns).append(
                    column
                )
            if X is not None and set(X.columns) - set(self._cur_X.columns):
                raise ValueError("Prediction covariates must also occur in fit X")
            if past_columns:
                kwargs["past_covariates"] = torch.as_tensor(
                    self._cur_X[past_columns].to_numpy().T, dtype=torch.float32
                )
            if future_columns:
                expected = fh.to_absolute(self.cutoff)._values
                if not expected.isin(X.index).all():
                    raise ValueError("Prediction X must contain all requested horizons")
                if len(X) < horizon or not X.index.is_monotonic_increasing:
                    raise ValueError(
                        "Prediction X must cover the full forecast horizon"
                    )
                expected_full = (
                    ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
                    .to_absolute(self.cutoff)
                    ._values
                )
                if not expected_full.equals(X.index[:horizon]):
                    raise ValueError(
                        "Prediction X must contain consecutive future steps"
                    )
                values = pd.concat(
                    [self._cur_X[future_columns], X[future_columns].iloc[:horizon]]
                )
                kwargs["future_covariates"] = torch.as_tensor(
                    values.to_numpy().T, dtype=torch.float32
                )
        elif X is not None:
            raise ValueError("Prediction covariates require fit-time X")

        output = self.model_.forecast(history, horizon=horizon, **kwargs)
        levels = np.asarray(self.model_.quantiles)
        values = np.asarray(output)
        index = fh.to_absolute(self.cutoff)._values
        return levels, values[:, :, steps - 1], index

    def _predict(self, fh, X):
        levels, values, index = self._forecast(fh, X)
        median = np.flatnonzero(np.isclose(levels, 0.5))
        if len(median) != 1:
            raise ValueError("ci-proto output must contain a 0.5 quantile")
        return pd.DataFrame(
            values[:, median[0], :].T, index=index, columns=self._cur_y.columns
        )

    def _predict_proba(self, fh, X, marginal=True):
        from skpro.distributions import HistogramQPD

        levels, values, index = self._forecast(fh, X)
        row_index = pd.MultiIndex.from_product([levels, index])
        data = values.transpose(1, 2, 0).reshape(len(levels) * len(index), -1)
        quantiles = pd.DataFrame(data, index=row_index, columns=self._cur_y.columns)
        return HistogramQPD(
            quantiles, tails="mass", index=index, columns=self._cur_y.columns
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return parameters for estimator contract tests."""
        return {"device": "cpu"}


@_multiton
class _CachedCIProto:
    """Share immutable, loaded model weights by model, revision and device."""

    def __init__(self, key, model_path, revision, device):
        self.model_path = model_path
        self.revision = revision
        self.device = device
        self.model = None

    def load(self):
        """Load a model once for a given configuration."""
        if self.model is None:
            import continuousintelligence as ci

            kwargs = {"device": self.device}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            self.model = ci.load_model(self.model_path, **kwargs)
        return self.model

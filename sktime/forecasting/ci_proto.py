# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Zero-shot forecasting with Continuous Intelligence's ci-proto."""

__all__ = ["CIProtoForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.singleton import _multiton


class CIProtoForecaster(BaseForecaster):
    """Interface to the ci-proto probabilistic forecasting model.

    The model [1]_ forecasts all target columns jointly and returns nine quantiles.
    The 0.5 quantile supplies point forecasts. Covariate columns supplied only
    during fit are past covariates; columns supplied at fit and predict time are
    known future covariates. Future covariates must cover every step through the
    largest requested horizon, including unrequested intermediate steps.
    Only numerical covariates are supported by this adapter. Regular datetime
    and period indices supply timestamps and frequency for calendar features.

    Parameters
    ----------
    model_path : str, default="ContinuousIntelligence/ci-proto"
        Hugging Face repository ID or local model bundle.
    revision : str or None, default=None
        Hugging Face commit SHA to pin the model. Applies to Hub models only.
    device : {"auto", "cpu", "cuda"}, default="auto"
        Inference device. ``"auto"`` selects CUDA when PyTorch reports it is
        available and uses CPU otherwise.

    Notes
    -----
    ``predict_proba`` interpolates between the native quantiles and clamps
    requests outside the 0.1 to 0.9 range to the nearest available quantile.

    The upstream ci-proto code and model weights use an Apache-2.0-based
    licence with additional terms in Section 10 [2]_. These include public
    attribution and a one-time notification to Continuous Intelligence for
    redistribution or externally available products and services. The terms
    also cover externally released AI models developed using ci-proto code,
    weights or outputs; purely internal use is exempt from Section 10.
    Consult the upstream licence for the full terms.

    References
    ----------
    .. [1] Continuous Intelligence. ci-proto.
       https://github.com/continuousintelligence/ci
    .. [2] ci-proto Code and Model Licence, Section 10.
       https://github.com/continuousintelligence/ci/blob/main/LICENCE

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
        "authors": ["KrishChadha001", "Benjamin-Walker", "berndtt"],
        "maintainers": ["KrishChadha001"],
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": True,
        "capability:categorical_in_X": False,
        "capability:missing_values": True,
        "capability:update": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "python_version": "<3.15",
        "python_dependencies": ["continuousintelligence", "skpro>=2.14"],
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_ci_proto"],
        # Only the CPU configuration avoids requiring accelerator hardware.
        "tests:skip_by_name": ["test_get_test_params_coverage"],
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
        self._load_model()
        return self

    def _update(self, y, X=None, update_params=True):
        """Refresh forecast context while retaining the pretrained model weights."""
        from sktime.datatypes import update_data

        self._cur_y = update_data(self._cur_y, y)
        self._cur_X = update_data(self._cur_X, X)
        return self

    def _load_model(self):
        return _CachedCIProto(
            key=(str(self.model_path), self.revision, self.device),
            model_path=self.model_path,
            revision=self.revision,
            device=self.device,
        ).load()

    def _forecast(self, fh, X):
        import torch

        steps = np.asarray(fh.to_relative(self.cutoff), dtype=int)
        horizon = int(steps.max())
        full_index = (
            ForecastingHorizon(
                np.arange(1, horizon + 1), is_relative=True, freq=fh.freq
            )
            .to_absolute(self.cutoff)
            ._values
        )
        history = torch.as_tensor(self._cur_y.to_numpy().T, dtype=torch.float32)
        kwargs = {}

        if self._cur_X is not None:
            # Updates may omit X; represent unobserved covariates as missing.
            context_X = self._cur_X.reindex(self._cur_y.index)
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
                    context_X[past_columns].to_numpy().T, dtype=torch.float32
                )
            if future_columns:
                if not full_index.equals(X.index[:horizon]):
                    raise ValueError(
                        "Prediction X must contain every step up to the largest horizon"
                    )
                future = pd.concat([context_X[future_columns], X[future_columns]])
                kwargs["future_covariates"] = torch.as_tensor(
                    future.to_numpy().T, dtype=torch.float32
                )
        elif X is not None:
            raise ValueError("Prediction covariates require fit-time X")

        history_index = self._cur_y.index
        if isinstance(history_index, (pd.DatetimeIndex, pd.PeriodIndex)):
            dates = history_index.append(full_index)
            if isinstance(dates, pd.PeriodIndex):
                dates = dates.to_timestamp()
            freq = dates.freqstr
            if freq is None and len(dates) >= 3:
                freq = pd.infer_freq(dates)
            if freq is None and isinstance(history_index, pd.PeriodIndex):
                freq = history_index.freqstr
            if freq is not None:
                kwargs["freq"] = freq
                kwargs["timestamps"] = torch.as_tensor(
                    dates.as_unit("ns").asi8.copy(), dtype=torch.int64
                )

        model = self._load_model()
        output = model.forecast(history, horizon=horizon, **kwargs)
        levels = np.asarray(model.quantiles)
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
        """Build marginal distributions from the native quantile forecasts."""
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

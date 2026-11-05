# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TimeH2O zero-shot forecasting through the optional h2ofm package."""

__all__ = ["TimeH2OForecaster"]

from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd

from sktime.datatypes import update_data
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


@lru_cache(maxsize=2)
def _load_model(model_type, path, device, batch_size, options):
    """Reuse immutable inference weights across fits with identical settings."""
    if path is not None:
        return model_type.from_pretrained(
            path, device=device, batch_size=batch_size, **dict(options)
        )

    import torch
    from h2ofm._checkpoints import resolve_device
    from h2ofm.timeh2o._checkpoints import with_rope
    from h2ofm.timeh2o._model.config import TimeH2OConfig
    from h2ofm.timeh2o._model.model import TimeH2OModel
    from h2ofm.timeh2o._pipeline.pipeline import TimeH2OPipeline

    config = TimeH2OConfig(
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_heads=4,
        patch_size=4,
        context_length=32,
        max_horizon=8,
        autoregressive=True,
        lookahead_patches=1,
        flip_equivariant=False,
        variate_attention=True,
        future_covariate_embed=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TimeH2OModel(config).eval().to(resolve_device(device))
    return model_type(TimeH2OPipeline(with_rope(model)), batch_size=batch_size)


class TimeH2OForecaster(BaseForecaster):
    """Interface to H2O.ai's TimeH2O forecasting foundation model.

    TimeH2O [1]_ is a pretrained model for zero-shot probabilistic forecasting.
    ``fit`` loads the checkpoint and stores context; no weights are trained.
    Point predictions are the upstream median. Quantile predictions use the
    upstream interpolation and Gaussian-tail extrapolation for requested levels
    outside the trained 0.1--0.9 grid. Multiple targets are forecast from their
    own histories, as in the upstream API [2]_. Panels are vectorized by sktime.

    All columns of ``X`` are treated as future-known covariates. Supply their
    historical values in ``fit`` and future values in ``predict`` for **every**
    step from 1 through ``max(fh)``, including gaps in a sparse horizon.
    Columns may be numeric or categorical. The default checkpoint supports
    future covariates; the tiny checkpoint ignores them with an upstream warning.
    Past-only covariates are not supported by these checkpoints.

    Install the optional dependency with ``pip install h2ofm`` (Python 3.12+).
    The default weights are gated: accept the conditions at [1]_ and authenticate
    using ``hf auth login``, ``HF_TOKEN``, or ``model_kwargs={"token": ...}``.
    Authentication and access errors from Hugging Face propagate to the caller.
    The weights use the H2O model non-commercial licence v1.0; consult [1]_ for
    the terms. Installing the Apache-2.0 ``h2ofm`` package does not grant access
    to the gated weights or change their license.

    Parameters
    ----------
    model_path : str or None, default="h2oai/timeh2o-v1"
        Hugging Face model id or local checkpoint path. ``None`` builds a small,
        deterministic model with untrained weights for offline interface tests;
        its forecasts have no predictive meaning.
    device : str, default="auto"
        Upstream device, e.g. ``"cpu"``, ``"cuda"`` or ``"auto"``.
    batch_size : int, default=256
        Maximum number of target and covariate rows processed per batch.
    model_kwargs : dict or None, default=None
        Additional ``TimeH2O.from_pretrained`` options: ``revision``, ``token``,
        ``cache_dir`` and ``local_files_only``. ``pretrained_model_name_or_path``,
        ``device`` and ``batch_size`` are reserved for the parameters above.
        Prefer environment authentication to storing a token in an estimator.
        See the `h2ofm TimeH2O API <https://pypi.org/project/h2ofm/>`_ for options.
    forecast_kwargs : dict or None, default=None
        Additional ``TimeH2O.forecast`` options: ``context_length``,
        ``autoregressive``, ``flip_equivariant``, ``clip_nonnegative``,
        ``snap_integer`` and ``patch_ensemble``. ``context``, ``horizon``,
        ``future_covariates`` and ``quantile_levels`` are managed by this adapter.
        ``past_covariates`` are unsupported. These keys must not appear here.
        See the `h2ofm TimeH2O API <https://pypi.org/project/h2ofm/>`_ for options.

    Attributes
    ----------
    model_ : h2ofm.TimeH2O
        Loaded upstream model.

    References
    ----------
    .. [1] https://huggingface.co/h2oai/timeh2o-v1
    .. [2] https://pypi.org/project/h2ofm/

    Examples
    --------
    Requires accepted Hugging Face model terms and authentication.

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timeh2o import TimeH2OForecaster
    >>> forecaster = TimeH2OForecaster(device="cpu")  # doctest: +SKIP
    >>> forecaster.fit(load_airline())  # doctest: +SKIP
    >>> forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    >>> forecaster.predict_quantiles(fh=[1, 3], alpha=[0.1, 0.5, 0.9])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["KrishChadha001", "H2O.ai"],
        "maintainers": ["KrishChadha001"],
        "category": ["deep_learning", "foundation_model"],
        "python_dependencies": ["h2ofm>=0.43.1"],
        "python_version": ">=3.12",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": True,
        "capability:categorical_in_X": True,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        # CI and test tags
        # ----------------
        "tests:specific": ["sktime.forecasting.tests.test_timeh2o"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="h2oai/timeh2o-v1",
        device="auto",
        batch_size=256,
        model_kwargs=None,
        forecast_kwargs=None,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs
        self.forecast_kwargs = forecast_kwargs
        super().__init__()

    def _fit(self, y, X, fh):
        """Load weights and retain context without training the model."""
        model_kwargs = {} if self.model_kwargs is None else deepcopy(self.model_kwargs)
        forecast_kwargs = (
            {} if self.forecast_kwargs is None else deepcopy(self.forecast_kwargs)
        )
        if self.model_path is None and model_kwargs:
            raise ValueError("model_kwargs require a pretrained model_path.")
        if "quantile_levels" in forecast_kwargs:
            raise ValueError("quantile_levels are managed by predict_quantiles.")
        if "past_covariates" in forecast_kwargs:
            raise ValueError("past_covariates are not supported by this adapter.")

        from h2ofm import TimeH2O

        self.model_ = _load_model(
            TimeH2O,
            self.model_path,
            self.device,
            self.batch_size,
            tuple(sorted(model_kwargs.items())),
        )
        self._forecast_kwargs = forecast_kwargs
        self._cur_y = y.copy()
        self._cur_X = None if X is None else X.loc[y.index].copy()
        return self

    def _update(self, y, X=None, update_params=True):
        """Append new observations; pretrained weights remain fixed."""
        if self._cur_X is not None:
            if X is None or set(X.columns) != set(self._cur_X.columns):
                raise ValueError("Update X must contain the same columns as fit X.")
            X = X.loc[y.index, self._cur_X.columns]
        elif X is not None:
            raise ValueError("X was supplied in update but not in fit.")
        self._cur_y = update_data(self._cur_y, y)
        self._cur_X = update_data(self._cur_X, X)
        return self

    def _run_forecast(self, fh, X, alpha=None):
        """Forecast the full future span, preserving per-covariate dtypes."""
        relative = fh.to_relative(self.cutoff)
        horizon = int(max(relative))
        future_index = ForecastingHorizon(
            range(1, horizon + 1), is_relative=True
        ).to_absolute_index(self.cutoff)
        covariates = None
        if self._cur_X is not None:
            if X is None:
                raise ValueError("Future X is required when X was supplied in fit.")
            if set(X.columns) != set(self._cur_X.columns):
                raise ValueError("Future X must contain the same columns as fit X.")
            if not future_index.isin(X.index).all():
                raise ValueError("Future X must cover every step through max(fh).")
            covariates = []
            for col in self._cur_X.columns:
                past = self._cur_X[col]
                future = X.loc[future_index, col]
                if past.dtype != future.dtype and (
                    isinstance(past.dtype, pd.CategoricalDtype)
                    or isinstance(future.dtype, pd.CategoricalDtype)
                ):
                    past, future = past.astype(object), future.astype(object)
                covariates.append(pd.concat([past, future]))
        elif X is not None:
            raise ValueError("X was supplied in predict but not in fit.")

        kwargs = dict(self._forecast_kwargs)
        if alpha is not None:
            kwargs["quantile_levels"] = list(alpha)
        output = self.model_.forecast(
            self._cur_y.to_numpy(dtype=np.float32, na_value=np.nan).T,
            horizon=horizon,
            future_covariates=covariates,
            **kwargs,
        )
        index = fh.to_absolute_index(self.cutoff)
        index.names = self._cur_y.index.names
        return output, relative.to_indexer(), index

    def _predict(self, fh, X):
        """Return the median for each requested horizon and target."""
        output, positions, index = self._run_forecast(fh, X)
        return pd.DataFrame(
            np.asarray(output.point).T[positions],
            index=index,
            columns=self._cur_y.columns,
        )

    def _predict_quantiles(self, fh, X, alpha):
        """Return upstream quantiles in sktime variable-major order."""
        output, positions, index = self._run_forecast(fh=fh, X=X, alpha=alpha)
        values = np.asarray(output.quantiles).transpose(1, 0, 2)[positions]
        columns = pd.MultiIndex.from_product([self._get_varnames(), alpha])
        return pd.DataFrame(
            values.reshape(len(index), -1), index=index, columns=columns
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return CPU settings; real weight tests require gated model access."""
        return [
            {"model_path": None, "device": "cpu"},
            {
                "model_path": None,
                "device": "cpu",
                "forecast_kwargs": {"context_length": 32},
            },
        ]

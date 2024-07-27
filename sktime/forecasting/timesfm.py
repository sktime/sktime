# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TimesFM (Time Series Foundation Model)."""

__author__ = ["geetu040"]


import os

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster


class TimesFMForecaster(_BaseGlobalForecaster):
    """Implementation of TimesFM (Time Series Foundation Model).

    todo: describe your custom forecaster here

    Parameters
    ----------
    broadcasting: bool (default=True)
        multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from `predict`.
    """

    _tags = {
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": [
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": [
            "tensorflow",
            "einshape",
            "jax",
            "praxis",
            "huggingface-hub",
            "paxml",
            "utilsforecast",
        ],
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        context_len,
        horizon_len,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        per_core_batch_size=32,
        backend="cpu",
        verbose=False,
        broadcasting=True,
        freq=0,
        repo_id="google/timesfm-1.0-200m",
    ):
        super().__init__()

        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.per_core_batch_size = per_core_batch_size
        self.backend = backend
        self.verbose = verbose
        self.broadcasting = broadcasting
        self.freq = freq
        self.repo_id = repo_id

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.Series",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

        # to avoid RuntimeError
        os.environ["JAX_PLATFORM_NAME"] = backend
        os.environ["JAX_PLATFORMS"] = backend

    def _fit(self, y, X, fh):
        # import after backend env has been set
        from sktime.libs.timesfm import TimesFm

        self.tfm = TimesFm(
            context_len=self.context_len,
            horizon_len=self.horizon_len,
            input_patch_len=self.input_patch_len,
            output_patch_len=self.output_patch_len,
            num_layers=self.num_layers,
            model_dims=self.model_dims,
            per_core_batch_size=self.per_core_batch_size,
            backend=self.backend,
            verbose=self.verbose,
        )
        self.tfm.load_from_checkpoint(repo_id=self.repo_id)

    def _predict(self, fh, X, y=None):
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        _y = y if self._global_forecasting else self._y

        # multi-index conversion goes here
        if isinstance(_y.index, pd.MultiIndex):
            hist = _frame2numpy(_y).squeeze(2)
        else:
            hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (batch_size, n_timestamps)

        pred, _ = self.tfm.forecast(hist)

        # converting pred datatype

        batch_size, n_timestamps = pred.shape

        if isinstance(_y.index, pd.MultiIndex):
            ins = np.array(list(np.unique(_y.index.droplevel(-1)).repeat(n_timestamps)))
            ins = [ins[..., i] for i in range(ins.shape[-1])] if ins.ndim > 1 else [ins]

            idx = (
                ForecastingHorizon(range(1, n_timestamps + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values.tolist()
                * pred.shape[0]
            )
            index = pd.MultiIndex.from_arrays(
                ins + [idx],
                names=_y.index.names,
            )
            pred = pd.DataFrame(
                # batch_size * num_timestams
                pred.ravel(),
                index=index,
                columns=_y.columns,
            )
        else:
            index = (
                ForecastingHorizon(range(1, n_timestamps + 1))
                .to_absolute(self._cutoff)
                ._values
            )
            pred = pd.Series(
                # batch_size * num_timestams
                pred.ravel(),
                index=index,
                name=_y.name,
            )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        pred = pred.loc[dateindex]

        return pred

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
        test_params = [
            {
                "context_len": 32,
                "horizon_len": 8,
                "freq": 0,
                "verbose": False,
            },
        ]
        params_no_broadcasting = [
            dict(p, **{"broadcasting": False}) for p in test_params
        ]
        test_params.extend(params_no_broadcasting)
        return test_params


def _same_index(data):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(
        lambda x: x.equals(data.iloc[0])
    ).all(), "All series must has the same index"
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr

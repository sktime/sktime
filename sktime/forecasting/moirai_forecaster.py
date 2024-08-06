"""Adapter for using MOIRAI Forecasters."""

import pandas as pd

from sktime.datatypes import get_cutoff
from sktime.forecasting.base import _BaseGlobalForecaster

__author__ = ["benheid", "pranavvp16"]


class MOIRAIForecaster(_BaseGlobalForecaster):
    """
    Adapter for using MOIRAI Forecasters.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint of the model.
    fit_strategy : str, default=None
        Strategy to fit the model. If None, the model is used as it is.
    context_length : int, default=200
        Length of the context window.
    patch_size : int, default=32
        Size of the patch.
    num_samples : int, default=100
        Number of samples to draw.
    map_location : str, default=None
        Hardware to use for the model.
    deterministic : bool, default=False
        Whether to use a deterministic model.
    batch_size : int, default=32
        Size of the batch.


    Examples
    --------
    >>> from sktime.forecasting.moirai_forecaster import MOIRAIForecaster
    >>> import pandas as pd
    >>> import numpy as np
    >>> morai_forecaster = MOIRAIForecaster(
    ...     checkpoint_path=f"Salesforce/moirai-1.0-R-small"
    ... )
    >>> y = np.random.normal(0, 1, (30, 2))
    >>> X = y * 2 + np.random.normal(0, 1, (30,1))
    >>> index = pd.date_range("2020-01-01", periods=30, freq="D")
    >>> y = pd.DataFrame(y, index=index)
    >>> X = pd.DataFrame(X, columns=["x1", "x2"], index=index)
    >>> df = pd.concat([y, X], axis=1)
    >>> morai_forecaster.fit(y, X=X)
    >>> X_test = pd.DataFrame(np.random.normal(0, 1, (10, 2)),
    ...                      columns=["x1", "x2"],
    ...                      index=pd.date_range("2020-01-31", periods=10, freq="D"),
    ... )
    >>> forecast = morai_forecaster.predict(fh=range(1, 11), X=X_test)
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_dependencies": ["salesforce-uni2ts", "gluonts", "torch"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex"],
        "y_inner_mtype": ["pd.Series", "pd.DataFrame", "pd-multiindex"],
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        "python_dependencies_alias": {"salesforce-uni2ts": "uni2ts"},
    }

    def __init__(
        self,
        checkpoint_path: str,
        context_length=200,
        patch_size=32,
        num_samples=100,
        num_feat_dynamic_real=0,
        num_past_feat_dynamic_real=0,
        map_location=None,
        deterministic=False,
        batch_size=32,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real
        self.map_location = map_location
        self.deterministic = deterministic
        self.batch_size = batch_size

    def _fit(self, y, X, fh):
        # Load model and extract config

        from uni2ts.model.moirai import MoiraiForecast

        if fh is not None:
            prediction_length = max(fh.to_relative(self.cutoff))
        else:
            prediction_length = 1

        if self.num_feat_dynamic_real is None and X is not None:
            self.num_feat_dynamic_real = X.shape[1]
        if self.num_past_feat_dynamic_real is None:
            self.num_past_feat_dynamic_real = 0

        # load the sktime moirai weights by default
        model_kwargs = {
            "prediction_length": prediction_length,
            "context_length": self.context_length,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "target_dim": 2,
            "feat_dynamic_real_dim": self.num_feat_dynamic_real,
            "past_feat_dynamic_real_dim": self.num_past_feat_dynamic_real,
        }

        # Instantiate model with latest salesforce weights
        if self.checkpoint_path.startswith("Salesforce"):
            from uni2ts.model.moirai import MoiraiModule

            model_kwargs["module"] = MoiraiModule.from_pretrained(self.checkpoint_path)

            self.model = MoiraiForecast(**model_kwargs)

        else:
            from huggingface_hub import hf_hub_download

            model_kwargs["checkpoint_path"] = hf_hub_download(
                repo_id=self.checkpoint_path, filename="model.ckpt", revision="temp"
            )

            self.model = MoiraiForecast.load_from_checkpoint(**model_kwargs)
        self.model.to(self.map_location)

    def _predict(self, fh, y=None, X=None):
        if self.deterministic:
            import torch

            torch.manual_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.hparams.prediction_length = max(fh._values)

        if min(fh._values) < 0:
            raise NotImplementedError(
                "The MORAI adapter is not supporting insample predictions."
            )

        pred_df = pd.concat([self._y, self._X], axis=1)
        target = self._y.columns
        future_length = 0
        feat_dynamic_real = None

        if self._X is not None:
            feat_dynamic_real = self._X.columns

        # Zero shot case
        if y is None and X is not None:
            pred_df = self._extend_df(pred_df)
            future_length = max(fh._values) + 1

        # Zero shot case with X
        elif X is not None:
            pred_df = self._extend_df(pred_df, X)
            future_length = max(fh._values) + 1

        ds_test, df_config = self._get_pandas_dataset(
            pred_df, target, feat_dynamic_real, future_length
        )

        predictor = self.model.create_predictor(batch_size=self.batch_size)
        forecasts = predictor.predict(ds_test)
        forecast_it = iter(forecasts)

        return self._get_prediction_df(forecast_it, df_config)

    def _get_prediction_df(self, forecast_iter, df_config):
        def handle_series_prediction(forecast, target):
            pred = forecast.mean_ts
            return pred.rename(target)

        def handle_panel_predictions(forecast_iter, df_config):
            panels = []
            for forecast in forecast_iter:
                df = forecast.mean_ts.reset_index()
                df.columns = [df_config["timepoints"], df_config["target"]]
                df[df_config["item_id"]] = forecast.item_id
                df.set_index(
                    [df_config["item_id"], df_config["timepoints"]], inplace=True
                )
                panels.append(df)
            return pd.concat(panels)

        # Assuming all forecasts are either series or panel type.
        first_forecast = next(iter(forecast_iter))
        if first_forecast.item_id is None:
            return handle_series_prediction(first_forecast, df_config["target"])
        else:
            return handle_panel_predictions(forecast_iter, df_config)

    def _get_pandas_dataset(self, df, target, feat_dynamic_real=None, future_length=0):
        from gluonts.dataset.pandas import PandasDataset

        df_config = {
            "target": target[0],
        }

        if isinstance(df.index, pd.MultiIndex):
            item_id = df.index.names[0]
            df_config["item_id"] = item_id
            timepoints = df.index.names[-1]
            df_config["timepoints"] = timepoints
            df = df.reset_index()
            df.set_index(timepoints, inplace=True)
            return (
                PandasDataset.from_long_dataframe(
                    df,
                    target=target,
                    feat_dynamic_real=feat_dynamic_real,
                    item_id=item_id,
                    future_length=future_length,
                ),
                df_config,
            )
        else:
            return (
                PandasDataset(
                    df,
                    target=target,
                    feat_dynamic_real=feat_dynamic_real,
                    future_length=future_length,
                ),
                df_config,
            )

    def _extend_df(self, df, X=None):
        cutoff = get_cutoff(self._y)
        pred_index = pd.date_range(
            cutoff, periods=max(self.fh._values) + 1, freq=self._y.index.freq
        )[1:]

        if isinstance(df.index, pd.MultiIndex):
            new_index = pd.MultiIndex.from_product(
                [df.index.get_level_values(0).unique(), pred_index],
                names=df.index.names,
            )
        else:
            new_index = pred_index

        if X is not None:
            df_y = pd.DataFrame(columns=self._y.columns, index=new_index)
            df_y.fillna(0, inplace=True)
            pred_df = pd.concat([df_y, X], axis=1)
            extended_df = pd.concat([df, pred_df])
            return extended_df.sort_index()
        else:
            extended_df = pd.DataFrame(columns=df.columns, index=new_index)
            extended_df.fillna(0, inplace=True)
            print(extended_df)
            extended_df = pd.concat([df, extended_df])
            return extended_df.sort_index()

    # def _check_if_range_index

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
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {
                "deterministic": True,
            }
        ]

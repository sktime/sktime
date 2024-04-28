"""Adapter for using MOIRAI Forecasters."""


import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch

if _check_soft_dependencies("gluonts", severity="none"):
    from gluonts.dataset.pandas import PandasDataset

if _check_soft_dependencies("uni2ts", severity="none"):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiFinetune

if _check_soft_dependencies("huggingface_hub", severity="none"):
    from huggingface_hub import hf_hub_download

from uni2ts.data.builder.simple import _from_wide_dataframe
import datasets
from peft import LoraConfig, get_peft_model

import transformers

from sktime.forecasting.base import BaseForecaster

__author__ = ["benheid"]


class MOIRAIForecaster(BaseForecaster):
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
        "python_dependencies": ["uni2ts", "gluonts", "torch"],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:insample": False,
        "capability:pred_int:insample": False,
    }

    def __init__(
        self,
        checkpoint_path: str,
        fit_strategy=None,
        context_length=200,
        patch_size=32,
        num_samples=100,
        map_location=None,
        deterministic=False,
        batch_size=32,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.fit_strategy = fit_strategy
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.map_location = map_location
        self.deterministic = deterministic
        self.batch_size = batch_size

    def _fit(self, y, X, fh):
        # Load model and extract config

        if fh is not None:
            prediction_length = max(fh.to_relative(self.cutoff))
        else:
            prediction_length = 1

        if X is not None:
            df = pd.concat([y, X], axis=1)
            target = y.columns
            feat_dynamic_real = X.columns
        else:
            df = y
            target = y.columns
            feat_dynamic_real = None
        ds = PandasDataset(df, target=target, feat_dynamic_real=feat_dynamic_real)

        self.model = MoiraiFinetune.load_from_checkpoint(
            checkpoint_path=hf_hub_download(
                repo_id=self.checkpoint_path, filename="model.ckpt"
            ),
            prediction_length=prediction_length,
            context_length=self.context_length,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=2,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.model.to(self.map_location)
        self.model.train()

        if self.fit_strategy is None:
            return
        elif self.fit_strategy == "lora":
            peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules=["v_proj"])
            self.model = get_peft_model(self.model, peft_config)
        else:
            raise ValueError(f"Unknown fit strategy: {self.fit_strategy}")
        
        training_args = transformers.TrainingArguments(
            **{
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 32,}
        )

        hf_dataset = PyTorchDataset(y, 200, 200, X)
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=hf_dataset,
            data_collator=None
        )
        trainer.train()

    def _predict(self, fh, X=None):
        if self.deterministic:
            torch.manual_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.hparams.prediction_length = max(fh._values)

        if min(fh._values) < 0:
            raise NotImplementedError(
                "The MORAI adapter is not supporting insample predictions."
            )

        self.model.eval()

        pred_index = pd.date_range(
            self.cutoff[0], periods=max(fh._values) + 1, freq=self._y.index.freq
        )[1:]

        df_y = pd.concat(
            [self._y, pd.DataFrame(columns=self._y.columns, index=pred_index)]
        )
        if X is not None:
            df_x = pd.concat([self._X, X], axis=0)
            df_test = pd.concat([df_y, df_x], axis=1)
            ds_test = PandasDataset(
                df_test,
                target=self._y.columns,
                feat_dynamic_real=self._X.columns,
                future_length=max(fh._values),
            )
        else:
            ds_test = PandasDataset(
                df_y, target=self._y.columns, future_length=max(fh._values)
            )

        predictor = self.model.create_predictor(batch_size=self.batch_size)
        forecasts = predictor.predict(ds_test)
        forecast_it = iter(forecasts)
        forecast = next(forecast_it)

        return forecast.mean_ts.to_timestamp().loc[fh.to_absolute(self.cutoff)._values]

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


from torch.utils.data import Dataset

# class PyTorchDataset(Dataset):
#     """Dataset for use in sktime deep learning forecasters."""

#     def __init__(self, df, length):
#         self.df = df
#         self.length = length


#     def __len__(self):
#         """Return length of dataset."""
#         return len(self.df) - self.length

#     def __getitem__(self, idx):
#         """Return data."""
#         return {
#             str(entry): 
#             torch.from_numpy(self.df[entry][idx: idx + self.length].values) for entry in self.df.columns
#         }
        
class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, fh=200, X=None):
        self.y = y.values
        self.X = X.values if X is not None else X
        self.seq_len = seq_len
        self.fh = 1

    def __len__(self):
        """Return length of dataset."""
        return max(len(self.y) - self.seq_len - self.fh + 1, 0)

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        hist_y = tensor(self.y[i : i + self.seq_len]).float()
        if self.X is not None:
            exog_data = tensor(
                self.X[i + self.seq_len : i + self.seq_len + self.fh]
            ).float()
            hist_exog = tensor(self.X[i : i + self.seq_len]).float()
        else:
            exog_data = tensor([[]] * self.fh)
            hist_exog = tensor([[]] * self.seq_len)
        return {
            "past_target": hist_y,
            "past_feat_dynamic_real": hist_exog,
            "feat_dynamic_real": exog_data,
            "observed_mask": (~hist_y.isnan()).to(int),
            "label": from_numpy(
                self.y[i + self.seq_len : i + self.seq_len + self.fh]
            ).float(),
        }
        # target: Float[torch.Tensor, "*batch seq_len max_patch"],
        # observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        # sample_id: Int[torch.Tensor, "*batch seq_len"],
        # time_id: Int[torch.Tensor, "*batch seq_len"],
        # variate_id: Int[torch.Tensor, "*batch seq_len"],
        # prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        # patch_size: Int[torch.Tensor, "*batch seq_len"],
        # num_samples: Optional[int] = None,
    # past_target,
    # past_observed_target,
    # past_is_pad,
    # feat_dynamic_real,
    # observed_feat_dynamic_real,
    # past_feat_dynamic_real,
    # past_observed_feat_dynamic_real,
    # num_samples,
    # label,
    # label_ids.

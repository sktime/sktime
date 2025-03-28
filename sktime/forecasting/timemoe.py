"""Implements TimeMOE forecaster."""

__author__ = ["Maple728", "KimMeen", "PranavBhatP"]
# Maple728 and KimMeen for timemoe
__all__ = ["TimeMoEForecaster"]


import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster
from sktime.utils.singleton import _multiton


class TimeMoEForecaster(_BaseGlobalForecaster):
    """
    Interface for TimeMOE forecaster for zero-shot forecasting.

    TimeMoE is a decoder-only time series foundational model that uses a mixture
    of experts algorithm to make predictions. designed to operate in an auto-regressive
    manner, enabling universal forecasting with arbitrary prediction horizons
    and context lengths of up to 4096. This method has been proposed in [2]_ and the
    official code is avalaible at [2]_.

    Parameters
    ----------
    model_path: str
        Path to the TimeMOE model. This can be:

        - A model ID from the HuggingFace Hub, e.g., "Maple728/TimeMoE-50M"
        - A local directory containing the model files, specified as an absolute or
        relative path to the current working directory
        The path should point to a directory containing the model weights and
        configuration files in the format expected by the HuggingFace Transformers
        library.
    config: dict, optional
        A dictionary specifying the configuration of the TimeMOE model.
        The available configuration options include hyperparameters that control
        the prediction behavior, sampling, and hardware utilization.
        - input_size: int, default=1
            The size of the input time series.
        - hidden_size: int, default=4096
            The size of the hidden layers in the TimeMOE model.
        - intermediate_size: int, default=22016
            The size of the intermediate layers in the TimeMOE model.
        - horizon_lengths: list[int], default=[1]
            The prediction horizon length.
        - num_hidden_layers: int, default=32
            The number of hidden layers in the TimeMOE model.
        - num_attention_heads: int, default=32
            The number of attention heads in the TimeMOE model.
        - num_experts_per_tok: int, default=2
            The number of experts per token in the TimeMOE model.
        - num_experts: int, default=1
            The number of experts in the TimeMOE model.
        - max_position_embeddings: int, default=32768
            The maximum position embeddings in the TimeMOE model.
        - rms_norm_eps: float, default=1e-6
            The epsilon value for RMS normalization in the TimeMOE model.
        - rope_theta: int, default=10000
            Initialise theta for RoPE (Rotational Positional Embeddings).
        - attention_dropout: float, default=0.1
            The dropout rate for attention layers in the TimeMOE model.
        - apply_aux_loss: bool, default=True
            Whether to apply auxiliary loss in the TimeMOE model.
        - router_aux_loss_factor: float, default=0.02
            The auxiliary loss factor for the router in the TimeMOE model.
        - tie_word_embeddings: bool, default=False
            Whether to tie word embeddings in the TimeMOE model.

    seed: int, optional (default=None)
        Seed for reproducibility.

    use_source_package: bool, optional (default=False)
        If True, the model will be loaded directly from the source package ``TimeMoE``.
        This is useful if you want to bypass the local version of the package
        or when working in an environment where the latest updates from the source
        package are needed. If False, the model will be loaded from the local version
        of package maintained in sktime. To install the source package,
        follow the instructions here [1]_.

    ignore_deps: bool, optional, default=False
        If True, dependency checks will be ignored, and the user is expected to handle
        the installation of required packages manually. If False, the class will enforce
        the default dependencies required for Chronos.

    References
    ----------
    .. [1] https://github.com/Time-MoE/Time-MoE
    .. [2] Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye and others
    Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts

    Examples
    --------
    >>> from sktime.forecasting.timemoe import TimeMoEForecaster
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=5)
    >>> forecaster = TimeMoEForecaster("Maple728/TimeMoE-50M")
    >>> forecaster.fit(y_train)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3], y = y_test)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Maple728", "KimMeen", "PranavBhatP"],
        # abdulfatir and lostella for amazon-science/chronos-forecasting
        "maintainers": ["PranavBhatP"],
        "python_dependencies": ["torch", "transformers<=4.40.1", "accelerate<=0.28.0"],
        # estimator type
        # --------------
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "scitype:y": "univariate",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        model_path: str,
        config: dict = None,
        seed: int = None,
        use_source_package: bool = False,
        ignore_deps: bool = False,
    ):
        if not ignore_deps:
            _check_soft_dependencies("torch", severity="error")
            _check_soft_dependencies("transformers", severity="error")
        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed

        self.config = config
        _config = self._get_default_config()
        _config.update(config if config is not None else {})
        self._config = _config

        self.model_path = model_path
        self.use_source_package = use_source_package
        self.ignore_deps = ignore_deps

        if self.ignore_deps:
            self.set_tags(python_dependencies=[])
        elif self.use_source_package:
            self.set_tags(python_dependencies=["timemoe"])
        else:
            self.set_tags(
                python_dependencies=[
                    "torch",
                    "transformers<=4.40.1",
                    "accelerate<=0.28.0",
                ]
            )

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        config = self._config
        if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
            config["input_size"] = y.shape[1]
        else:
            config["input_size"] = 1
        self._config = config
        self.model = _CachedTimeMoE(
            key=self._get_unique_timemoe_key(),
            timemoe_kwargs=self._get_timemoe_kwargs(),
            use_source_package=self.use_source_package,
        ).load_from_checkpoint()

        return self

    def _get_timemoe_kwargs(self):
        """Get the kwargs for TimeMoE model."""
        kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self._config["torch_dtype"],
            "device_map": self._config["device_map"],
        }

        return kwargs

    def _get_unique_timemoe_key(self):
        """Get a unique key for TimeMoE model."""
        model_path = self.model_path
        use_source_package = self.use_source_package
        kwargs = self._get_timemoe_kwargs()

        kwargs_plus_model_path = {
            **kwargs,
            "model_path": model_path,
            "use_source_package": use_source_package,
        }

        return str(sorted(kwargs_plus_model_path.items()))

    def _get_default_config(self):
        """Return the default configuration for TimeMoE model.

        Returns
        -------
        dict
            The default configuration for TimeMoE model.
        """
        import torch

        default_config = {
            "input_size": 1,
            "hidden_size": 4096,
            "intermediate_size": 22016,
            "horizon_lengths": [1],
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": None,
            "hidden_act": "silu",
            "num_experts_per_tok": 2,
            "num_experts": 1,
            "max_position_embeddings": 32768,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": True,
            "use_dense": False,
            "rope_theta": 10000,
            "attention_dropout": 0.0,
            "apply_aux_loss": True,
            "router_aux_loss_factor": 0.02,
            "tie_word_embeddings": False,
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu",
        }
        return default_config

    def _predict(
        self,
        fh,
        X=None,
        y=None,
    ):
        """Forecast time series at future horizon.

        Private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        y : pd.Series, optional (default=None)
            Optional series to use instead of the series passed in fit.

        Returns
        -------
        y_pred : pd.DataFrame
            Predicted forecasts.
        """
        import torch
        import transformers

        transformers.set_seed(self._seed)
        if fh is not None:
            prediction_length = int(max(fh.to_relative(self.cutoff)))
        else:
            prediction_length = 1

        _y = self._y.copy()
        if y is not None:
            _y = y.copy()
        _y_df = _y

        index_names = _y.index.names
        if isinstance(_y.index, pd.MultiIndex):
            _y = _frame2numpy(_y)
        else:
            if isinstance(_y, pd.DataFrame):
                _y = _y.values.reshape(1, -1, _y.shape[1])
            else:
                _y = _y.values.reshape(1, -1, 1)

        results = []
        for i in range(_y.shape[0]):
            current_results = []
            for j in range(_y.shape[2]):
                _y_i = _y[i, :, j]

                input_tensor = torch.tensor(
                    _y_i, dtype=self._config["torch_dtype"]
                ).unsqueeze(0)

                attention_mask = torch.ones(input_tensor.shape[:2], dtype=torch.long)

                with torch.no_grad():
                    output = self.model(
                        input_tensor,
                        attention_mask,
                        max_horizon_length=prediction_length,
                        use_cache=True,
                        return_dict=True,
                    )

                predictions = output.logits.squeeze(0).to(torch.float).cpu().numpy()
                final_predictions = predictions[-prediction_length:]
                final_predictions = final_predictions.reshape(
                    prediction_length, self._config["input_size"]
                )
                selected_indices = [h - 1 for h in fh.to_relative(self.cutoff)]
                final_predictions = final_predictions[selected_indices]
                current_results.append(final_predictions)
            combined_results = np.concatenate(current_results, axis=1)
            results.append(combined_results)

        if len(results) > 1:
            combined_results = np.concatenate(results, axis=0)
        else:
            combined_results = results[0]

        forecast_index = fh.to_absolute(self.cutoff)

        if hasattr(forecast_index, "to_numpy"):
            forecast_index = forecast_index.to_numpy()
        else:
            forecast_index = list(forecast_index)

        if isinstance(_y_df.index, pd.MultiIndex):
            # creates a a time index which replaces the existing tiume index with
            # the forecast index.
            idx = pd.MultiIndex.from_product(
                [
                    _y_df.index.get_level_values(i).unique()
                    for i in range(len(_y_df.index.names) - 1)
                ]
                + [forecast_index],
                names=index_names,
            )

            y_pred = pd.DataFrame(
                combined_results.reshape(-1, self._config["input_size"]),
                index=idx,
                columns=_y_df.columns if isinstance(_y_df, pd.DataFrame) else None,
            )
            y_pred.index.names = _y_df.index.names
        else:
            # this is for univariate data.
            y_pred = pd.DataFrame(
                combined_results,
                index=forecast_index,
                columns=_y_df.columns if isinstance(_y_df, pd.DataFrame) else None,
            )
            y_pred.index.names = _y_df.index.names

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_default="default"):
        """Get the test parameters for the forecaster.

        Parameters
        ----------
        parameter_default : str, optional (default='default')
            The default parameter to use for the test.

        Returns
        -------
        params : dict
            Dictionary of test parameters.
        """
        test_params = []
        test_params.append(
            {
                "model_path": "Maple728/TimeMoE-50M",
            }
        )
        test_params.append(
            {"model_path": "Maple728/TimeMoE-50M", "config": {"num_experts": 2}}
        )

        return test_params


def _same_index(data):
    """
    Ensure that all series within a multi-indexed DataFrame share the same index.

    Parameters
    ----------
    data : pandas.DataFrame
        A multi-indexed DataFrame where the last level of the index should be the same
        across all grouped series.

    Returns
    -------
    pandas.Index, int
        The common index found at the last level and the length of this index
    """
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(lambda x: x.equals(data.iloc[0])).all(), (
        "All series must has the same index"
    )
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    """
    Convert a multi-indexed DataFrame into a 3D NumPy array.

    The function first ensures that all series in `data` share the same index at the
    last level using `_same_index`, then reshapes the DataFrame values into a NumPy
    array with dimensions `(batch_size, sequence_length, feature_dim)`.

    Parameters
    ----------
    data : pandas.DataFrame
        A multi-indexed DataFrame with consistent last-level indices across all series.

    Returns
    -------
    numpy.ndarray
        A 3D NumPy array of shape `(n_groups, sequence_length, n_features)`, where:
        - `n_groups` is the number of unique index groups in `data`
        - `sequence_length` is the length of the common index
        - `n_features` is the number of columns in `data`.
    """
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


@_multiton
class _CachedTimeMoE:
    """Cached TimeMoE model to ensure only one instance exists in memory.

    TimeMoE is a zero-shot model and immutable, hence there will not be
    any side effects of sharing the same instance across multiple uses.
    This caching mechanism uses the _multiton decorator to ensure
    that models with the same configuration are reused, preventing
    duplicate models in memory when handling multivariate data.
    """

    def __init__(self, key, timemoe_kwargs, use_source_package):
        self.key = key
        self.timemoe_kwargs = timemoe_kwargs
        self.use_source_package = use_source_package
        self.model = None

    def load_from_checkpoint(self):
        """Load the model from checkpoint."""
        if self.use_source_package:
            if not _check_soft_dependencies("timemoe", severity="none"):
                raise ImportError(
                    "To use TimeMoE with use_source_package=True, "
                    "you must install the TimeMoE package from "
                    "https://github.com/Time-MoE/Time-MoE"
                )
            from timemoe.models.modeling_timemoe import TimeMoeForPrediction

            model = TimeMoeForPrediction.from_pretrained(**self.timemoe_kwargs)
        else:
            from sktime.libs.timemoe import TimeMoeForPrediction

            model = TimeMoeForPrediction.from_pretrained(**self.timemoe_kwargs)

        return model

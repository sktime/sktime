"""Implements TimeMOE forecaster."""

__author__ = ["Maple728", "KimMeen", "PranavBhatP"]
# Maple728 and KimMeen for timemoe
__all__ = ["TimeMoEForecaster"]


import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _GlobalForecastingDeprecationMixin
from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    ModelHandle,
)


class TimeMoEForecaster(_GlobalForecastingDeprecationMixin, BaseFoundationForecaster):
    """
    Interface for TimeMOE forecaster for zero-shot forecasting.

    TimeMoE is a decoder-only time series foundational model that uses a mixture
    of experts algorithm to make predictions. designed to operate in an auto-regressive
    manner, enabling universal forecasting with arbitrary prediction horizons
    and context lengths of up to 4096. This method has been proposed in [2]_ and the
    official code is available at [2]_.

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
    TimeMoEForecaster(model_path='Maple728/TimeMoE-50M')
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
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "capability:multivariate": False,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        # testing configuration
        # ---------------------
        "tests:libs": ["sktime.libs.timemoe"],
    }

    def __init__(
        self,
        model_path: str,
        config: dict = None,
        seed: int = None,
        use_source_package: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.config = config
        self.seed = seed
        self.use_source_package = use_source_package
        self.ignore_deps = ignore_deps

        super().__init__(
            model_path=model_path,
            config=config,
            device="cpu",
            dtype="torch.bfloat16",
            random_state=seed,
            ignore_deps=ignore_deps,
        )

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values condition on parameters.

        This method should be used for setting dynamic tags only.
        """
        super().__dynamic_tags__()
        if self.ignore_deps:
            return
        if self.use_source_package:
            self.set_tags(python_dependencies=["timemoe"])
        else:
            self.set_tags(
                python_dependencies=[
                    "torch",
                    "transformers<=4.40.1",
                    "accelerate<=0.28.0",
                ]
            )

    def _load_model(self):
        """Load vendored or source-package TimeMoE into a model handle."""
        if self.use_source_package:
            if not _check_soft_dependencies("timemoe", severity="none"):
                raise ImportError(
                    "To use TimeMoE with use_source_package=True, "
                    "you must install the TimeMoE package from "
                    "https://github.com/Time-MoE/Time-MoE"
                )
            from timemoe.models.modeling_timemoe import TimeMoeForPrediction
        else:
            from sktime.libs.timemoe import TimeMoeForPrediction

        model = TimeMoeForPrediction.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype_,
            device_map=self.device_,
        )
        return ModelHandle(model=model)

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        """Run channel-wise autoregressive TimeMoE inference."""
        import torch

        model = handle.model

        y_values = context_y.copy()
        if isinstance(y_values, pd.DataFrame):
            y_values = y_values.values.reshape(1, -1, y_values.shape[1])
        else:
            y_values = y_values.values.reshape(1, -1, 1)

        results = []
        for i in range(y_values.shape[0]):
            current_results = []
            for j in range(y_values.shape[2]):
                channel = y_values[i, :, j]

                input_tensor = torch.tensor(channel, dtype=self.dtype_).unsqueeze(0)

                attention_mask = torch.ones(input_tensor.shape[:2], dtype=torch.long)

                output = model(
                    input_tensor,
                    attention_mask,
                    max_horizon_length=pred_len,
                    use_cache=True,
                    return_dict=True,
                )

                predictions = output.logits.squeeze(0).to(torch.float).cpu().numpy()
                final_predictions = predictions[-pred_len:]
                final_predictions = final_predictions.reshape(pred_len, 1)
                current_results.append(final_predictions)
            combined_results = np.concatenate(current_results, axis=1)
            results.append(combined_results)

        if len(results) > 1:
            combined_results = np.concatenate(results, axis=0)
        else:
            combined_results = results[0]

        return ForecastResult(mean=combined_results)

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

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from pytorch-forecasting."""
import functools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pytorch_forecasting.metrics import MultiHorizonMetric
from torch import nn

from sktime.forecasting.base.adapters._pytorchforecasting import (
    _none_check,
    _PytorchForecastingAdapter,
)

__author__ = ["XinyuWu"]


class PytorchForecastingTFT(_PytorchForecastingAdapter):
    """pytorch-forecasting Temporal Fusion Transformer model."""

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting", "torch", "lightning"],
    }

    def __init__(
        self: "PytorchForecastingTFT",
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: Optional[List[str]] = None,
        static_reals: Optional[List[str]] = None,
        time_varying_categoricals_encoder: Optional[List[str]] = None,
        time_varying_categoricals_decoder: Optional[List[str]] = None,
        categorical_groups: Optional[Dict[str, List[str]]] = None,
        time_varying_reals_encoder: Optional[List[str]] = None,
        time_varying_reals_decoder: Optional[List[str]] = None,
        x_reals: Optional[List[str]] = None,
        x_categoricals: Optional[List[str]] = None,
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Optional[Dict[str, int]] = None,
        embedding_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
        embedding_paddings: Optional[List[str]] = None,
        embedding_labels: Optional[Dict[str, np.ndarray]] = None,
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Optional[Dict[str, int]] = None,
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,
        allowed_encoder_known_variable_names: List[str] | None = None,
        trainer_params: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            loss,
            logging_metrics,
            allowed_encoder_known_variable_names,
            trainer_params,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.output_size = output_size
        self.attention_head_size = attention_head_size
        self.max_encoder_length = max_encoder_length
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_categoricals_encoder = time_varying_categoricals_encoder
        self.time_varying_categoricals_decoder = time_varying_categoricals_decoder
        self.categorical_groups = categorical_groups
        self.time_varying_reals_encoder = time_varying_reals_encoder
        self.time_varying_reals_decoder = time_varying_reals_decoder
        self.x_reals = x_reals
        self.x_categoricals = x_categoricals
        self.hidden_continuous_size = hidden_continuous_size
        self.hidden_continuous_sizes = hidden_continuous_sizes
        self.embedding_sizes = embedding_sizes
        self.embedding_paddings = embedding_paddings
        self.embedding_labels = embedding_labels
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.log_val_interval = log_val_interval
        self.log_gradient_flow = log_gradient_flow
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        self.monotone_constaints = monotone_constaints
        self.share_single_variable_networks = share_single_variable_networks
        self.causal_attention = causal_attention

    @functools.cached_property
    def algorithm_class(self: "PytorchForecastingTFT"):
        """Import underlying pytorch-forecasting algorithm class."""
        from pytorch_forecasting import TemporalFusionTransformer

        return TemporalFusionTransformer

    @functools.cached_property
    def algorithm_parameters(self: "PytorchForecastingTFT") -> dict:
        """Get keyword parameters for the TFT class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        self._static_categoricals = _none_check(self.static_categoricals, [])
        self._static_reals = _none_check(self.static_reals, [])
        self._time_varying_categoricals_encoder = _none_check(
            self.time_varying_categoricals_encoder, []
        )
        self._time_varying_categoricals_decoder = _none_check(
            self.time_varying_categoricals_decoder, []
        )
        self._categorical_groups = _none_check(self.categorical_groups, {})
        self._time_varying_reals_encoder = _none_check(
            self.time_varying_reals_encoder, []
        )
        self._time_varying_reals_decoder = _none_check(
            self.time_varying_reals_decoder, []
        )
        self._x_reals = _none_check(self.x_reals, [])
        self._x_categoricals = _none_check(self.x_categoricals, [])
        self._hidden_continuous_sizes = _none_check(self.hidden_continuous_sizes, {})
        self._embedding_sizes = _none_check(self.embedding_sizes, {})
        self._embedding_paddings = _none_check(self.embedding_paddings, [])
        self._embedding_labels = _none_check(self.embedding_labels, {})
        self._monotone_constaints = _none_check(self.monotone_constaints, {})

        return {
            "hidden_size": self.hidden_size,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
            "attention_head_size": self.attention_head_size,
            "max_encoder_length": self.max_encoder_length,
            "static_categoricals": self._static_categoricals,
            "static_reals": self._static_reals,
            "time_varying_categoricals_encoder": (
                self._time_varying_categoricals_encoder
            ),
            "time_varying_categoricals_decoder": (
                self._time_varying_categoricals_decoder
            ),
            "categorical_groups": self._categorical_groups,
            "time_varying_reals_encoder": self._time_varying_reals_encoder,
            "time_varying_reals_decoder": self._time_varying_reals_decoder,
            "x_reals": self._x_reals,
            "x_categoricals": self._x_categoricals,
            "hidden_continuous_size": self.hidden_continuous_size,
            "hidden_continuous_sizes": self._hidden_continuous_sizes,
            "embedding_sizes": self._embedding_sizes,
            "embedding_paddings": self._embedding_paddings,
            "embedding_labels": self._embedding_labels,
            "learning_rate": self.learning_rate,
            "log_interval": self.log_interval,
            "log_val_interval": self.log_val_interval,
            "log_gradient_flow": self.log_gradient_flow,
            "reduce_on_plateau_patience": self.reduce_on_plateau_patience,
            "monotone_constaints": self._monotone_constaints,
            "share_single_variable_networks": self.share_single_variable_networks,
            "causal_attention": self.causal_attention,
        }

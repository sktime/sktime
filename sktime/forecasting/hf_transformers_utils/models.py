"""Model adapters and mappings for hugging face transformers models."""


class TransformersForecaster:
    """Adapter for Timer model."""

    def __init__(self, config):
        """Initialize the adapter with the model configuration."""
        self.config = config
        from transformers import (
            AutoformerForPrediction,
            InformerForPrediction,
            TimeSeriesTransformerForPrediction,
        )

        # Select the appropriate model based on the model type in config
        if self.config.model_type == "autoformer":
            self.source_model_class = AutoformerForPrediction
        elif self.config.model_type == "informer":
            self.source_model_class = InformerForPrediction
        elif self.config.model_type == "time_series_transformer":
            self.source_model_class = TimeSeriesTransformerForPrediction

    def pred_output(
        self,
        model,
        past_values,
        past_time_features,
        future_time_features,
        past_observed_mask,
        fh,
    ):
        """Predict output based on unique method of each model."""
        # fh param for interface consistency as we're giving
        # superset of possible args for each function
        pred = model.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask,
        )
        pred = pred.sequences.mean(dim=1).detach().cpu().numpy().T

        pred = pred.reshape((-1,))
        return pred

    def update_config(self, config, user_config, X, fh):
        """Update config with user provided config."""
        _config = config.to_dict()
        _config.update(user_config)
        _config["num_dynamic_real_features"] = X.shape[-1] if X is not None else 0
        _config["num_static_real_features"] = 0
        _config["num_dynamic_real_features"] = 0
        _config["num_static_categorical_features"] = 0
        _config["num_time_features"] = 0 if X is None else X.shape[-1]

        if hasattr(config, "feature_size"):
            del _config["feature_size"]

        if fh is not None:
            _config["prediction_length"] = max(
                *(fh),
                _config["prediction_length"],
            )

        config = config.from_dict(_config)
        self.config = config
        return config

    def get_seq_args(self):
        """Get context and prediction length."""
        context_len = self.config.context_length + max(self.config.lags_sequence)
        pred_len = self.config.prediction_length
        return context_len, pred_len


class TimerForecaster:
    """Adapter for Autoformer, Informer, and TimeSeriesTransformer models."""

    def __init__(self, config):
        """Initialize the adapter with the model configuration."""
        self.config = config
        from transformers import AutoModelForCausalLM

        self.source_model_class = AutoModelForCausalLM

    def pred_output(
        self,
        model,
        past_values,
        past_time_features,
        future_time_features,
        past_observed_mask,
        fh,
    ):
        """Predict output based on unique method of each model."""
        # unused args are for interface consistency as we're giving
        # superset of possible args for each function

        pred = model.generate(
            inputs=past_values,
            max_new_tokens=max(fh._values),
        )
        pred = pred.reshape((-1,))
        return pred

    def update_config(self, config, user_config, X, fh):
        """Update config with user provided config."""
        """Update config with user provided config."""
        # X for interface consistency as we're giving
        # superset of possible args for each function
        _config = config.to_dict()
        _config.update(user_config)

        if fh is not None:
            _config["output_token_lens"][0] = max(
                *(fh),
                _config["output_token_lens"][0],
            )

        config = config.from_dict(_config)
        self.config = config
        return config

    def get_seq_args(self):
        """Get context and pred length."""
        context_length = self.config.input_token_len
        prediction_length = self.config.output_token_lens[0]
        return context_length, prediction_length


# Dictionary mapping model types to custom adapter class
MODEL_MAPPINGS = {
    "timer": {
        "custom_adapter_class": TimerForecaster,
    },
    "autoformer": {
        "custom_adapter_class": TransformersForecaster,
    },
    "informer": {
        "custom_adapter_class": TransformersForecaster,
    },
    "timeseriestransformer": {
        "custom_adapter_class": TransformersForecaster,
    },
}

"""Model adapters and mappings for hugging face transformers models."""


class TransformersForecaster:
    """Adapter for Autoformer, Informer, and TimeSeriesTransformer models."""

    def __init__(self, config):
        """Initialize the adapter with the model configuration."""
        self.config = config
        import transformers

        # Select the appropriate model based on the architecture in config
        source_model_class = config.architectures[0]
        self.source_model_class = getattr(transformers, source_model_class)

    def predict_output(
        self,
        model,
        past_values,
        past_time_features,
        future_time_features,
        past_observed_mask,
        fh,
    ):
        """Predict output based on unique method of each model."""
        # fh param passed for interface consistency as we're giving
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

    def update_config(self, config, X, fh):
        """Update config attributes."""
        config["num_dynamic_real_features"] = X.shape[-1] if X is not None else 0
        config["num_static_real_features"] = 0
        config["num_dynamic_real_features"] = 0
        config["num_static_categorical_features"] = 0
        config["num_time_features"] = 0 if X is None else X.shape[-1]

        if "feature_size" in config:
            del config["feature_size"]

        if fh is not None:
            config["prediction_length"] = max(
                *(fh),
                config["prediction_length"],
            )

        self.config = config
        return config

    def get_seq_args(self):
        """Get context and prediction length."""
        context_len = self.config["context_length"] + max(self.config["lags_sequence"])
        pred_len = self.config["prediction_length"]
        return context_len, pred_len


class TimerForecaster:
    """Adapter for Timer model."""

    def __init__(self, config):
        """Initialize the adapter with the model configuration."""
        self.config = config
        from transformers import AutoModelForCausalLM

        self.source_model_class = AutoModelForCausalLM

    def predict_output(
        self,
        model,
        past_values,
        past_time_features,
        future_time_features,
        past_observed_mask,
        fh,
    ):
        """Predict output based on unique method of each model."""
        # unused args passed for interface consistency as we're
        # giving superset of possible args for each function

        pred = model.generate(
            inputs=past_values,
            max_new_tokens=max(fh._values),
        )
        pred = pred.reshape((-1,))
        return pred

    def update_config(self, config, X, fh):
        """Update config attributes."""
        # X passed for interface consistency as we're giving
        # superset of possible args for each function

        if fh is not None:
            config["output_token_lens"][0] = max(
                *(fh),
                config["output_token_lens"][0],
            )

        self.config = config
        return config

    def get_seq_args(self):
        """Get context and pred length."""
        context_length = self.config["input_token_len"]
        prediction_length = self.config["output_token_lens"][0]
        return context_length, prediction_length


# Dictionary mapping model types from config to custom adapter class
ADAPTER_MAPPINGS = {
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

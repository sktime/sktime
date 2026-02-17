"""Implements Chronos 2 forecaster."""

__all__ = ["Chronos2Forecaster"]


from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster

if _check_soft_dependencies("torch", severity="none"):
    import torch
else:

    class torch:
        """Dummy class if torch is unavailable."""

        bfloat16 = None

        class Tensor:
            """Dummy class if torch is unavailable."""


class Chronos2Forecaster(BaseForecaster):
    """Interface to the Chronos 2 Zero-Shot Forecaster by Amazon Research.

    Chronos 2 is a pretrained time series foundation model developed by Amazon
    for zero-shot forecasting. Unlike Chronos v1, it natively supports
    multivariate forecasting, exogenous covariates (past and future), and
    quantile-based probabilistic predictions. This method has been proposed
    in [2]_ and official code is given at [1]_.

    Parameters
    ----------
    model_path : str, default="amazon/chronos-2"
        Path to the Chronos 2 HuggingFace model. Available models include
        ``"amazon/chronos-2"`` (120M parameters).

    config : dict, optional, default={}
        A dictionary specifying the configuration settings for the model.
        The available configuration options include:

        - "batch_size" : int, default=256
            Batch size for inference.
        - "context_length" : int or None, default=None
            Maximum context length. If None, uses the model default (8192).
        - "cross_learning" : bool, default=False
            If True, enables cross-learning across variates for
            multivariate forecasting.
        - "limit_prediction_length" : bool, default=False
            If True, limits the prediction length to the model's maximum.
        - "torch_dtype" : torch.dtype, default=torch.bfloat16
            Data type to use for model weights and operations.
        - "device_map" : str, default="cpu"
            Specifies the device on which to run the model, e.g.,
            "cpu", "cuda", or "mps".

    quantile_levels : list of float, default=[0.1, 0.2, 0.3, 0.4, 0.5, \
        0.6, 0.7, 0.8, 0.9]
        Quantile levels for probabilistic predictions.

    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    model_pipeline : Chronos2Pipeline
        The underlying model pipeline used for forecasting.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    .. [2] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, and others (2025).
       Chronos 2: From Univariate to Universal Forecasting

    """

    def __init__(
        self,
        model_path: str = "amazon/chronos-2",
        config: dict = None,
        quantile_levels: list = None,
        seed: int | None = None,
    ):
        self.model_path = model_path
        self.config = config
        self.quantile_levels = quantile_levels
        self.seed = seed
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            The target time series to which the forecaster will be fitted.
            Can be univariate (Series) or multivariate (DataFrame).
        X : pd.DataFrame, optional, default=None
            Exogenous covariates corresponding to the target series. Should
            have the same index as `y` and columns for each covariate.
        fh : ForecastingHorizon, optional, default=None
            The forecasting horizon with the time points to predict. If None,
            it will be inferred from the training data.

        Returns
        -------
        self : returns an instance of self.
        """
        # Placeholder for actual fitting logic, which would involve loading the
        # Chronos 2 model and preparing it for forecasting.
        return self

    def _get_chronos2_kwargs(self):
        """Get Chronos 2 specific kwargs from the config."""
        # This method would extract and validate the relevant configuration
        # settings for the Chronos 2 model, such as batch size, context length,
        # cross-learning flag, etc., and return them in a format suitable for
        # initializing the model pipeline.
        pass

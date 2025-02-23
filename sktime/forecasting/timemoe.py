"""Implements TimeMOE forecaster."""

__author__ = ["Maple728", "KimMeen", "PranavBhatP"]
# Maple728 and KimMeen for timemoe
__all__ = ["TimeMoEForecaster"]


from sktime.forecasting.base import _BaseGlobalForecaster


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
        Path to the TimeMOE HuggingFace model.
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
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> forecaster = TimeMoEForecaster("Maple728/TimeMoE-50M")
    >>> forecaster.fit(y_train)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
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

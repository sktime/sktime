"""ConvTimeNet (PyTorch) Forecaster for time series forecasting."""

__author__ = ["Tanuj-Taneja1"]
__all__ = ["ConvTimeNetForecaster"]

import warnings

from sktime.forecasting.base.adapters import _pytorch
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")


class ConvTimeNetForecaster(_pytorch.BaseDeepNetworkPyTorch):
    """ConvTimeNet for time series forecasting.

    ConvTimeNet is a hierarchical pure convolutional model designed.
    Unlike prevalent methods centered around self-attention mechanisms,
    ConvTimeNet introduces two key innovations:

    1. A deformable patch layer that adaptively perceives local patterns of
       temporally dependent basic units in a data-driven manner.
    2. Hierarchical pure convolutional blocks that capture dependency relationships
        among the representations of basic units at different scales.

    The model employs a large kernel mechanism allowing convolutional blocks
    to be deeply stacked, achieving a larger receptive field. This architecture
    effectively models both local patterns and their multi-scale dependencies
    within a single model, addressing common challenges in time series analysis
    such as adaptive perception of local patterns and multi-scale dependency capture.

    This forecaster has been wrapped around implementations from [1]_ and [2]_.

    Parameters
    ----------
    context_window : int
        Length of the input sequence (context window).
    patch_ks : int
        Kernel size for patch creation. Determines the size of each patch extracted
        from the input sequence for patch embedding.
    patch_sd : int
        Stride length for patch creation. Determines the step size for moving the
        patch window across the input sequence.
    dw_ks : tuple, optional (default=(9, 3))
        Kernel sizes for depthwise convolution layers.
    d_model : int, optional (default=64)
        Dimension of the model (number of features in the hidden state).
    d_ff : int, optional (default=256)
        Dimension of the feedforward network.
    norm : str, optional (default="batch")
        Type of normalization to use ("batch" or "layer").
    dropout : float, optional (default=0.0)
        Dropout rate to apply to layers.
    act : str, optional (default="gelu")
        Activation function to use ("relu", "gelu", etc.).
    head_dropout : float, optional (default=0)
        Dropout rate for the head layer.
    padding_patch : int or None, optional (default=None)
        Padding size for patch embedding. If None, no padding is applied.
    revin : bool, optional (default=True)
        Whether to use RevIN normalization.
    affine : bool, optional (default=True)
        Whether RevIN uses affine transformation.
    subtract_last : bool, optional (default=False)
        Whether to subtract the last value in RevIN.
    deformable : bool, optional (default=True)
        Whether to use deformable patch embedding.
    enable_res_param : bool, optional (default=True)
        Whether to enable residual parameterization.
    re_param : bool, optional (default=True)
        Whether to use re-parameterization.
    re_param_kernel : int, optional (default=3)
        Kernel size for re-parameterization.
    num_epochs : int, optional (default=16)
        The number of epochs to train the model.
    batch_size : int, optional (default=8)
        The size of each mini-batch during training.
    criterion : callable, optional (default=None)
        The loss function to use. If None, MSELoss will be used.
    criterion_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    optimizer : str or torch.optim.Optimizer, optional (default=None)
        The optimizer to use. If None, Adam will be used.
    optimizer_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
    lr : float, optional (default=0.001)
        The learning rate to use for the optimizer.
    device : str, optional (default="cpu")
        Device to use for computation ("cpu" or "cuda").
    random_state : int, RandomState instance or None, optional (default=None)
        Random state for reproducibility. If int, it's the seed for the random
        number generator. If None, the random number generator uses a random seed.

    Examples
    --------
    >>> from sktime.forecasting.convtimenet import ConvTimeNetForecaster
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Create a sample univariate time series
    >>> y = pd.Series(np.arange(1024))  # Example univariate time series data
    >>> # Create and fit the forecaster
    >>> forecaster = ConvTimeNetForecaster(
    ...     context_window=48,
    ...     patch_ks=8,
    ...     patch_sd=1,
    ...     dw_ks=(13,7),
    ...     d_model=128,
    ...     d_ff=128,
    ...     norm="batch",
    ...     dropout=0.01,
    ...     act="gelu",
    ...     head_dropout=0.01,
    ...     padding_patch=None,
    ...     revin=True,
    ...     affine=True,
    ...     subtract_last=False,
    ...     deformable=True,
    ...     enable_res_param=True,
    ...     re_param=True,
    ...     re_param_kernel=3,
    ...     num_epochs=10,
    ...     batch_size=64,
    ...     lr=0.002,
    ...     device="cpu",
    ...     random_state=42
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1,2,3,4,5,6,7,8,9,10,11,12])  # doctest: +SKIP
    ConvTimeNetForecaster(...)
    >>> # Make predictions
    >>> y_pred = forecaster.predict(fh=[1,2,3,4,5,6,7,8,9,10,11,12])  # doctest: +SKIP
    >>> print(y_pred)  # doctest: +SKIP

    References
    ----------
    .. [1] Cheng, M., Yang, J., Pan, T., Liu, Q., & Li, Z. (2024). ConvTimeNet: A deep
        hierarchical fully convolutional model for multivariate time series analysis.
        arXiv preprint arXiv:2403.01493. https://arxiv.org/abs/2403.01493
    .. [2] https://github.com/Mingyue-Cheng/ConvTimeNet
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Mingyue-Cheng", "0russewt0", "pty12345", "Tanuj-Taneja1"],
        "maintainers": ["Tanuj-Taneja1"],
        "tests:skip_by_name": [
            "test_fit_idempotent",
            "test_update_predict_predicted_index",
        ],
        "python_dependencies": ["torch"],
        # estimator type
        # --------------
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # CI and testing
        # --------------
        "tests:libs": [
            "sktime.networks.convtimenet.forecaster._convtimenet",
            "sktime.networks.convtimenet.forecaster._convtimenet_backbone",
            "sktime.networks.convtimenet.forecaster._patch_layers",
            "sktime.networks.convtimenet.forecaster._revin",
        ],
        "tests:vm": True,
    }

    def __init__(
        self,
        context_window,
        patch_ks,
        patch_sd,
        dw_ks=(9, 3),
        d_model=64,
        d_ff=256,
        norm="batch",
        dropout=0.0,
        act="gelu",
        head_dropout=0,
        padding_patch=None,
        revin=True,
        affine=True,
        subtract_last=False,
        deformable=True,
        enable_res_param=True,
        re_param=True,
        re_param_kernel=3,
        num_epochs=16,
        batch_size=8,
        criterion_kwargs=None,
        criterion=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        device="cpu",
        random_state=None,
    ):
        self.context_window = context_window
        self.patch_ks = patch_ks
        self.patch_sd = patch_sd
        self.dw_ks = dw_ks
        self.d_model = d_model
        self.d_ff = d_ff
        self.norm = norm
        self.dropout = dropout
        self.act = act
        self.head_dropout = head_dropout
        self.padding_patch = padding_patch
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.deformable = deformable
        self.enable_res_param = enable_res_param
        self.re_param = re_param
        self.re_param_kernel = re_param_kernel
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.criterion = criterion
        self.custom_dataset_train = None
        self.custom_dataset_pred = None
        super().__init__(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            criterion_kwargs=self.criterion_kwargs,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            lr=self.lr,
        )

    def _build_network(self, fh):
        from sktime.networks.convtimenet.forecaster._convtimenet import Model

        # Check if context_window needs adjustment
        # Formula: len(y) - context_window - fh + 1 must be > 0 for training samples
        dataset_len = len(self._y) - self.context_window - fh + 1

        if dataset_len <= 0:
            original_context_window = self.context_window
            adjusted_context_window = max(1, len(self._y) - fh)

            warnings.warn(
                f"The context_window ({original_context_window}) is too large "
                f"for the given time series (length={len(self._y)}) and forecast "
                f"horizon (fh={fh}). Adjusting context_window from "
                f"{original_context_window} to {adjusted_context_window} to ensure "
                f"at least one training sample.\nConsider using a longer time series "
                f"or reducing context_window or forecast horizon.\n",
                UserWarning,
            )
            self.context_window = adjusted_context_window

        # Check if patch_ks needs adjustment based on context_window
        # patch_num = (context_window - patch_ks) / patch_sd + 1 must be > 0
        # => context_window - patch_ks >= 0
        # => patch_ks <= context_window
        if self.patch_ks > self.context_window:
            original_patch_ks = self.patch_ks
            adjusted_patch_ks = self.context_window

            warnings.warn(
                f"The patch_ks ({original_patch_ks}) is too large for the "
                f"context_window ({self.context_window}). Adjusting patch_ks "
                f"from {original_patch_ks} to {adjusted_patch_ks} to ensure valid "
                f"patch_ks.\n",
                UserWarning,
            )
            self.patch_ks = adjusted_patch_ks

        dataset_len = len(self._y) - self.context_window - fh + 1

        if self.norm == "batch" and dataset_len == 1:
            self.norm = "layer"
            warnings.warn(
                "Normalization automatically switched from 'batch' to 'layer' "
                "because the effective training sample size is 1 "
                "(computed as input_length - context_window - fh + 1 == 1). "
                "\nTo avoid this automatic change, increase your input length "
                "or reduce the context/fh values.\n",
                UserWarning,
            )
        self.n_layers = len(self.dw_ks)
        configs = {
            "enc_in": self._y.shape[-1],
            "seq_len": self.context_window,
            "pred_len": fh,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.n_layers,
            "patch_ks": self.patch_ks,
            "patch_sd": self.patch_sd,
            "dw_ks": self.dw_ks,
            "dropout": self.dropout,
            "head_dropout": self.head_dropout,
            "padding_patch": self.padding_patch,
            "revin": self.revin,
            "affine": self.affine,
            "subtract_last": self.subtract_last,
            "deformable": self.deformable,
            "enable_res_param": self.enable_res_param,
            "re_param": self.re_param,
            "re_param_kernel": self.re_param_kernel,
            "device": self.device,
        }
        model = Model(
            configs,
            norm=self.norm,
            act=self.act,
            random_state=self.random_state,
        )

        # Add required properties for the adapter
        model.seq_len = self.context_window
        model.pred_len = fh

        return model

    def build_pytorch_train_dataloader(self, y):
        """Build PyTorch DataLoader for training with custom batch handling.

        This method handles the case where the last batch has only 1 sample,
        which causes issues with BatchNorm. When using batch normalization and
        the dataset would result in a last batch of size 1, drop_last is set
        to True to discard that single sample.
        """
        from torch.utils.data import DataLoader

        # Use parent's dataset creation logic
        dataset = _pytorch.PyTorchTrainDataset(
            y=y,
            seq_len=self.network.seq_len,
            fh=self._fh.to_relative(self.cutoff)._values[-1],
        )

        # Check if we need to drop the last batch to avoid single sample
        dataset_len = len(dataset)
        batch_size = self.batch_size

        drop_last = (
            self.norm == "batch"
            and dataset_len > self.batch_size
            and dataset_len % self.batch_size == 1
        )
        gen = torch.Generator()
        if self.random_state:
            gen.manual_seed(self.random_state)
        if drop_last:
            warnings.warn(
                "The last batch has only 1 sample which may cause issues with "
                "BatchNorm. Dropping the last batch.\n"
                "To avoid this, consider changing hyperparameters such that the "
                "number of training samples (len(y) - context_window - max(fh) + 1) "
                "is not equal to (batch_size * n + 1) for any integer n.\n",
                UserWarning,
            )
        return DataLoader(
            dataset, batch_size, shuffle=True, drop_last=drop_last, generator=gen
        )

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        params1 = {
            "context_window": 10,
            "patch_ks": 6,
            "patch_sd": 3,
            "num_epochs": 1,
        }

        params2 = {
            "context_window": 8,
            "patch_ks": 8,
            "patch_sd": 4,
            "d_model": 128,
            "d_ff": 512,
            "dropout": 0.1,
            "num_epochs": 1,
        }
        params3 = {
            "context_window": 16,
            "patch_ks": 4,
            "patch_sd": 2,
            "dw_ks": (7, 3),
            "d_model": 32,
            "d_ff": 64,
            "dropout": 0.05,
            "act": "relu",
            "head_dropout": 0.05,
            "padding_patch": 1,
            "revin": False,
            "deformable": False,
            "enable_res_param": False,
            "re_param": False,
            "num_epochs": 1,
        }

        return [params1, params2, params3]

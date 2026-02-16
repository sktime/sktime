"""ConvTimeNet (PyTorch) classifier for time series classification."""

__author__ = ["Tanuj-Taneja1"]
__all__ = ["ConvTimeNetClassifier"]

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch


class ConvTimeNetClassifier(BaseDeepClassifierPytorch):
    """ConvTimeNet for time series classification.

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

    This classifier has been wrapped around implementations from [1]_, [2]_ and [3]_.

    Parameters
    ----------
    d_model : int
        Hidden dimension size for model processing.
    patch_size : int
        Size of patches for sequence splitting.
    patch_stride : int
        Stride length for patch creation.
    dropout : float, optional (default=0)
        Dropout rate to apply to layers.
    d_ff : int, optional (default=128)
        Dimension of feedforward network.
    dw_ks : int or list, optional (default=3)
        Depthwise convolution kernel size(s). Can be a single int or list of ints.
    device : str, optional (default="cpu")
        Device to use for computation ("cpu" or "cuda").
    num_epochs : int, optional (default=16)
        The number of epochs to train the model.
    batch_size : int, optional (default=8)
        The size of each mini-batch during training.
    criterion : callable, optional (default=None)
        The loss function to use. If None, CrossEntropyLoss will be used.
    criterion_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    optimizer : str, optional (default=None)
        The optimizer to use. If None, Adam will be used.
    optimizer_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
    lr : float, optional (default=0.001)
        The learning rate to use for the optimizer.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    random_state : int, optional (default=None)
        Seed to ensure reproducibility.

    Examples
    --------
    >>> from sktime.classification.deep_learning import ConvTimeNetClassifier
    >>> import numpy as np
    >>> # Create a sample multivariate time series dataset
    >>> # 48 samples, 3 variables, length 128
    >>> X = np.random.randn(16 * 3, 3, 128).astype("float32")
    >>> y = np.array([0, 1, 2] * 16)  # 3 classes
    >>> # Create and fit the classifier
    >>> clf = ConvTimeNetClassifier(
    ...     patch_size=4,
    ...     patch_stride=2,
    ...     d_model=64,
    ...     d_ff=128,
    ...     dw_ks=[5, 7, 9],
    ...     batch_size=8,
    ...     device="cpu",
    ...     random_state=10
    ... ) # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    ConvTimeNetClassifier(...)
    >>> # Make predictions
    >>> y_pred = clf.predict(X)  # doctest: +SKIP
    >>> y_proba = clf.predict_proba(X)  # doctest: +SKIP

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
        "python_dependencies": ["torch"],
        # estimator type
        # --------------
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # CI and testing
        # --------------
        "tests:libs": [
            "sktime.networks.convtimenet._convtimenet",
            "sktime.networks.convtimenet._dlutils",
            "sktime.networks.convtimenet._convtimenet_backbone",
        ],
        "tests:skip_by_name": ["test_fit_idempotent"],
    }

    def __init__(
        self,
        d_model,
        patch_size,
        patch_stride,
        dropout=0,
        d_ff=128,
        dw_ks=3,
        device="cpu",
        num_epochs=16,
        batch_size=8,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        verbose=False,
        random_state=None,
    ):
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dropout = dropout
        self.d_ff = d_ff
        # Ensure dw_ks is a list
        if isinstance(dw_ks, int):
            self.dw_ks = [dw_ks]
        else:
            self.dw_ks = dw_ks
        self.device = device

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            verbose=verbose,
            random_state=random_state,
        )

    def _build_network(self, X, y):
        from sktime.networks.convtimenet._convtimenet import ConvTimeNet

        self.n_classes = len(set(y))
        self.enc_in = X.shape[1]
        self.seq_len = X.shape[2]

        model = ConvTimeNet._ConvTimeNet(
            enc_in=self.enc_in,
            d_model=self.d_model,
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            n_classes=self.n_classes,
            dropout=self.dropout,
            d_ff=self.d_ff,
            dw_ks=self.dw_ks,
            device=self.device,
        )
        return model.to(self.device)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        params1 = {
            "d_model": 16,
            "patch_size": 2,
            "patch_stride": 1,
            "dw_ks": [3],
            "d_ff": 16,
            "batch_size": 2,
            "optimizer": "Adam",
            "lr": 1e-3,
            "device": "cpu",
            "verbose": False,
            "dropout": 0.0,
            "num_epochs": 1,
            "random_state": 0,
        }

        params2 = {
            "d_model": 32,
            "patch_size": 4,
            "patch_stride": 2,
            "dw_ks": [5, 7],
            "d_ff": 64,
            "batch_size": 4,
            "optimizer": "Adam",
            "lr": 5e-4,
            "device": "cpu",
            "verbose": False,
            "dropout": 0.1,
            "num_epochs": 2,
            "random_state": 42,
        }

        params3 = {
            "d_model": 64,
            "patch_size": 5,
            "patch_stride": 1,
            "dw_ks": [7, 13, 19],  # very large depthwise kernels
            "d_ff": 128,
            "batch_size": 8,
            "optimizer": "SGD",  # different optimizer
            "lr": 1e-2,
            "device": "cpu",
            "verbose": False,
            "dropout": 0.2,
            "num_epochs": 2,
            "random_state": 123,
        }

        return [params1, params2, params3]

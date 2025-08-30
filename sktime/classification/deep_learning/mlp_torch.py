"""Multi Layer Perceptron Network (MLP) for classification."""

__author__ = ["Jack Russon"]
__all__ = ["MLPTorchClassifier"]

import numpy as np

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch
from sktime.networks.mlp_torch import PyTorchMLPNetwork


class MLPTorchClassifier(BaseDeepClassifierPytorch):
    """Multi Layer Perceptron Network (MLP) for classification.
    
    Parameters
    ----------
    num_epochs : int, default=16
        The number of epochs to train the model.
    batch_size : int, default=8
        The number of samples per gradient update.
    criterion : str, optional, default=None
        Loss function to use. If None, uses CrossEntropyLoss.
    criterion_kwargs : dict, optional, default=None
        Additional arguments for the criterion.
    optimizer : str, optional, default=None
        Optimizer to use. If None, uses Adam.
    optimizer_kwargs : dict, optional, default=None
        Additional arguments for the optimizer.
    lr : float, default=0.001
        Learning rate for the optimizer.
    verbose : bool, default=True
        Whether to output extra information during training.
    random_state : int, optional, default=None
        Seed for random number generation.
    hidden_dims : list, default=[500, 500, 500]
        List of hidden layer dimensions.
    activation : str, default="relu"
        Activation function to use in hidden layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    use_bias : bool, default=True
        Whether to use bias in linear layers.
    """

    def __init__(
        self,
        num_epochs=16,
        batch_size=8,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        verbose=True,
        random_state=None,
        hidden_dims=None,
        activation="relu",
        dropout=0.0,
        use_bias=True,
    ):
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [500, 500, 500]
        
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.use_bias = use_bias
        
        # Call parent constructor with the parameters it expects
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
        self.criterions = {}

    def _build_network(self, X, y):
        # Validate input shape - X should be 3D (n_instances, n_dims, series_length)
        
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is properly formatted."
            )
        input_shape = X.shape[1:]  # (n_dims, series_length)
        num_classes = len(np.unique(y))
        
        # Build the complete network with the correct number of output classes
        return PyTorchMLPNetwork(
            input_shape, 
            num_classes,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_bias=self.use_bias
        )
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params = [
            {
                "num_epochs": 3,
                "batch_size": 4,
                "hidden_dims": [100, 50],
                "activation": "relu",
                "dropout": 0.0,
                "use_bias": True,
                "random_state": 42,
            },
            {
                "num_epochs": 2,
                "batch_size": 2,
                "hidden_dims": [64],
                "activation": "tanh",
                "dropout": 0.1,
                "use_bias": False,
                "optimizer": "AdamW",
                "lr": 0.01,
                "random_state": 0,
            },
        ]
        return params
        

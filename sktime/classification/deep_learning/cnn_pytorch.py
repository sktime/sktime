"""Time Convolutional Neural Network (CNN) for classification."""

__author__ = ["James-Large", "AurumnPegasus"]
__all__ = ["CNNClassifier"]

import numpy as np
from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class CNNNetwork(nn.Module):
        """
        Establish the network structure for a CNN.

        Adapted from the implementation used in [1]_.

        Parameters
        ----------
        n_layers : int, default = 2
            the number of convolutional plus average pooling layers
        kernel_size : int, default = 7
            specifying the length of the 1D convolution window
        n_filters : int or list of int, default = [6, 12]
            size of filter for each conv layer
        avg_pool_size : int, default = 3
            size of the average pooling windows
        activation : string, default = "sigmoid"
            activation function used for hidden layers
        padding : string, default = "valid"
            Controls padding logic for the convolutional layers
        """

        def __init__(
            self,
            n_layers=2,
            kernel_size=7,
            n_filters=None,
            avg_pool_size=3,
            activation="sigmoid",
            padding="valid",
            n_channels=1,
            n_classes=2,  # Added for final layer
            random_state=None,
        ):
            super().__init__()
            self.n_layers = n_layers
            self.kernel_size = kernel_size
            self.n_filters = n_filters if n_filters is not None else [6, 12]
            self.avg_pool_size = avg_pool_size
            self.activation = activation
            self.padding = padding
            self.n_channels = n_channels
            self.n_classes = n_classes

            # Ensure n_filters list matches n_layers
            if len(self.n_filters) < self.n_layers:
                self.n_filters = self.n_filters + [self.n_filters[-1]] * (
                    self.n_layers - len(self.n_filters)
                )
            
            # Define activation function
            if activation == "sigmoid":
                self.act_fn = nn.Sigmoid()
            elif activation == "relu":
                self.act_fn = nn.ReLU()
            else:
                 # Default or fallback
                self.act_fn = nn.Sigmoid()

            self.layers = nn.ModuleList()
            
            # Input channels for first layer
            in_channels = n_channels

            # Create Convolutional + Pooling Layers
            for i in range(self.n_layers):
                out_channels = self.n_filters[i]
                
                # Padding logic (PyTorch implementation of 'same' vs 'valid' is manual or via padding arg)
                # simpler approach: rely on nn.Conv1d padding logic. 
                # 'valid' -> 0, 'same' -> 'same' (in newer torch) or calculated.
                # sktime usage often implies specific padding. Let's start basic.
                pad_arg = 0
                if self.padding == "same":
                    pad_arg = "same"
                elif self.padding == "valid":
                     pad_arg = 0 # valid means no padding
                
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=pad_arg
                )
                
                pool = nn.AvgPool1d(kernel_size=self.avg_pool_size)
                
                self.layers.append(conv)
                self.layers.append(self.act_fn) # Activation is part of the layer block in Keras
                self.layers.append(pool)
                
                in_channels = out_channels

            self.flatten = nn.Flatten()
            
            # Linear layer needs input size calculation or lazy linear
            # For simplicity in this migration step, we can use LazyLinear if torch version supports it, 
            # OR typically we calculate it. 
            # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html
            self.output_layer = nn.LazyLinear(n_classes) 
            # Note: LazyLinear infers input shape on first pass.

        def forward(self, x):
            # x shape: (batch, channels, length)
            for layer in self.layers:
                x = layer(x)
            
            x = self.flatten(x)
            x = self.output_layer(x)
            return x

else:
    class CNNNetwork:
        """Dummy class if torch is unavailable."""

class CNNClassifierTorch(BaseDeepClassifier):
    """Time Convolutional Neural Network (CNN).

    Parameters
    ----------
    n_epochs : int, default = 2000
    batch_size : int, default = 16
    kernel_size : int, default = 7
    avg_pool_size : int, default = 3
    n_conv_layers : int, default = 2
    callbacks : list of keras.callbacks, default = None
    verbose : boolean, default = False
    loss : string, default="cross_entropy"
    metrics : list of strings, default=["accuracy"],
    random_state : int or None, default=None
    activation : string, default="sigmoid"
    use_bias : boolean, default = True
    optimizer : torch.optim, default = None
    """

    _tags = {
        "authors": ["James-Large", "AurumnPegasus"],
        "maintainers": ["James-Large"],
        "python_dependencies": "torch",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        callbacks=None,
        verbose=False,
        loss="cross_entropy",
        metrics=None,
        random_state=None,
        activation="sigmoid",
        use_bias=True,
        optimizer=None,
        padding="valid"
    ):
        super().__init__()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.padding = padding
        self.history = None
        self._network = None

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.
        y : np.ndarray of shape n
            The training data class labels.

        Returns
        -------
        self : object
        """
        # 1. Encode labels
        self.label_encoder_ = None
        if not all(isinstance(yi, (int, float)) for yi in y):
             from sklearn.preprocessing import LabelEncoder
             self.label_encoder_ = LabelEncoder()
             y = self.label_encoder_.fit_transform(y)
        
        # 2. Convert to PyTorch tensors
        # X is (n, d, m), so we transpose to (n, m, d) for Dataset but wait...
        # CNN usually expects (Batch, Channels, Length).
        # sktime X is (Batch, Dimensions, Length).
        # Our CNNNetwork expects (Batch, Channels, Length).
        # So X is already in correct shape (n, d, m).
        
        # 3. Instantiate Network
        self.input_shape_ = X.shape[1:] # (d, m)
        n_channels = X.shape[1]
        n_classes = len(np.unique(y))
        
        self._network = CNNNetwork(
            n_layers=self.n_conv_layers,
            kernel_size=self.kernel_size,
            n_filters=None, #/Using default logic in network
            avg_pool_size=self.avg_pool_size,
            activation=self.activation,
            padding=self.padding,
            n_channels=n_channels,
            n_classes=n_classes,
            random_state=self.random_state,
        )

        # 4. Define Loss and Optimizer
        if self.loss == "cross_entropy":
            self.criterion_ = nn.CrossEntropyLoss()
        else:
            self.criterion_ = nn.CrossEntropyLoss() # Default fallback

        if self.optimizer is None:
            self.optimizer_ = torch.optim.Adam(self._network.parameters(), lr=0.001)
        else:
             self.optimizer_ = self.optimizer
        
        # 5. Create DataLoader
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float),
            torch.tensor(y, dtype=torch.long)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 6. Training Loop
        self._network.train()
        for epoch in range(self.n_epochs):
            for i, (inputs, targets) in enumerate(dataloader):
                # Zero the parameter gradients
                self.optimizer_.zero_grad()

                # Forward + backward + optimize
                outputs = self._network(inputs)
                loss = self.criterion_(outputs, targets)
                loss.backward()
                self.optimizer_.step()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}")

        return self

    def _predict(self, X, **kwargs):
        self._network.eval()
        with torch.no_grad():
             X_tensor = torch.tensor(X, dtype=torch.float)
             outputs = self._network(X_tensor)
             _, predicted = torch.max(outputs.data, 1)
             y_pred = predicted.numpy()
             
        if self.label_encoder_ is not None:
             return self.label_encoder_.inverse_transform(y_pred)
        return y_pred

    def _predict_proba(self, X, **kwargs):
        self._network.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float)
            outputs = self._network(X_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.numpy()

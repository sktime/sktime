"""PyTorch implementation of CNN classifier."""

__author__ = ["Rklearns"]
__all__ = ["CNNClassifierTorch"]

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from sktime.classification.deep_learning.base import BaseDeepClassifier


class CNNNetwork:
    """PyTorch CNN Network for time series classification."""

    def __init__(
        self,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        filter_sizes=None,
        activation="relu",
        padding="auto",
        random_state=None,
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes or [6, 12]
        self.padding = padding

        # Will be built dynamically
        self.conv_layers = None
        self.global_pool = None

    def build_network(self, input_shape):
        """Build the CNN architecture."""
        import torch.nn as nn

        layers = []
        in_channels = input_shape[-1]  # Features dimension

        for i in range(self.n_conv_layers):
            out_channels = (
                self.filter_sizes[i]
                if i < len(self.filter_sizes)
                else self.filter_sizes[-1]
            )

            # Determine padding based on original logic
            if self.padding == "auto":
                pad = "same" if input_shape[0] < 60 else "valid"
            else:
                pad = self.padding

            padding_val = self.kernel_size // 2 if pad == "same" else 0

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels, out_channels, self.kernel_size, padding=padding_val
                    ),
                    nn.ReLU(),
                    nn.AvgPool1d(self.avg_pool_size, padding=self.avg_pool_size // 2),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        return in_channels  # Return number of features for classifier

    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch, sequence, features) → (batch, features, sequence)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)  # Remove last dimension
        return x


class CNNClassifierTorch(BaseDeepClassifier):
    """Time Convolutional Neural Network (CNN) using PyTorch.

    This is a drop-in replacement for CNNClassifier using PyTorch instead of TensorFlow.

    Parameters
    ----------
    n_epochs       : int, default = 2000
        the number of epochs to train the model
    batch_size      : int, default = 16
        the number of samples per gradient update.
    kernel_size     : int, default = 7
        the length of the 1D convolution window
    avg_pool_size   : int, default = 3
        size of the average pooling windows
    n_conv_layers   : int, default = 2
        the number of convolutional plus average pooling layers
    callbacks       : list, default = None
        callbacks for training (currently not implemented for PyTorch)
    verbose         : boolean, default = False
        whether to output extra information
    loss            : string, default="categorical_crossentropy"
        loss function name (mapped to PyTorch equivalent)
    metrics         : list of strings, default=["accuracy"],
        metrics to track during training
    random_state    : int or None, default=None
        Seed for random number generation.
    activation      : string, default="softmax"
        Activation function used in the output linear layer.
    use_bias        : boolean, default = True
        whether the layer uses a bias vector.
    optimizer       : torch.optim object, default = Adam(lr=0.01)
        specify the optimizer and the learning rate to be used.
    filter_sizes    : array of shape (n_conv_layers) default = [6, 12]
        number of filters for each convolutional layer
    padding : string, default = "auto"
        Controls padding logic for the convolutional layers

    References
    ----------
    .. [1] Zhao et. al, Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from sktime.classification.deep_learning.cnn import CNNClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> cnn = CNNClassifierTorch(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> cnn.fit(X_train, y_train)  # doctest: +SKIP
    CNNClassifierTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["hfawaz", "James-Large", "Rklearns"],
        "maintainers": ["James-Large", "Rklearns"],
        "python_dependencies": "torch",
        # estimator type handled by parent class
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
        loss="categorical_crossentropy",
        metrics=None,
        random_state=None,
        activation="softmax",
        use_bias=True,
        optimizer=None,
        filter_sizes=None,
        padding="auto",
    ):
        # Check for PyTorch dependencies
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PyTorch (torch) is required for CNNClassifierTorch. "
                "Please install with: pip install torch"
            ) from e

        self.batch_size = batch_size
        self.n_conv_layers = n_conv_layers
        self.avg_pool_size = avg_pool_size
        self.kernel_size = kernel_size
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None
        self.filter_sizes = filter_sizes
        self.padding = padding

        if metrics is None:
            self.metrics = ["accuracy"]
        else:
            self.metrics = metrics

        if filter_sizes is None:
            self.filter_sizes = [6, 12]

        super().__init__()

        self._network = CNNNetwork(
            kernel_size=self.kernel_size,
            avg_pool_size=self.avg_pool_size,
            n_conv_layers=self.n_conv_layers,
            filter_sizes=self.filter_sizes,
            activation=self.activation,
            padding=self.padding,
            random_state=self.random_state,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, PyTorch model ready for training."""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Build the network and get output features
        n_features = self._network.build_network(input_shape)

        # Create classifier layer
        self.classifier = nn.Linear(n_features, n_classes, bias=self.use_bias)

        # Create complete model
        class CNNModel(nn.Module):
            def __init__(self, network, classifier):
                super().__init__()
                self.network = network
                self.classifier = classifier

            def forward(self, x):
                x = self.network.forward(x)
                x = self.classifier(x)
                return x

        model = CNNModel(self._network, self.classifier)

        # Set up optimizer
        self.optimizer_ = (
            optim.Adam(model.parameters(), lr=0.01)
            if self.optimizer is None
            else self.optimizer
        )

        return model

    def _convert_y_to_index(self, y):
        """Convert class labels to indices."""
        if not hasattr(self, "label_encoder_"):
            self.label_encoder_ = LabelEncoder()
            y_indices = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
        else:
            y_indices = self.label_encoder_.transform(y)

        return y_indices

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y)."""
        import torch
        import torch.nn as nn

        # Convert labels to indices
        y_indices = self._convert_y_to_index(y)

        # Transpose to conform to PyTorch input style (batch, sequence, features)
        X = X.transpose(0, 2, 1)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y_indices)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            print("Model architecture:")
            print(self.model_)
            total_params = sum(p.numel() for p in self.model_.parameters())
            print(f"Total parameters: {total_params}")

        # Set up loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model_.train()
        self.history = {"loss": [], "accuracy": []}

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # Batch training
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i : i + self.batch_size]
                batch_y = y_tensor[i : i + self.batch_size]

                self.optimizer_.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer_.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += batch_y.size(0)
                epoch_correct += (predicted == batch_y).sum().item()

            # Record metrics
            avg_loss = epoch_loss / (len(X_tensor) // self.batch_size + 1)
            accuracy = epoch_correct / epoch_total
            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(accuracy)

            if self.verbose and (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.n_epochs}], "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
                )

        return self

    def _predict(self, X):
        """Predict class labels for samples in X."""
        import torch

        X = X.transpose(0, 2, 1)
        X_tensor = torch.FloatTensor(X)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return self.classes_[predicted.numpy()]

    def _predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        import torch

        X = X.transpose(0, 2, 1)
        X_tensor = torch.FloatTensor(X)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.numpy()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
            "avg_pool_size": 4,
        }

        param2 = {
            "n_epochs": 12,
            "batch_size": 6,
            "kernel_size": 2,
            "n_conv_layers": 1,
        }

        param3 = {
            "n_epochs": 2,
            "batch_size": 4,
            "use_bias": False,
        }

        test_params = [param1, param2, param3]

        if parameter_set == "results_comparison":
            return param3
        else:
            return test_params

"""Time Recurrent Neural Network (RNN) for classification in PyTorch."""

__author__ = ["RecreationalMath"]
__all__ = ["SimpleRNNClassifierTorch"]

from sktime.networks.rnn import RNNNetworkTorch
from sktime.sktime.classification.deep_learning.base import BaseDeepClassifierTorch


class SimpleRNNClassifierTorch(BaseDeepClassifierTorch):
    """Simple recurrent neural network in PyTorch.

    Parameters
    ----------
    n_epochs : int, default = 100
        the number of epochs to train the model
    batch_size : int, default = 1
        the number of samples per gradient update.
    hidden_dim : int, default = 6
        Number of features in the hidden state
    random_state : int or None, default=0
        Seed for random number generation.
    verbose : boolean, default = False
        whether to output extra information
    loss : torch.nn loss function, default = nn.MSELoss()
        loss function to be used in training the neural network.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
    metrics : list of str, default = ["accuracy"]
        List of metrics to be used for evaluation.
    activation : str, default = "relu"
        Activation function to be used in the output layer.
        List of available activation functions:
        https://pytorch.org/docs/stable/nn.html#activation-functions
    bias : bool, default = True
        If False, then the layer does not use bias weights.
    optimizer : torch.optim object, default = torch.optim.RMSprop(lr=0.001)
        specify the optimizer and the learning rate to be used.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    """

    _tags = {
        # packaging info
        # --------------
        "author": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        n_epochs=100,
        batch_size=1,
        units=6,
        random_state=0,
        verbose=False,
        criterion=None,
        optimizer=None,
    ):
        self.units = units
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
            verbose=verbose,
            criterion=criterion,
            optimizer=optimizer,
        )

    def build_network(self, input_shape, **kwargs):
        """Build the RNN network.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (n_timepoints, n_dimensions)
        **kwargs :
            Additional keyword arguments for the network.

        Returns
        -------
        model : RNNNetworkTorch instance
            The constructed RNN network.
        """
        n_timepoints, n_dimensions = input_shape
        model = RNNNetworkTorch(
            input_size=n_dimensions,
            hidden_size=self.units,
            num_layers=1,
            random_state=self.random_state,
        )
        return model

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {"n_epochs": 50, "batch_size": 2, "hidden_size": 5, "bias": False}

        return [params1, params2]

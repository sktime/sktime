"""Abstract base class for the Pytorch neural network classifiers."""

__authors__ = ["geetu040", "RecreationalMath"]

__all__ = ["BaseDeepClassifierPytorch"]

import abc

import numpy as np
from sklearn.preprocessing import LabelEncoder

from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _safe_import


class BaseDeepClassifierPytorch(BaseClassifier):
    """Abstract base class for the Pytorch neural network classifiers."""

    _tags = {
        "authors": ["geetu040", "RecreationalMath"],
        "maintainers": ["geetu040", "RecreationalMath"],
        "python_dependencies": ["torch"],
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "capability:multioutput": False,
    }

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
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # use this when y has str
        self.label_encoder = None
        super().__init__()

        # set random seed for torch
        if self.random_state is not None:
            torchManual_seed = _safe_import("torch.manual_seed")
            torchManual_seed(self.random_state)

        # optimizers and criterions will be instantiated in
        # _instantiate_optimizer & _instantiate_criterion methods respectively
        self._all_optimizers = None
        self._all_criterions = None

    def _fit(self, X, y):
        y = self._encode_y(y)

        self.network = self._build_network(X, y)

        # instantiate loss function and optimizer
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        dataloader = self._build_dataloader(X, y)

        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

    def _run_epoch(self, epoch, dataloader):
        losses = []
        for inputs, outputs in dataloader:
            y_pred = self.network(**inputs)
            loss = self._criterion(y_pred, outputs)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())
        if self.verbose:
            print(f"Epoch {epoch + 1}: Loss: {np.average(losses)}")

    def _instantiate_optimizer(self):
        if self._all_optimizers is None:
            self._all_optimizers = {
                "adadelta": _safe_import("torch.optim.Adadelta"),
                "adagrad": _safe_import("torch.optim.Adagrad"),
                "adam": _safe_import("torch.optim.Adam"),
                "adamw": _safe_import("torch.optim.AdamW"),
                "sparseadam": _safe_import("torch.optim.SparseAdam"),
                "adamax": _safe_import("torch.optim.Adamax"),
                "asgd": _safe_import("torch.optim.ASGD"),
                "lbfgs": _safe_import("torch.optim.LBFGS"),
                "nadam": _safe_import("torch.optim.NAdam"),
                "radam": _safe_import("torch.optim.RAdam"),
                "rmsprop": _safe_import("torch.optim.RMSprop"),
                "rprop": _safe_import("torch.optim.Rprop"),
                "sgd": _safe_import("torch.optim.SGD"),
            }
        # if no optimizer is passed, use Adam as default
        if not self.optimizer:
            return self._all_optimizers["adam"](self.network.parameters(), lr=self.lr)
        # import the base class for all optimizers in PyTorch
        torchOptimizer = _safe_import("torch.optim.Optimizer")
        # if optimizer is a string, look it up in the available optimizers
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() in self._all_optimizers:
                if self.optimizer_kwargs:
                    return self._all_optimizers[self.optimizer.lower()](
                        self.network.parameters(), lr=self.lr, **self.optimizer_kwargs
                    )
                else:
                    return self._all_optimizers[self.optimizer.lower()](
                        self.network.parameters(), lr=self.lr
                    )
            else:
                raise ValueError(
                    f"Unknown optimizer: {self.optimizer}. Please pass one of "
                    f"{', '.join(self._all_optimizers)} for `optimizer`."
                )
        # if optimizer is already an instance of torch.optim.Optimizer, use it directly
        elif isinstance(self.optimizer, torchOptimizer):
            return self.optimizer
        # if optimizer is neither a string nor an instance of
        # a valid PyTorch optimizer, raise an error
        else:
            raise TypeError(
                "`optimizer` can either be None, a str or an instance of "
                "optimizers defined in torch.optim. "
                "See https://pytorch.org/docs/stable/optim.html#algorithms. "
                f"But got {type(self.optimizer)} instead."
            )

    def _instantiate_criterion(self):
        if self._all_criterions is None:
            self._all_criterions = {
                "l1loss": _safe_import("torch.nn.L1Loss"),
                "mseloss": _safe_import("torch.nn.MSELoss"),
                "crossentropyloss": _safe_import("torch.nn.CrossEntropyLoss"),
                "ctcloss": _safe_import("torch.nn.CTCLoss"),
                "nllloss": _safe_import("torch.nn.NLLLoss"),
                "poissonnllloss": _safe_import("torch.nn.PoissonNLLLoss"),
                "gaussiannllloss": _safe_import("torch.nn.GaussianNLLLoss"),
                "kldivloss": _safe_import("torch.nn.KLDivLoss"),
                "bceloss": _safe_import("torch.nn.BCELoss"),
                "bcewithlogitsloss": _safe_import("torch.nn.BCEWithLogitsLoss"),
                "marginrankingloss": _safe_import("torch.nn.MarginRankingLoss"),
                "hingeembeddingloss": _safe_import("torch.nn.HingeEmbeddingLoss"),
                "multilabelmarginloss": _safe_import("torch.nn.MultiLabelMarginLoss"),
                "huberloss": _safe_import("torch.nn.HuberLoss"),
                "smoothl1loss": _safe_import("torch.nn.SmoothL1Loss"),
                "softmarginloss": _safe_import("torch.nn.SoftMarginLoss"),
                "multilabelsoftmarginloss": _safe_import(
                    "torch.nn.MultiLabelSoftMarginLoss"
                ),  # noqa: E501
                "cosineembeddingloss": _safe_import("torch.nn.CosineEmbeddingLoss"),
                "multimarginloss": _safe_import("torch.nn.MultiMarginLoss"),
                "tripletmarginloss": _safe_import("torch.nn.TripletMarginLoss"),
                "tripletmarginwithdistanceloss": _safe_import(
                    "torch.nn.TripletMarginWithDistanceLoss"
                ),  # noqa: E501
            }

        # if no criterion is passed, use CrossEntropyLoss as default
        if not self.criterion:
            return self._all_criterions["crossentropyloss"]()
        # import the base class for all loss functions in PyTorch
        torchLossFunction = _safe_import("torch.nn.modules.loss._Loss")
        # if criterion is a string, look it up in the available criterions
        if isinstance(self.criterion, str):
            if self.criterion.lower() in self._all_criterions:
                if self.criterion_kwargs:
                    return self._all_criterions[self.criterion.lower()](
                        **self.criterion_kwargs
                    )
                else:
                    return self._all_criterions[self.criterion.lower()]()
            else:
                raise ValueError(
                    f"Unknown criterion: {self.criterion}. Please pass one of "
                    f"{', '.join(self._all_criterions)} for `criterion`."
                )
        # if criterion is already an instance of torch.nn.modules.loss._Loss, use it
        elif isinstance(self.criterion, torchLossFunction):
            return self.criterion
        else:
            # if criterion is neither a string nor an instance of
            # a valid PyTorch loss function, raise an error
            raise TypeError(
                "`criterion` can either be None, a str or an instance of "
                "loss functions defined in "
                "https://pytorch.org/docs/stable/nn.html#loss-functions "
                f"But got {type(self.criterion)} instead."
            )

    @abc.abstractmethod
    def _build_network(self):
        pass

    def _build_dataloader(self, X, y=None):
        # default behaviour if estimator doesnot implement
        # dataloader of its own
        dataset = PytorchDataset(X, y)
        DataLoader = _safe_import("torch.utils.data.DataLoader")
        return DataLoader(dataset, self.batch_size)

    def _predict(self, X):
        """Predict labels for sequences in X.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : should be of mtype in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            predicted class labels
            indices correspond to instance indices in X
            if self.get_tag("capaility:multioutput") = False, should be 1D
            if self.get_tag("capaility:multioutput") = True, should be 2D
        """
        y_pred_prob = self._predict_proba(X)
        y_pred = np.argmax(y_pred_prob, axis=-1)
        y_decoded = self._decode_y(y_pred)
        return y_decoded

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        Fsoftmax = _safe_import("torch.nn.functional.softmax")
        cat = _safe_import("torch.cat")

        self.network.eval()
        dataloader = self._build_dataloader(X)
        y_pred = []
        for inputs in dataloader:
            y_pred.append(self.network(**inputs).detach())
        y_pred = cat(y_pred, dim=0)
        # (batch_size, num_outputs)
        y_pred = Fsoftmax(y_pred, dim=-1)
        y_pred = y_pred.numpy()
        return y_pred

    def _encode_y(self, y):
        unique = np.unique(y)
        if np.array_equal(unique, np.arange(len(unique))):
            return y

        self.label_encoder = LabelEncoder()
        return self.label_encoder.fit_transform(y)

    def _decode_y(self, y):
        if self.label_encoder is None:
            return y

        return self.label_encoder.inverse_transform(y)

    def _internal_convert(self, X, y=None):
        """Override to enforce strict 3D input validation for PyTorch classifiers.

        PyTorch classifiers require 3D input and we don't allow automatic conversion
        from 2D to 3D as this can mask user errors and lead to unexpected behavior.
        """
        if isinstance(X, np.ndarray) and X.ndim != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. PyTorch classifiers require properly "
                f"formatted 3D time series data. Please reshape your data or "
                "use a supported Panel mtype."
            )

        # Call parent method for other conversions
        return super()._internal_convert(X, y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return []


Dataset = _safe_import("torch.utils.data.Dataset")


class PytorchDataset(Dataset):
    """Dataset for use in sktime deep learning classifier based on pytorch."""

    def __init__(self, X, y=None):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        torchTensor = _safe_import("torch.tensor")
        torchFLoat = _safe_import("torch.float")
        torchLong = _safe_import("torch.long")
        x = self.X[i]
        x = torchTensor(x, dtype=torchFLoat)
        inputs = {"X": x}
        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torchTensor(y, dtype=torchLong)
        return inputs, y

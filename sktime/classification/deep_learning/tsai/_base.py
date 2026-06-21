"""Base class for tsai-based time series classifiers."""

__author__ = ["obaidsafi51"]
__all__ = ["BaseTsaiClassifier"]


import abc 
import numpy as np 
from sktime.utils.dependencies import _check_dl_dependencies
from sktime.classification.base import BaseClassifier



class BaseTsaiClassifier(BaseClassifier):
    """
    Abstract base class wrapping tsai models for sktime classification.
    
    Handles all data conversion , label encoding , training,loop and inference. Subclasses only need to implement 

    `` _build_model``.

    Note: _DelegatedClassifier was considered but not used here because
    tsai's TSClassifier uses fastai's Learner API (get_X_preds, fit_one_cycle)
    which is not sklearn-compatible. Manual delegation is necessary to
    handle label encoding, binary prob shape correction, and vocab passing.

    Parameters 

    ------------

    n_epoches : int , default = 16
        Number of epochs for fit_one_cycle.
    batch_size : int , default = 16
        Batch size for training.
    lr : float, default = 1e-3
        Maximum learning rate for fit_one_cycle.
    valid_size : float, default = 0.2 
        Fraction of training data used for internal validation.
    random_state : int or None , default = None
        Seed for reproducibility.
    verbose : bool, default = False
        Whether to print training progress

    """

    _tags = {
        "python_dependencies" : ["tsai", "fastai"],
        "X_inner_mtype" : "numpy3D",
        "y_inner_mtype" : "numpy1D",
        "capability:multivariate" : True,
        "capability:unequal_length" : False,
        "capability:random_state": True, 
    }

    def __init__(
        self,
        n_epochs = 16,
        batch_size = 16,
        lr = 0.001,
        valid_size = 0.2,
        random_state = None,
        verbose = False,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.valid_size = valid_size
        self.random_state = random_state
        self.verbose = verbose
        super().__init__()
    
    def _fit(self,X,y):
        """ Fit the classifier

        Parameters 

        -----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)

        y : np.ndarray of shape (n_instances,) - string or integer labels

        Returns
        -------

        self
        """
        import random
        import torch

        # seed all sources of randomness for idempotent fit
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        _check_dl_dependencies("tsai", severity="error")
        from tsai.all import TSClassifier, get_splits
        from sklearn.preprocessing import LabelEncoder


        # Endocing labels to integers

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # vocab forces tsai to use CrossEntropyLoss with n_classes outputs
        # without this, tsai uses BCELoss with 1 output for any integer-encoded y

        vocab = list(range(self.n_classes_))


        # splits - tsai needs train/valid indices 
        random_state = self.random_state if self.random_state is not None else 42
        splits  = get_splits(
            y_enc,
            valid_size = self.valid_size,
            stratify = True,
            random_state = random_state,
            show_plot = False,
        )

        # X is already (n_instances, n_dims, n_timespoints) 
        #tsai expects the same axis order - no transpose needed 

        X = X.astype(np.float32)

        model = self._build_model(
            n_vars =X.shape[1],
            n_classes = self.n_classes_,
        )

        self.learn_ = TSClassifier(
            X,
            y_enc,
            splits = splits,
            arch = model.__class__,
            arch_config = self._get_arch_config(),
            batch_size = self.batch_size,
            metrics = None,
            vocab = vocab,
            verbose = self.verbose,
        )

        self.learn_.fit_one_cycle(self.n_epochs, lr_max = self.lr)
        return self
    
    def _predict_proba(self, X):
        """Return class probability estimates.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)

        Returns
        -------
        probs : np.ndarray of shape (n_instances, n_classes)
        """
        X = X.astype(np.float32)
        probs, *_ = self.learn_.get_X_preds(X)
        probs = probs.numpy()

        if probs.shape[1] == 1:
            probs = np.hstack([1-probs, probs])

        probs = probs / probs.sum(axis = 1, keepdims =True)

        return probs
    
    def _predict(self, X):
        
        """Return predicted class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)

        Returns
        -------
        y : np.ndarray of shape (n_instances,)
        """

        probs = self._predict_proba(X)
        indices = np.argmax(probs, axis = 1)
        
        return self._label_encoder.inverse_transform(indices)
    
    
    @abc.abstractmethod
    def _build_model(self, n_vars, n_class):
        """Instantiate and return the tsai model (nn.Module).

        Parameters
        ----------
        n_vars : int - number of dimensions in X
        n_classes : int - number of target classes

        Returns
        -------
        model : a tsai/torch nn.Module instance
        """

    def _get_arch_config(self):
        """Return architecture kwargs passed to TSClassifier arch_config.

        Override in subclasses to pass model-specific hyperparameters.

        Returns
        -------
        dict
        """
        return {}
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return  {"n_epochs" : 1, "batch_size": 4}


"""LSTM-based time series anomaly detector via the PyOD adapter."""

import numpy as np
import pandas as pd

from sktime.detection.adapters import PyODDetector
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
pyod = _safe_import("pyod")


class LSTMAD(PyODDetector):
    """LSTM-based time series anomaly detector via the PyOD adapter.

    Trains a stacked LSTM to predict the next timestep, then scores
    each timestamp by the Mahalanobis distance of its prediction error
    from a multivariate Gaussian fitted on training errors.

    Note: This implemenation uses PyOD LSTMAD.

    Parameters
    ----------
    window_size : int, default=50
        Number of past timesteps used as input context for prediction.

    hidden_size : int, default=64
        Number of hidden units in each LSTM layer.

    n_layers : int, default=2
        Number of stacked LSTM layers.

    epochs : int, default=50
        Number of training epochs.

    lr : float, default=1e-3
        Learning rate for Adam optimizer.

    batch_size : int, default=32
        Mini-batch size for training.

    contamination : float, default=0.1
        Expected proportion of outliers. Must be in ``(0, 0.5]``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.detection.lstm import LSTMAD
    >>> X = pd.DataFrame(np.random.randn(40, 1))
    >>> detector = LSTMAD(window_size=5, hidden_size=8, n_layers=1, epochs=1)
    >>> y = detector.fit_predict(X)  # doctest: +SKIP

    References
    ----------
    .. [1] Malhotra, P., Vig, L., Shroff, G. and Agarwal, P., 2015.
       Long short term memory networks for anomaly detection in time series.
       In *Proceedings of the European Symposium on Artificial Neural Networks*
       (ESANN).
    """

    _tags = {
        "authors": ["Yue Zhao"],
        "maintainers": ["tanuj-taneja1"],
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "python_dependencies": ["pyod", "torch"],
        "tests:core": True,
    }

    def __init__(
        self,
        window_size=50,
        hidden_size=64,
        n_layers=2,
        epochs=50,
        lr=1e-3,
        batch_size=32,
        contamination=0.1,
    ):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.contamination = contamination

        from pyod.models.ts_lstm import LSTMAD as PyODLSTMAD

        estimator = PyODLSTMAD(
            window_size=window_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            contamination=contamination,
        )
        super().__init__(estimator=estimator)

    @staticmethod
    def _to_lstmad_array(X):
        """Convert a DataFrame or numpy array to the shape LSTMAD expects.

        LSTMAD expects univariate data to be shape (n,) and multivariate data to be
        shape (n, k). This method converts DataFrames to numpy arrays and flattens
        univariate data from (n, 1) to (n,).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        -------
        X_np : np.ndarray, shape (n_timestamps,) or (n_timestamps, n_channels)
        """
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Flatten univariate (n, 1) → (n,)
        if X_np.ndim == 2 and X_np.shape[1] == 1:
            X_np = X_np.ravel()

        return X_np

    def _fit(self, X, y=None):
        """Fit to training data.

        Overrides _fit to reshape input data properly for LSTMAD.
        The base version keeps univariate data as (n, 1), but LSTMAD needs (n,).

        Parameters
        ----------
        X : pd.DataFrame
            Training time series. Univariate ``(n, 1)`` or multivariate
            ``(n, k)`` DataFrames are both supported.
        y : ignored

        Returns
        -------
        self
        """
        from sklearn.base import clone

        X_np = self._to_lstmad_array(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_np)
        return self

    def _predict(self, X):
        """Predict anomaly labels on test data.

        Overrides _predict to reshape input data properly for LSTMAD.
        The base version keeps univariate data as (n, 1), but LSTMAD needs (n,).

        Parameters
        ----------
        X : pd.DataFrame
            Time series to score.

        Returns
        -------
        Y : pd.Series
            Sparse series of anomaly indicators or scores, indexed by the
            anomalous positions in ``X``.
        """
        X_np = self._to_lstmad_array(X)

        Y_binary = self.estimator_.predict(X_np)

        anomaly_locs = np.where(Y_binary)[0]
        return pd.Series(anomaly_locs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        params : list of dict
        """
        params1 = {
            "window_size": 1,
            "hidden_size": 8,
            "n_layers": 1,
            "epochs": 1,
            "batch_size": 4,
            "contamination": 0.1,
        }
        params2 = {
            "window_size": 2,
            "hidden_size": 16,
            "n_layers": 2,
            "epochs": 2,
            "batch_size": 8,
            "contamination": 0.2,
        }
        return [params1, params2]

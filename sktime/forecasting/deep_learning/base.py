# -*- coding: utf-8 -*-
"""Base Deep Forecaster."""
__author__ = ["AurumnPegasus"]
__all__ = ["BaseDeepForecastor"]

from abc import ABC, abstractmethod

import numpy as np

from sktime.forecasting.base import BaseForecaster


class BaseDeepForecastor(BaseForecaster, ABC):
    """Abstract base class for deep learning time series forecasters.

    The base classifier provides a deep learning default method for
    _predict, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    batch_size : int, default = 4
        training batch size for the model

    Arguments
    ---------
    self.model = None

    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(self, batch_size=4, random_state=None):
        super(BaseDeepForecastor).__init__()

        self.batch_size = batch_size
        self.model_ = None

    @abstractmethod
    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes: int
            The number of classes, which shall become the size of the output
            layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def _predict(self, fh, X=None):
        """Get predictions for steps mentioned in fh based on given y and X.

        Parameters
        ----------
        fh: list of int
            Forecasting Horizon for the forecaster.
        X: np.ndarray of shape = (n_instances (n), exog_dimensions (d))
            Exogeneous data for data prediction.

        Returns
        -------
        fvalues: list with predictions of relevant fh.
        """
        currentPred = 1
        lastPred = max(fh)
        fvalues = []
        fh = set(fh)
        source = self.source[-1]
        source = source[np.newaxis, :, :]
        while currentPred <= lastPred:
            yhat = self.model_.predict(source)
            source = np.delete(source, axis=2, obj=0)
            source = np.insert(source, obj=source.shape[-1], values=yhat, axis=-1)
            if currentPred in fh:
                fvalues.append(yhat)

            currentPred += 1
        return fvalues

    def splitSeq(self, steps, seq):
        """Get window sized instances of sequence.

        Parameters
        ----------
        steps: int
            Window Size of the forecaster.
        seq: np.ndarray of shape = (n_instances (n), n_dimensions (d))
            Data to split in window-sized instances.

        Returns
        -------
        source: list containing the data on which model is trained.
        target: list of future predictions of data.
        """
        source, target = [], []
        for i in range(len(seq)):
            end_idx = i + steps
            if end_idx > len(seq) - 1:
                break
            seq_src, seq_tgt = seq[i:end_idx], seq[end_idx]
            source.append(seq_src)
            target.append(seq_tgt)
        return source, target

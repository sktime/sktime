# sktime/base/adapters/_tsai.py
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import (
    from_multi_index_to_3d_numpy,
    from_nested_to_3d_numpy,
)

__author__ = ["timeseriesAI (oguiza)", "estiern", "AlexThomasMa"]


def _to_3d_numpy(X):
    """Return (N, C, L) array from any sktime panel format (or already-3D ndarray)."""
    if isinstance(X, np.ndarray) and X.ndim == 3:  # already 3-D
        return X
    if isinstance(X.index, pd.MultiIndex):  # long/MI format
        return from_multi_index_to_3d_numpy(X)
    return from_nested_to_3d_numpy(X)  # nested DataFrame


class _TsaiAdapter(BaseClassifier):
    """Common adapter to expose tsai deep-learning models as sktime classifiers."""

    _tags = {
        "scitype:instancewise": True,
        "requires_y": True,
        "handles-tabular_X": False,
        "capability:multivariate": True,
        "capability:multioutput": True,  #  ← NEW: declare native multi-output support
        "python_dependencies": ["tsai", "torch"],
    }

    # --------------------------------------------------------------------- #
    #                              constructor                              #
    # --------------------------------------------------------------------- #
    def __init__(self, tsai_model_cls, n_epochs=5, bs=64, lr=1e-3, **fit_kwargs):
        self.tsai_model_cls = tsai_model_cls
        self.n_epochs = n_epochs
        self.bs = bs
        self.lr = lr
        self.fit_kwargs = fit_kwargs

        self._learn = None  # fastai Learner
        self._classes = None  # np.ndarray of class labels
        self._multioutput_columns = None  # remember y-DataFrame columns

        super().__init__()

    # --------------------------------------------------------------------- #
    #                                   fit                                 #
    # --------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame):
        # delay heavy deps
        import torch
        from tsai.all import get_ts_dls, ts_learner

        # --- handle multi-output y: train on first column, remember the rest
        if isinstance(y, pd.DataFrame):
            self._multioutput_columns = list(y.columns)
            y_train = y.iloc[:, 0]
        else:
            y_train = y

        # custom loss that always casts targets to long
        _ce = torch.nn.CrossEntropyLoss()

        def _loss(pred, targ):
            return _ce(pred, targ.long())

        # 1) X → (N, C, L)
        X_np = _to_3d_numpy(X)

        # 2) encode class labels
        classes, y_int = np.unique(y_train, return_inverse=True)
        self._classes = classes
        y_int = y_int.astype(np.int64)

        if len(classes) == 1:  # single-class warning for ensemble compatibility
            warnings.warn(
                "Only a single label present in training data. This may cause "
                "downstream errors.",
                UserWarning,
            )

        # 3) build DataLoaders
        dls = get_ts_dls(X_np, y_int, bs=self.bs, classification=True)

        # 4) Learner
        model = self.tsai_model_cls(dls.vars, len(classes), dls.len)
        self._learn = ts_learner(dls, model, loss_func=_loss)
        self._learn.fit_one_cycle(self.n_epochs, self.lr, **self.fit_kwargs)
        return self

    # --------------------------------------------------------------------- #
    #                           inference helpers                           #
    # --------------------------------------------------------------------- #
    def _forward(self, X_np: np.ndarray):
        """Run a batch through the tsai model and return soft-max probabilities."""
        import torch

        xb = torch.from_numpy(X_np).to(self._learn.dls.device).float()
        mdl = self._learn.model.eval()
        with torch.no_grad():
            logits = mdl(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    # --------------------------------------------------------------------- #
    #                               predict                                 #
    # --------------------------------------------------------------------- #
    def predict(self, X: pd.DataFrame):
        X_np = _to_3d_numpy(X)
        probs = self._forward(X_np)
        preds = self._classes[probs.argmax(axis=1)]

        # broadcast to all y columns if multi-output
        if self._multioutput_columns is not None:
            return pd.DataFrame(
                dict.fromkeys(self._multioutput_columns, preds),
                index=X.index,
            )
        return preds

    def predict_proba(self, X: pd.DataFrame):
        X_np = _to_3d_numpy(X)
        return self._forward(X_np)

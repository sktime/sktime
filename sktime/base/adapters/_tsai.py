# sktime/base/adapters/_tsai.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy

__author__ = ["timeseriesAI (oguiza)", "estiern", "AlexThomasMa"]


# -------------------------------------------------------------------------
# custom loss: CrossEntropy that always sees integer targets
# -------------------------------------------------------------------------
_ce = torch.nn.CrossEntropyLoss()

def _loss(pred, targ):
    """Wrapper around torch.nn.CrossEntropyLoss that casts targets to long."""
    return _ce(pred, targ.long())
# -------------------------------------------------------------------------


class _TsaiAdapter(BaseClassifier):
    """Shared adapter logic to wrap any tsai model class as a sktime estimator."""

    _tags = {
        "scitype:instancewise": True,
        "requires_y": True,
        "handles-tabular_X": False,
        "python_dependencies": ["tsai", "torch"],
    }

    def __init__(self, tsai_model_cls, n_epochs=5, bs=64, lr=1e-3, **fit_kwargs):
        self.tsai_model_cls = tsai_model_cls
        self.n_epochs = n_epochs
        self.bs = bs
        self.lr = lr
        self.fit_kwargs = fit_kwargs

        self._learn = None          # fastai Learner (tsai learner)
        self._classes = None        # np.ndarray of class labels

        super().__init__()

    # ---------------------- Fit logic ---------------------- #
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Delay-import tsai until fit time, so sktime can still import
        from tsai.all import get_ts_dls, ts_learner

        # 1) Convert nested DataFrame -> 3‐D NumPy array (N, C, L)
        X_np = from_nested_to_3d_numpy(X)

        # 2) Encode y labels to integer indices
        classes, y_int = np.unique(y, return_inverse=True)
        self._classes = classes
        y_int = y_int.astype(np.int64)  # CrossEntropyLoss expects int64

        # 3) Build tsai DataLoaders
        dls = get_ts_dls(X_np, y_int, bs=self.bs, classification=True)

        # 4) Instantiate tsai model & fastai Learner
        n_classes = len(self._classes)
        model = self.tsai_model_cls(dls.vars, n_classes, dls.len)
        self._learn = ts_learner(dls, model, loss_func=_loss)
        self._learn.fit_one_cycle(self.n_epochs, self.lr, **self.fit_kwargs)
        return self

    # ---------------------- Inference / Predict logic ---------------------- #
    def _forward(self, X_np: np.ndarray):
        """
        Run a batch (N,C,L) through the tsai model and return
        soft-max probabilities, avoiding fastai’s predict path.
        """
        xb = torch.from_numpy(X_np).to(self._learn.dls.device).float()  # (N,C,L)
        mdl = self._learn.model.eval()
        with torch.no_grad():
            logits = mdl(xb)                                       # (N, n_classes)
            probs = torch.softmax(logits, dim=1).cpu().numpy()      # (N, n_classes)
        return probs

    def predict(self, X: pd.DataFrame):
        X_np = from_nested_to_3d_numpy(X)     # (N, C, L)
        probs = self._forward(X_np)           # (N, n_classes)
        idx = probs.argmax(axis=1)            # (N,)
        return self._classes[idx]             # map back to original labels

    def predict_proba(self, X: pd.DataFrame):
        X_np = from_nested_to_3d_numpy(X)
        return self._forward(X_np)
